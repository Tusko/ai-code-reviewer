import logging
import time
from dataclasses import dataclass
from typing import Sequence

from reviewer import config, gitlab_client, openrouter_client, prompt as prompt_mod
from reviewer.chat_types import FINDING_TAGS, LGTM_TEXT, ChatResult
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.filters import is_reviewable
from reviewer.memes import snark
from reviewer.ollama_client import chat
from reviewer.queue import DedupeCache
from reviewer.voice import VoiceState, flavor_review, prefer_ukrainian


@dataclass(frozen=True)
class FileOutcome:
    path: str
    status: str   # "reviewed" | "clean" | "skipped" | "error"
    detail: str


dedupe = DedupeCache()

SKIP_BRANCH_PREFIXES = ("release/", "hotfix/")


def should_skip_branch(branch: str) -> bool:
    branch = branch or ""
    return any(branch.startswith(prefix) for prefix in SKIP_BRANCH_PREFIXES)


def select_files(file_diffs: Sequence[FileDiff]) -> tuple[list[FileDiff], list[FileOutcome]]:
    kept: list[FileDiff] = []
    outcomes: list[FileOutcome] = []

    for fd in file_diffs:
        ok, reason = is_reviewable(fd)
        if ok:
            kept.append(fd)
        else:
            outcomes.append(FileOutcome(fd.new_path, "skipped", reason))

    kept.sort(key=lambda f: f.total_lines)
    if len(kept) > config.MAX_FILES:
        for fd in kept[config.MAX_FILES:]:
            outcomes.append(FileOutcome(fd.new_path, "skipped", "over MAX_FILES limit"))
        kept = kept[: config.MAX_FILES]

    return kept, outcomes


# With a 262k-token review context essentially every file fits at L1, so the L2
# per-hunk rung no longer fires in practice. It is kept because it is what
# guarantees no file is silently dropped for being too large.
def build_prompt_ladder(path: str, hunks: Sequence[Hunk], context: str) -> list[tuple[str, str]]:
    """Ordered attempts, cheapest-viable first. L3 is the absence of any fitting level."""
    ladder: list[tuple[str, str]] = []
    if config.INCLUDE_FILE_CONTEXT and context:
        ladder.append(("L0", prompt_mod.build_file_prompt(path, hunks, context)))
    ladder.append(("L1", prompt_mod.build_file_prompt(path, hunks)))
    if len(hunks) > 1:
        for hunk in hunks:
            ladder.append(("L2", prompt_mod.build_file_prompt(path, [hunk])))
    return ladder


def review_file(
    mr, file_diff: FileDiff, context: str, voice: "VoiceState | None" = None,
) -> FileOutcome:
    path = file_diff.new_path
    ladder = build_prompt_ladder(path, file_diff.hunks, context)
    attempts = [(level, text) for level, text in ladder if prompt_mod.fits(text)]

    if not attempts:
        return FileOutcome(path, "skipped", "single hunk exceeds context budget")

    # Take the first fitting level; if it is L2, take every L2 entry that fits.
    chosen_level = attempts[0][0]
    prompts = [text for level, text in attempts if level == chosen_level]

    # L2 hunk-prompts can individually fail to fit; the ladder's guarantee is
    # that no level is silent, so any dropped hunks must be named here.
    l2_detail = ""
    if chosen_level == "L2":
        total_l2 = sum(1 for level, _ in ladder if level == "L2")
        fit_l2 = len(prompts)
        missing = total_l2 - fit_l2
        if missing:
            hunk_word = "hunk" if missing == 1 else "hunks"
            verb = "exceeds" if missing == 1 else "exceed"
            l2_detail = (
                f"{fit_l2} of {total_l2} hunks reviewed; "
                f"{missing} {hunk_word} {verb} context budget"
            )

    bodies: list[str] = []
    truncated = False
    for text in prompts:
        result = chat(prompt_mod.SYSTEM_PROMPT, text, deadline_s=config.PER_FILE_TIMEOUT_S)
        if result.failed:
            return FileOutcome(path, "error", result.done_reason)
        if result.done_reason == "timeout":
            return FileOutcome(path, "error", f"timeout after {config.PER_FILE_TIMEOUT_S}s")
        if result.done_reason == "incomplete":
            # Stream ended with no terminal payload. An absence of signal must
            # never render as a positive (clean) result.
            return FileOutcome(path, "error", "no response from model")
        if not result.text and result.done_reason == "stop":
            return FileOutcome(path, "error", "no response from model")
        if result.done_reason == "length":
            truncated = True
        if result.text and not is_lgtm(result.text):
            finding = result.text
            if result.done_reason != "length":
                finding = flavor_review(finding, voice)
            bodies.append(finding)

    if not bodies:
        return FileOutcome(path, "clean", l2_detail)

    body = f"### 📄 `{path}`\n\n" + "\n\n".join(bodies)
    if truncated:
        body += "\n\n_⚠️ This review was truncated at the output token limit and may be incomplete._"
    anchor = file_diff.hunks[0].first_added_line()
    posted = False
    if anchor is not None:
        posted = gitlab_client.post_inline(mr, path, anchor, body)
    if not posted:
        gitlab_client.post_note(mr, body)

    detail = l2_detail if l2_detail else f"{len(bodies)} response(s)"
    if truncated:
        detail += ", truncated at output token limit"
    return FileOutcome(path, "reviewed", detail)


def is_lgtm(text: str) -> bool:
    """True for a clean verdict in any of the shapes the model emits."""
    stripped = (text or "").strip()
    if any(tag in stripped for tag in FINDING_TAGS):
        return False
    return stripped.startswith(LGTM_TEXT) or stripped.upper().startswith("LGTM")


def render_summary(outcomes: Sequence[FileOutcome]) -> str:
    reviewed = [o for o in outcomes if o.status == "reviewed"]
    clean = [o for o in outcomes if o.status == "clean"]
    skipped = [o for o in outcomes if o.status == "skipped"]
    errored = [o for o in outcomes if o.status == "error"]

    lines = []
    if config.SNARK:
        lines.append(f"\n_{snark()}_")

    if reviewed:
        lines.append(f"\n**Findings on {len(reviewed)} file(s):** "
                     + ", ".join(f"`{o.path}`" for o in reviewed))
    if clean and not reviewed and not errored:
        lines.append("\nLGTM. No logic or security issues found in the changed lines.")
    elif clean:
        lines.append(f"\n**Clean:** {len(clean)} file(s)")
    if skipped:
        lines.append("\n**Skipped:**")
        lines.extend(f"- `{o.path}` — {o.detail}" for o in skipped)
    if errored:
        lines.append("\n**Errors:**")
        lines.extend(f"- `{o.path}` — {o.detail}" for o in errored)

    return "\n".join(lines)


def render_commit_digest(commits: Sequence[dict]) -> str:
    """Plain commit list, posted when Sidorovich has no model that can voice it."""
    lines = ["**Release/hotfix — full review skipped.** Commits:"]
    for commit in list(commits)[:prompt_mod.MAX_COMMITS_IN_PROMPT]:
        title = (commit.get("title") or "").strip() or "(no message)"
        author = commit.get("author") or "unknown"
        lines.append(f"- {title} — _{author}_")
    omitted = len(commits) - min(len(commits), prompt_mod.MAX_COMMITS_IN_PROMPT)
    if omitted > 0:
        lines.append(f"- …and {omitted} more commit(s)")
    return "\n".join(lines)


def summary_chat(
    system: str, user: str, deadline_s: int, history: Sequence[dict] = (),
) -> ChatResult:
    """Sidorovich summaries: OpenRouter first, local Ollama only when allowed."""
    if config.OPENROUTER_API_KEY:
        result = openrouter_client.chat(
            system, user, deadline_s, temperature=1.0, history=history,
        )
        if not result.failed and result.text.strip():
            return result
        logging.warning(
            "OpenRouter summary failed (%s)", result.done_reason,
        )
        if result.done_reason == "ratelimit" and not config.SIDOROVICH_OLLAMA_FALLBACK:
            # Transient, not "this deployment has no voice model". Keep the
            # ratelimit reason so the caller retries instead of deduping the MR.
            return result
    if not config.SIDOROVICH_OLLAMA_FALLBACK:
        # A code model writing Ukrainian surzhyk produces gibberish under
        # Sidorovich's name. Report the voice as unavailable instead.
        return ChatResult(
            text="",
            done_reason="unavailable",
            prompt_eval_count=0,
            eval_count=0,
            elapsed_s=0.0,
        )
    logging.warning("Falling back to Ollama %s for the summary", config.OLLAMA_MODEL)
    return chat(
        system, user, deadline_s, temperature=0.95, seed=None, history=history,
    )


def summarize_release_mr(project_id: int, mr_iid: int, mr, force: bool) -> None:
    """Skip full review; post a Sidorovich commit-list roast instead."""
    commits = gitlab_client.fetch_commits(mr)
    if not commits:
        logging.info("MR !%s source branch is release/ or hotfix/; no commits to summarise", mr_iid)
        return

    fingerprint = gitlab_client.commit_fingerprint(project_id, mr_iid, commits)
    if not force and dedupe.seen(fingerprint):
        logging.info("MR !%s release/hotfix commits unchanged; skipping", mr_iid)
        return

    user = prompt_mod.build_commit_summary_prompt(commits)
    result = prefer_ukrainian(
        summary_chat(
            prompt_mod.SIDOROVICH_SYSTEM_PROMPT,
            user,
            deadline_s=config.PER_FILE_TIMEOUT_S,
        ),
        lambda bad: summary_chat(
            prompt_mod.SIDOROVICH_SYSTEM_PROMPT,
            prompt_mod.SIDOROVICH_UKRAINIAN_RETRY,
            deadline_s=config.PER_FILE_TIMEOUT_S,
            history=(
                {"role": "user", "content": user},
                {"role": "assistant", "content": bad},
            ),
        ),
    )
    if result.done_reason == "unavailable":
        # Not a transient failure — no model here can do the voice. Post the
        # plain digest and dedupe it, rather than retrying on every webhook.
        logging.info("MR !%s: no Sidorovich voice available; posting commit digest", mr_iid)
        gitlab_client.post_note(mr, render_commit_digest(commits))
        dedupe.remember(fingerprint)
        return
    if result.failed or not result.text.strip():
        # Nothing is deduped here: the next webhook for this MR retries.
        logging.error(
            "Sidorovich summary failed for MR !%s: %s", mr_iid, result.done_reason,
        )
        return

    gitlab_client.post_note(mr, result.text.strip())
    dedupe.remember(fingerprint)
    logging.info("MR !%s release/hotfix summarised (%s commits)", mr_iid, len(commits))


def review_merge_request(project_id: int, mr_iid: int, force: bool = False) -> None:
    started = time.monotonic()
    try:
        project, mr = gitlab_client.fetch_mr(project_id, mr_iid)

        if should_skip_branch(getattr(mr, "source_branch", "")):
            summarize_release_mr(project_id, mr_iid, mr, force)
            return

        file_diffs = gitlab_client.fetch_file_diffs(mr)

        fingerprint = gitlab_client.diff_fingerprint(project_id, mr_iid, file_diffs)
        if not force and dedupe.seen(fingerprint):
            logging.info("MR !%s diff unchanged since last review; skipping", mr_iid)
            return

        kept, outcomes = select_files(file_diffs)
        voice = VoiceState()

        if not kept:
            logging.info("MR !%s: nothing reviewable", mr_iid)
            gitlab_client.post_note(mr, render_summary(outcomes))
            dedupe.remember(fingerprint)
            return

        for file_diff in kept:
            if time.monotonic() - started > config.MR_TIMEOUT_S:
                outcomes.append(FileOutcome(
                    file_diff.new_path, "skipped",
                    f"MR deadline of {config.MR_TIMEOUT_S}s reached",
                ))
                continue

            context = ""
            if config.INCLUDE_FILE_CONTEXT:
                content = gitlab_client.fetch_file_content(
                    project, file_diff.new_path, mr.source_branch,
                )
                if content:
                    context = gitlab_client.surgical_context(
                        content, file_diff.hunks, config.CONTEXT_WINDOW,
                    )

            try:
                outcomes.append(review_file(mr, file_diff, context, voice))
            except Exception as exc:
                logging.error("Review failed for %s: %s", file_diff.new_path, exc)
                outcomes.append(FileOutcome(file_diff.new_path, "error", str(exc)[:120]))

        gitlab_client.post_note(mr, render_summary(outcomes))
        dedupe.remember(fingerprint)
        logging.info("MR !%s reviewed in %.1fs", mr_iid, time.monotonic() - started)

    except Exception as exc:
        logging.error("Critical error reviewing MR !%s: %s", mr_iid, exc)
