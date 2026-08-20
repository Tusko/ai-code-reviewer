import logging
import re
import time
from dataclasses import dataclass
from typing import Sequence

from reviewer import config, gitlab_client, openrouter_client, prompt as prompt_mod
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.filters import is_reviewable
from reviewer.memes import snark
from reviewer.ollama_client import ChatResult, chat
from reviewer.queue import DedupeCache

VOICE_DEADLINE_S = 20
FINDING_TAGS = ("[BLOCKER]", "[SUGGESTION]", "[NIT]")
FENCE_BODY_RE = re.compile(r"```(?:\w*)\n?(.*?)```", re.DOTALL)


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


def review_file(mr, file_diff: FileDiff, context: str) -> FileOutcome:
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
        if result.text and not result.text.startswith("LGTM."):
            finding = result.text
            if result.done_reason != "length":
                finding = flavor_review(finding)
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


def _fence_bodies(text: str) -> list[str]:
    return FENCE_BODY_RE.findall(text)


def preserves_findings(original: str, flavored: str) -> bool:
    """True when the rewrite kept every finding tag and every Fix code block."""
    for tag in FINDING_TAGS:
        if original.count(tag) != flavored.count(tag):
            return False
    original_fences = _fence_bodies(original)
    return not original_fences or _fence_bodies(flavored) == original_fences


def flavor_review(text: str) -> str:
    """Rewrite a dry finding as Sidorovich. Voice is extra: never block or replace."""
    if not config.SNARK or not config.OPENROUTER_API_KEY:
        return text
    result = prefer_ukrainian(
        openrouter_client.chat(
            prompt_mod.SIDOROVICH_REVIEW_VOICE_PROMPT,
            text,
            deadline_s=VOICE_DEADLINE_S,
            temperature=0.8,
        ),
        lambda: openrouter_client.chat(
            prompt_mod.SIDOROVICH_REVIEW_VOICE_PROMPT,
            text + "\n\n" + prompt_mod.SIDOROVICH_UKRAINIAN_RETRY,
            deadline_s=VOICE_DEADLINE_S,
            temperature=0.8,
        ),
    )
    flavored = (result.text or "").strip()
    if result.failed or result.done_reason == "length" or not flavored:
        logging.warning(
            "Sidorovich voice rewrite skipped (%s); posting dry review",
            result.done_reason,
        )
        return text
    if not preserves_findings(text, flavored):
        logging.warning(
            "Sidorovich voice rewrite changed findings; posting dry review",
        )
        return text
    if prompt_mod.looks_too_russian(flavored):
        logging.warning(
            "Sidorovich voice still Russian after retry; posting dry review",
        )
        return text
    return flavored


def prefer_ukrainian(result: ChatResult, retry) -> ChatResult:
    """One retry when Sidorovich slipped into Russian."""
    if result.failed or not prompt_mod.looks_too_russian(result.text):
        return result
    logging.warning("Sidorovich wrote Russian; retrying in Ukrainian")
    retried = retry()
    if retried.failed or not (retried.text or "").strip():
        return result
    return retried


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


def summary_chat(system: str, user: str, deadline_s: int) -> ChatResult:
    """Sidorovich summaries: OpenRouter first, local Ollama as fallback."""
    if config.OPENROUTER_API_KEY:
        result = openrouter_client.chat(
            system, user, deadline_s, temperature=1.0,
        )
        if not result.failed and result.text.strip():
            return result
        logging.warning(
            "OpenRouter summary failed (%s); falling back to Ollama %s",
            result.done_reason, config.OLLAMA_MODEL,
        )
    return chat(system, user, deadline_s, temperature=0.95, seed=None)


def summarize_release_mr(project_id: int, mr_iid: int, mr, force: bool) -> None:
    """Skip full review; post a Sidorovich commit-list roast instead."""
    commits = gitlab_client.fetch_commits(mr)
    fingerprint = gitlab_client.commit_fingerprint(project_id, mr_iid, commits)
    if not force and dedupe.seen(fingerprint):
        logging.info("MR !%s release/hotfix commits unchanged; skipping", mr_iid)
        return

    if not commits:
        logging.info("MR !%s source branch is release/ or hotfix/; no commits to summarise", mr_iid)
        return

    user = prompt_mod.build_commit_summary_prompt(commits)
    result = prefer_ukrainian(
        summary_chat(
            prompt_mod.SIDOROVICH_SYSTEM_PROMPT,
            user,
            deadline_s=config.PER_FILE_TIMEOUT_S,
        ),
        lambda: summary_chat(
            prompt_mod.SIDOROVICH_SYSTEM_PROMPT,
            user + "\n\n" + prompt_mod.SIDOROVICH_UKRAINIAN_RETRY,
            deadline_s=config.PER_FILE_TIMEOUT_S,
        ),
    )
    if result.failed or not result.text.strip():
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
                outcomes.append(review_file(mr, file_diff, context))
            except Exception as exc:
                logging.error("Review failed for %s: %s", file_diff.new_path, exc)
                outcomes.append(FileOutcome(file_diff.new_path, "error", str(exc)[:120]))

        gitlab_client.post_note(mr, render_summary(outcomes))
        dedupe.remember(fingerprint)
        logging.info("MR !%s reviewed in %.1fs", mr_iid, time.monotonic() - started)

    except Exception as exc:
        logging.error("Critical error reviewing MR !%s: %s", mr_iid, exc)
