import logging
import time
from dataclasses import dataclass
from typing import Sequence

from reviewer import config, gitlab_client, prompt as prompt_mod
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.filters import is_reviewable
from reviewer.ollama_client import chat
from reviewer.queue import DedupeCache


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
            bodies.append(result.text)

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


def render_summary(outcomes: Sequence[FileOutcome]) -> str:
    reviewed = [o for o in outcomes if o.status == "reviewed"]
    clean = [o for o in outcomes if o.status == "clean"]
    skipped = [o for o in outcomes if o.status == "skipped"]
    errored = [o for o in outcomes if o.status == "error"]

    lines = ["## 🤖 AI Code Review"]

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


def review_merge_request(project_id: int, mr_iid: int, force: bool = False) -> None:
    started = time.monotonic()
    try:
        project, mr = gitlab_client.fetch_mr(project_id, mr_iid)

        if should_skip_branch(getattr(mr, "source_branch", "")):
            logging.info("MR !%s source branch is release/ or hotfix/; skipping", mr_iid)
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
