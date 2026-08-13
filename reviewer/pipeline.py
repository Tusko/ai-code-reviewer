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

SKIP_BRANCH_PREFIXES = ("release/",)


def should_skip_branch(branch: str) -> bool:
    return any(prefix in (branch or "") for prefix in SKIP_BRANCH_PREFIXES)


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

    bodies: list[str] = []
    truncated = False
    for text in prompts:
        result = chat(prompt_mod.SYSTEM_PROMPT, text, deadline_s=config.PER_FILE_TIMEOUT_S)
        if result.failed:
            return FileOutcome(path, "error", result.done_reason)
        if result.done_reason == "timeout":
            return FileOutcome(path, "error", f"timeout after {config.PER_FILE_TIMEOUT_S}s")
        if result.done_reason == "length":
            truncated = True
        if result.text and not result.text.startswith("LGTM."):
            bodies.append(result.text)

    if not bodies:
        return FileOutcome(path, "clean", "")

    body = f"### 📄 `{path}`\n\n" + "\n\n".join(bodies)
    if truncated:
        body += "\n\n_⚠️ This review was truncated at the output token limit and may be incomplete._"
    anchor = file_diff.hunks[0].first_added_line()
    posted = False
    if anchor is not None:
        posted = gitlab_client.post_inline(mr, path, anchor, body)
    if not posted:
        gitlab_client.post_note(mr, body)

    detail = f"{len(bodies)} response(s)"
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


def review_merge_request(project_id: int, mr_iid: int) -> None:
    started = time.monotonic()
    try:
        project, mr = gitlab_client.fetch_mr(project_id, mr_iid)

        if should_skip_branch(getattr(mr, "source_branch", "")):
            logging.info("MR !%s targets a release branch; skipping", mr_iid)
            return

        file_diffs = gitlab_client.fetch_file_diffs(mr)

        fingerprint = gitlab_client.diff_fingerprint(file_diffs)
        if dedupe.seen(fingerprint):
            logging.info("MR !%s diff unchanged since last review; skipping", mr_iid)
            return
        dedupe.remember(fingerprint)

        kept, outcomes = select_files(file_diffs)

        if not kept:
            logging.info("MR !%s: nothing reviewable", mr_iid)
            gitlab_client.post_note(mr, render_summary(outcomes))
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
        logging.info("MR !%s reviewed in %.1fs", mr_iid, time.monotonic() - started)

    except Exception as exc:
        logging.error("Critical error reviewing MR !%s: %s", mr_iid, exc)
