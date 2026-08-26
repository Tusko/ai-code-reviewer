import logging
import time
from dataclasses import dataclass, replace
from typing import Sequence

from reviewer import (
    config, gitlab_client, ollama_client, openrouter_client, prompt as prompt_mod,
)
from reviewer.chat_types import FINDING_TAGS, LGTM_TEXT, ChatResult
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.filters import is_reviewable
from reviewer.ledger import Ledger, LedgerStore, LedgerUnavailable, hunk_key
from reviewer.memes import closer_for, opener_for, snark
from reviewer.openrouter_client import review_chat
from reviewer.queue import DedupeCache
from reviewer.voice import VoiceState, flavor_review, prefer_ukrainian

# Consecutive rate-limited review calls before the run gives up. On the paid
# tier this should never fire; it exists because OPENROUTER_REVIEW_MODEL is a
# knob and someone will eventually point it at a `:free` model.
REVIEW_FAILURE_LIMIT = 2


class ReviewState:
    """Per-MR circuit breaker for rate-limited review calls."""

    def __init__(self) -> None:
        self.ratelimits = 0

    @property
    def open(self) -> bool:
        return self.ratelimits < REVIEW_FAILURE_LIMIT

    def record(self, done_reason: str) -> None:
        # Only rate limits. "error" is per-file and content-dependent -- a 400
        # on one oversized payload, a moderation refusal -- and counting it
        # meant two unlucky files in a row stopped every later run of that MR
        # for good, since select_files sorts deterministically.
        if done_reason == "ratelimit":
            self.ratelimits += 1
            if not self.open:
                logging.warning(
                    "Review stopped for this MR after %s consecutive rate limits",
                    self.ratelimits,
                )
        else:
            self.ratelimits = 0


@dataclass(frozen=True)
class FileOutcome:
    path: str
    status: str   # "reviewed" | "clean" | "skipped" | "error"
    detail: str
    # False when the comment may never have reached GitLab. Such a file keeps
    # its budget slot (it may have posted) but stays out of the ledger, so the
    # next push retries it instead of losing the finding forever.
    settled: bool = True


dedupe = DedupeCache()

SKIP_BRANCH_PREFIXES = ("release/", "hotfix/")


def should_skip_branch(branch: str) -> bool:
    branch = branch or ""
    return any(branch.startswith(prefix) for prefix in SKIP_BRANCH_PREFIXES)


def partition_reviewable(
    file_diffs: Sequence[FileDiff],
) -> tuple[list[FileDiff], list[FileOutcome]]:
    """Splits diffs into reviewable files and named skip outcomes."""
    kept: list[FileDiff] = []
    outcomes: list[FileOutcome] = []
    for fd in file_diffs:
        ok, reason = is_reviewable(fd)
        if ok:
            kept.append(fd)
        else:
            outcomes.append(FileOutcome(fd.new_path, "skipped", reason))
    return kept, outcomes


def select_files(
    reviewable: Sequence[FileDiff], skipped: Sequence[FileOutcome] = (),
) -> tuple[list[FileDiff], list[FileOutcome]]:
    """Applies the per-run MAX_FILES cap, smallest files first."""
    outcomes = list(skipped)
    kept = sorted(reviewable, key=lambda f: f.total_lines)
    if len(kept) > config.MAX_FILES:
        for fd in kept[config.MAX_FILES:]:
            outcomes.append(FileOutcome(fd.new_path, "skipped", "over MAX_FILES limit"))
        kept = kept[: config.MAX_FILES]
    return kept, outcomes


def drop_known_hunks(
    file_diffs: Sequence[FileDiff], value: "Ledger",
) -> list[FileDiff]:
    """Returns the diffs with already-reviewed hunks removed.

    A file whose every hunk is known disappears from the result entirely.
    """
    if value.saturated:
        # The ledger stopped being able to name what it has already reviewed.
        # Treating the untracked remainder as unseen is what let a /review
        # replay the whole merge request, so treat it as seen and go quiet.
        logging.error(
            "Ledger is saturated at LEDGER_MAX_HUNKS=%s; nothing further will "
            "be reviewed on this MR", config.LEDGER_MAX_HUNKS,
        )
        return []
    known = set(value.hunks)
    fresh: list[FileDiff] = []
    for fd in file_diffs:
        hunks = tuple(h for h in fd.hunks if hunk_key(fd.new_path, h) not in known)
        if hunks:
            fresh.append(replace(fd, hunks=hunks))
    return fresh


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
    mr,
    file_diff: FileDiff,
    context: str,
    voice: "VoiceState | None" = None,
    review_state: "ReviewState | None" = None,
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
        result = review_chat(
            prompt_mod.SYSTEM_PROMPT, text, deadline_s=config.PER_FILE_TIMEOUT_S,
        )
        if review_state is not None:
            review_state.record(result.done_reason)
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
    try:
        posted = False
        if anchor is not None:
            posted = gitlab_client.post_inline(mr, path, anchor, body)
        if not posted:
            gitlab_client.post_note(mr, body)
    except Exception as exc:
        # post_inline only catches GitlabError; a connection reset while reading
        # the response of a discussion GitLab already created escapes it. The
        # comment may well be public, so this counts as reviewed and keeps its
        # slot charged. Paying twice for one comment is survivable; posting one
        # for free is how the ceiling silently disengages.
        logging.error("Post failed for %s (the comment may exist anyway): %s", path, exc)
        return FileOutcome(
            path, "reviewed", f"could not be posted ({exc})", settled=False,
        )

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

    delivered = [o for o in reviewed if o.settled]
    undelivered = [o for o in reviewed if not o.settled]

    if delivered:
        lines.append(f"\n**Findings on {len(delivered)} file(s):** "
                     + ", ".join(f"`{o.path}`" for o in delivered))
    if undelivered:
        lines.append("\n**Findings that could not be posted:**")
        lines.extend(f"- `{o.path}` — {o.detail}" for o in undelivered)
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


def render_oversized(count: int) -> str:
    """The single comment an over-threshold MR receives."""
    lines = []
    if config.SNARK:
        lines.append(f"_{snark()}_\n")
    lines.append(
        f"**MR завеликий: {count} файлів до рев'ю, поріг — {config.MAX_MR_FILES}.**\n"
    )
    lines.append(
        "Пофайлове рев'ю пропущено. На такому обсязі воно дає сотні коментарів "
        "і нуль користі — розбий MR на менші або рев'юйте руками.\n"
    )
    lines.append("_Це єдиний коментар, який я лишу в цьому MR._")
    return "\n".join(lines)


def render_budget_exhausted(outcomes: Sequence[FileOutcome] = ()) -> str:
    """The closing comment when an MR has used its whole budget.

    `outcomes` is passed so a run whose comments never reached GitLab says so.
    This note is the only one such a run produces — the budget branch runs
    before the summary — so without it the reader is told thirty comments were
    spent and shown none of them.
    """
    lines = []
    if config.SNARK:
        lines.append(f"_{snark()}_\n")
    lines.append(
        f"**Ліміт вичерпано: {config.MR_COMMENT_BUDGET} коментарів у цьому MR.**\n"
    )
    undelivered = [o for o in outcomes if o.status == "reviewed" and not o.settled]
    if undelivered:
        lines.append("Частина з них до GitLab не долетіла:\n")
        lines.extend(f"- `{o.path}` — {o.detail}" for o in undelivered)
        lines.append("")
    if undelivered and len(undelivered) == len(
        [o for o in outcomes if o.status == "reviewed"]
    ):
        # Nothing this run wrote actually landed. Telling the reader to spend a
        # /review here is telling them to replay a merge request that produced
        # no visible comments at all.
        lines.append(
            "Дивитись поки нема на що. Полагодь звʼязок із GitLab, а тоді вже "
            "кидай `/review`.\n"
        )
    else:
        lines.append(
            "Далі мовчу до мержу. Розгреби те, що вже написав, а тоді кинь "
            "`/review` у коментар — лічильник обнулиться.\n"
        )
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
    return ollama_client.chat(
        system, user, deadline_s, temperature=0.95, seed=None, history=history,
    )


def _charge(store, mr_iid: int, head: str = None) -> bool:
    """Persists the cost of a comment BEFORE it is posted. Returns whether it
    is safe to post.

    Charging afterwards is what makes a failing save unbounded: the comment
    goes out, `posted` never reaches GitLab, and every later push starts from
    zero with no ceiling at all. Charging first turns that same failure into
    "one comment too few", which is the direction we can afford to be wrong in.
    """
    store.ledger = store.ledger.spend(1)
    if head is not None:
        store.ledger = store.ledger.at_head(head)
    try:
        store.save()
        return True
    except Exception as exc:
        logging.error(
            "MR !%s: review state could not be saved (%s); posting nothing",
            mr_iid, exc,
        )
        return False


def _save_quietly(store, mr_iid: int) -> bool:
    """Best-effort save for bookkeeping. Returns whether it stuck."""
    try:
        store.save()
        return True
    except Exception as exc:
        logging.error("MR !%s: review state could not be saved: %s", mr_iid, exc)
        return False


def render_saturated(tracked: int) -> str:
    """Said once per transition into saturation, not once per push.

    Going quiet is the right failure; going quiet without saying so is not.
    Before this the bot answered /review with "I have seen everything, push
    something new" while blind to brand new files.
    """
    lines = []
    if config.SNARK:
        lines.append(f"_{snark()}_\n")
    lines.append(
        f"**Цей MR переріс мою памʼять: {tracked} хунків.**\n"
    )
    lines.append(
        "Я більше не можу відрізнити переглянуте від нового, тому далі мовчу. "
        "Поділи MR на менші — інакше я тут марний.\n"
    )
    return "\n".join(lines)


def render_nothing_to_do(reason: str) -> str:
    """Answers a manual /review that found no work. Silence reads as a crash."""
    lines = []
    if config.SNARK:
        lines.append(f"_{snark()}_\n")
    lines.append(f"**Нема чого дивитись.** {reason}\n")
    return "\n".join(lines)


def summarize_release_mr(project_id: int, mr_iid: int, mr, force: bool, store) -> None:
    """Skip full review; post a Sidorovich commit-list roast instead."""
    commits = gitlab_client.fetch_commits(mr)
    if not commits:
        logging.info("MR !%s source branch is release/ or hotfix/; no commits to summarise", mr_iid)
        return

    fingerprint = gitlab_client.commit_fingerprint(project_id, mr_iid, commits)
    if not force and (dedupe.seen(fingerprint) or store.ledger.head == fingerprint):
        # Checked against the ledger too, not just the in-process cache: the
        # cache is empty after every container restart, and this path used to
        # re-post the same roast on each one.
        logging.info("MR !%s release/hotfix commits unchanged; skipping", mr_iid)
        return
    if store.ledger.remaining() <= 0:
        logging.warning(
            "MR !%s: release summary suppressed, MR_COMMENT_BUDGET=%s spent",
            mr_iid, config.MR_COMMENT_BUDGET,
        )
        return

    # Seeded from the fingerprint, so the same commits always get the same
    # opener and two different merge requests almost never share one.
    user = prompt_mod.build_commit_summary_prompt(
        commits,
        opener=opener_for(fingerprint),
        closer=closer_for(fingerprint),
    )
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
        if not _charge(store, mr_iid, head=fingerprint):
            return
        gitlab_client.post_note(mr, render_commit_digest(commits))
        dedupe.remember(fingerprint)
        return
    if result.failed or not result.text.strip():
        # Nothing is deduped here: the next webhook for this MR retries.
        logging.error(
            "Sidorovich summary failed for MR !%s: %s", mr_iid, result.done_reason,
        )
        return

    if not _charge(store, mr_iid, head=fingerprint):
        return
    gitlab_client.post_note(mr, result.text.strip())
    dedupe.remember(fingerprint)
    logging.info("MR !%s release/hotfix summarised (%s commits)", mr_iid, len(commits))


def mute_merge_request(project_id: int, mr_iid: int) -> None:
    """Silences the bot for one MR. Acknowledged by editing the state note.

    Deliberately posts no review comment: a mute that costs a comment defeats
    itself. The state note is the one exception, and only when this is the
    first thing the bot has ever written on the MR.
    """
    try:
        _, mr = gitlab_client.fetch_mr(project_id, mr_iid)
        store = LedgerStore.load(mr)
        store.ledger = store.ledger.mute()
        store.save()
        logging.info("MR !%s muted by /sidorovich stop", mr_iid)
    except Exception as exc:
        logging.error("Could not mute MR !%s: %s", mr_iid, exc)


def review_merge_request(project_id: int, mr_iid: int, force: bool = False) -> None:
    started = time.monotonic()
    try:
        project, mr = gitlab_client.fetch_mr(project_id, mr_iid)

        try:
            store = LedgerStore.load(mr)
        except LedgerUnavailable as exc:
            # An unreadable ledger must never be treated as an empty one: that
            # re-reviews the whole MR, which is the flood this prevents.
            logging.error("MR !%s: review state unreadable (%s); skipping", mr_iid, exc)
            return

        if store.ledger.muted and not force:
            logging.info("MR !%s is muted; skipping", mr_iid)
            return
        if force:
            store.ledger = store.ledger.unmute_and_reset()

        if should_skip_branch(getattr(mr, "source_branch", "")):
            summarize_release_mr(project_id, mr_iid, mr, force, store)
            return

        file_diffs = gitlab_client.fetch_file_diffs(mr)
        head_sha = (getattr(mr, "diff_refs", None) or {}).get("head_sha", "")
        reviewable, skipped = partition_reviewable(file_diffs)

        if len(reviewable) > config.MAX_MR_FILES:
            if not store.ledger.oversized:
                store.ledger = store.ledger.mark_oversized()
                if not _charge(store, mr_iid, head=head_sha):
                    return
                gitlab_client.post_note(mr, render_oversized(len(reviewable)))
                logging.info(
                    "MR !%s: %s reviewable files over MAX_MR_FILES=%s; posted one note",
                    mr_iid, len(reviewable), config.MAX_MR_FILES,
                )
                return
            if force and _charge(store, mr_iid, head=head_sha):
                # A human typed /review and deserves an answer, even a refusal.
                gitlab_client.post_note(mr, render_nothing_to_do(
                    f"{len(reviewable)} файлів — це більше за MAX_MR_FILES="
                    f"{config.MAX_MR_FILES}. Поділи MR."))
                return
            store.ledger = store.ledger.at_head(head_sha)
            _save_quietly(store, mr_iid)
            return

        # Against every file in the diff, not just the reviewable ones: a file
        # that turned non-reviewable this push must not lose its record.
        store.ledger = store.ledger.keep_only({
            hunk_key(fd.new_path, h) for fd in file_diffs for h in fd.hunks
        })

        fresh = drop_known_hunks(reviewable, store.ledger)
        if not fresh:
            # Every hunk has been reviewed already. Say nothing at all: this is
            # what makes a container restart or a no-op push cost zero comments.
            logging.info("MR !%s: no unreviewed hunks; staying silent", mr_iid)
            if force and _charge(store, mr_iid, head=head_sha):
                # render_budget_exhausted() tells the human to type /review.
                # Answering that with nothing at all reads as a dead bot.
                gitlab_client.post_note(mr, render_nothing_to_do(
                    "Цей MR переріс мою памʼять, я вже нічого тут не бачу. "
                    "Поділи його."
                    if store.ledger.saturated
                    else "Усе в цьому MR я вже дивився. Запуш щось нове."))
                return
            store.ledger = store.ledger.at_head(head_sha)
            _save_quietly(store, mr_iid)
            return

        kept, outcomes = select_files(fresh, skipped)
        voice = VoiceState()
        review_state = ReviewState()

        for file_diff in kept:
            if time.monotonic() - started > config.MR_TIMEOUT_S:
                outcomes.append(FileOutcome(
                    file_diff.new_path, "skipped",
                    f"MR deadline of {config.MR_TIMEOUT_S}s reached",
                ))
                continue
            if not review_state.open:
                outcomes.append(FileOutcome(
                    file_diff.new_path, "skipped", "review backend rate limited",
                ))
                continue
            if store.ledger.remaining() <= 1:
                # The last slot is reserved for the closing note, so the run can
                # always tell the reader why it stopped.
                outcomes.append(FileOutcome(
                    file_diff.new_path, "skipped", "MR comment budget reached",
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

            # Charged before review_file posts anything, and refunded below if
            # it turns out nothing was posted. Saved per file, not once at the
            # end: a crash mid-run would otherwise leave comments posted but
            # unrecorded, and the next push would post every one of them again.
            if not _charge(store, mr_iid):
                outcomes.append(FileOutcome(
                    file_diff.new_path, "skipped", "review state could not be saved",
                ))
                break

            try:
                outcome = review_file(mr, file_diff, context, voice, review_state)
            except Exception as exc:
                logging.error("Review failed for %s: %s", file_diff.new_path, exc)
                outcomes.append(FileOutcome(file_diff.new_path, "error", str(exc)[:120]))
                # Safe only because review_file swallows its own post failures:
                # nothing can have been posted by the time we get here, so the
                # slot really is unspent.
                store.ledger = store.ledger.refund()
                # Persisted here, not left to the end of the loop body: the
                # bare continue skipped the save, so the last file's charge
                # stayed on the ledger and thirty outage pushes muted the MR.
                _save_quietly(store, mr_iid)
                continue

            outcomes.append(outcome)
            keys = [hunk_key(file_diff.new_path, h) for h in file_diff.hunks]
            gave_up = False
            if not outcome.settled:
                # One retry, then record it anyway. Retrying for ever looks
                # generous until you notice /review resets the budget: each
                # command replayed the whole merge request, so twenty-one of
                # them posted six hundred comments.
                if store.ledger.already_retried(keys):
                    gave_up = True
                    logging.error(
                        "MR !%s: %s failed to post twice; recording it rather "
                        "than replaying the MR on every /review",
                        mr_iid, file_diff.new_path,
                    )
                else:
                    store.ledger = store.ledger.mark_retried(keys)
            # Deliberately a local, not a rewrite of the outcome: the closing
            # note reads `settled` to tell the human which comments never
            # arrived, and marking them delivered for the ledger's benefit made
            # the note claim they had.
            if outcome.status in ("reviewed", "clean") and (outcome.settled or gave_up):
                # Only settled files are recorded. An errored or rate-limited
                # file must be retried on the next push, so its hunks stay out.
                store.ledger = store.ledger.record(keys)
            if outcome.status != "reviewed":
                store.ledger = store.ledger.refund()
            _save_quietly(store, mr_iid)

        if store.ledger.saturated:
            # Reached at most once per transition without needing a guard: a
            # ledger that was already saturated returns above, at `not fresh`,
            # because drop_known_hunks reports nothing new for it.
            if _charge(store, mr_iid, head=head_sha):
                gitlab_client.post_note(
                    mr, render_saturated(len(store.ledger.hunks)),
                )
            logging.error(
                "MR !%s saturated the ledger at LEDGER_MAX_HUNKS=%s",
                mr_iid, config.LEDGER_MAX_HUNKS,
            )

        settled = any(o.status in ("reviewed", "clean") for o in outcomes)
        if settled and store.ledger.outage_reported:
            # The backend answered again, so the next outage is worth a report.
            store.ledger = store.ledger.clear_outage()
        if store.ledger.remaining() <= 1:
            store.ledger = store.ledger.mute()
            if _charge(store, mr_iid, head=head_sha):
                gitlab_client.post_note(mr, render_budget_exhausted(outcomes))
            logging.warning(
                "MR !%s hit MR_COMMENT_BUDGET=%s; muted until /review",
                mr_iid, config.MR_COMMENT_BUDGET,
            )
        elif not settled:
            # A run that got nothing out of the model must cost nothing. Keying
            # this on the head SHA was wrong: developers push, so a day-long
            # outage spent one slot per push and muted the MR, and a muted MR
            # reviews nothing once the backend comes back. Report it free, once
            # per outage, and record nothing so the next push retries the lot.
            if force or not store.ledger.outage_reported:
                store.ledger = store.ledger.report_outage()
                # Persisted first even though the note is free: if the flag
                # cannot be stored, the note would repeat on every push.
                if _save_quietly(store, mr_iid):
                    gitlab_client.post_note(mr, render_summary(outcomes))
            else:
                logging.error(
                    "MR !%s: still nothing reviewable; already reported", mr_iid,
                )
                _save_quietly(store, mr_iid)
        elif _charge(store, mr_iid, head=head_sha):
            gitlab_client.post_note(mr, render_summary(outcomes))
        logging.info("MR !%s reviewed in %.1fs", mr_iid, time.monotonic() - started)

    except Exception as exc:
        logging.error("Critical error reviewing MR !%s: %s", mr_iid, exc)
