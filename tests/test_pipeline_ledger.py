import logging

import pytest

from reviewer import config, pipeline
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.ledger import Ledger, hunk_key, parse_marker, render_note
from reviewer.pipeline import drop_known_hunks


def fd(path, added=1, hunks=None):
    lines = [" ctx"] + [f"+line{i}" for i in range(added)]
    return FileDiff(
        old_path=path, new_path=path, is_new=False, is_deleted=False,
        is_renamed=False, is_binary=False,
        hunks=hunks or (Hunk(1, 1, tuple(lines)),),
    )


class FakeMR:
    source_branch = "feature/x"
    diff_refs = {"base_sha": "b", "start_sha": "s", "head_sha": "abc1234"}


class Recorder:
    """Captures every comment the pipeline tries to post."""

    def __init__(self):
        self.notes = []
        self.inline = []


def seed(state, ledger):
    """Pre-loads the ledger the way GitLab would hold it: as a marker."""
    state["note"] = render_note(ledger)
    state["ledger"] = parse_marker(state["note"])


@pytest.fixture
def harness(monkeypatch):
    """Wires review_merge_request to fakes and returns (recorder, state)."""
    recorder = Recorder()
    state = {"ledger": Ledger(), "saves": 0, "note": render_note(Ledger())}
    mr = FakeMR()

    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_mr", lambda pid, iid: (object(), mr),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.post_note",
        lambda mr, body: recorder.notes.append(body),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.post_inline",
        lambda mr, path, line, body: (recorder.inline.append((path, body)), True)[1],
    )

    class FakeStore:
        """Round-trips through the real marker, exactly as GitLab would.

        Holding the Ledger object directly was a trap: dropping a field from
        to_marker left the whole suite green while, in production, the flag
        never persisted and its note repeated on every push.
        """

        def __init__(self):
            self.mr = mr
            self.ledger = parse_marker(state["note"])

        @classmethod
        def load(cls, mr):
            return cls()

        def save(self):
            state["saves"] += 1
            state["note"] = render_note(self.ledger)
            # Read straight back, so a field that does not survive the marker
            # cannot survive the test either.
            state["ledger"] = parse_marker(state["note"])

    monkeypatch.setattr("reviewer.pipeline.LedgerStore", FakeStore)
    monkeypatch.setattr("reviewer.config.SNARK", False)
    return recorder, state


def test_oversized_mr_posts_one_note_and_no_inline(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 3)
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(10)],
    )
    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 1
    assert recorder.inline == []
    assert "10" in recorder.notes[0]
    assert state["ledger"].oversized is True
    assert state["ledger"].posted == 1


def test_oversized_mr_stays_silent_on_the_next_push(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 3)
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(10)],
    )
    pipeline.review_merge_request(1, 1)
    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 1, "the oversized note must not repeat"


def test_drop_known_hunks_removes_seen_hunks():
    first = Hunk(1, 1, (" ctx", "+one"))
    second = Hunk(9, 9, (" ctx", "+two"))
    diff = fd("a.py", hunks=(first, second))
    value = Ledger().record([hunk_key("a.py", first)])
    fresh = drop_known_hunks([diff], value)
    assert len(fresh) == 1
    assert fresh[0].hunks == (second,)


def test_drop_known_hunks_drops_fully_seen_files():
    only = Hunk(1, 1, (" ctx", "+one"))
    value = Ledger().record([hunk_key("a.py", only)])
    assert drop_known_hunks([fd("a.py", hunks=(only,))], value) == []


def test_drop_known_hunks_survives_a_rebase():
    """Same content at a new line number must still count as seen."""
    lines = (" ctx", "+one")
    value = Ledger().record([hunk_key("a.py", Hunk(1, 1, lines))])
    rebased = fd("a.py", hunks=(Hunk(400, 400, lines),))
    assert drop_known_hunks([rebased], value) == []


def test_unchanged_diff_posts_absolutely_nothing(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    diff = fd("a.py")
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: [diff])
    seed(state, Ledger().record([hunk_key("a.py", h) for h in diff.hunks]))
    pipeline.review_merge_request(1, 1)
    assert recorder.notes == []
    assert recorder.inline == []
    assert state["ledger"].head == "abc1234", "the head must still be stamped"


def test_reviewed_hunks_are_recorded(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult(
            "**🔴 [BLOCKER]** boom", "stop", 0, 0, 0.0,
        ),
    )
    diff = fd("a.py")
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: [diff])
    pipeline.review_merge_request(1, 1)
    assert hunk_key("a.py", diff.hunks[0]) in state["ledger"].hunks
    assert state["saves"] >= 2, "the ledger must be saved incrementally"


def _clean_review(monkeypatch):
    """Stubs review_file at the real 5-arg signature, always returning clean."""
    monkeypatch.setattr(
        "reviewer.pipeline.review_file",
        lambda mr, file_diff, context, voice, review_state: pipeline.FileOutcome(
            file_diff.new_path, "clean", "",
        ),
    )


def test_an_unchanged_re_push_costs_nothing(harness, monkeypatch):
    recorder, _ = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    _clean_review(monkeypatch)
    diffs = [fd("a.py")]
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: diffs)

    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 1
    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 1, "an unchanged re-push must post nothing at all"


def test_a_changed_diff_is_reviewed_again(harness, monkeypatch):
    recorder, _ = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    _clean_review(monkeypatch)
    diffs = [fd("a.py", hunks=(Hunk(1, 1, (" ctx", "+one")),))]
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: diffs)

    pipeline.review_merge_request(1, 1)
    diffs[0] = fd("a.py", hunks=(Hunk(1, 1, (" ctx", "+two")),))
    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 2, "new content must earn a fresh review"


def test_budget_exhaustion_posts_one_final_note_and_mutes(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.MR_COMMENT_BUDGET", 3)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult(
            "**🔴 [BLOCKER]** boom", "stop", 0, 0, 0.0,
        ),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(10)],
    )
    pipeline.review_merge_request(1, 1)

    assert len(recorder.inline) == 2, "budget 3 leaves 2 inline plus a final note"
    assert len(recorder.notes) == 1
    assert "ліміт" in recorder.notes[0].lower()
    assert state["ledger"].muted is True
    # The 8 files the budget refused must stay unrecorded, or the /review that
    # lifts the mute would find nothing fresh and their bugs would be lost.
    assert len(state["ledger"].hunks) == 2
    assert state["ledger"].posted == config.MR_COMMENT_BUDGET


def test_the_summary_note_costs_a_slot(harness, monkeypatch):
    """Every comment counts, the run summary included. If the summary were
    free, a push per file would post an unbounded number of them."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.MR_COMMENT_BUDGET", 30)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult("[LGTM]", "stop", 0, 0, 0.0),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs", lambda mr: [fd("a.py")],
    )
    pipeline.review_merge_request(1, 1)
    assert len(recorder.inline) == 1
    assert len(recorder.notes) == 1
    assert state["ledger"].posted == 2, (
        "one inline comment plus the run summary is two slots, not one"
    )


def test_muted_mr_ignores_a_plain_push(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    seed(state, Ledger(muted=True))
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: [fd("a.py")])
    pipeline.review_merge_request(1, 1)
    assert recorder.notes == []
    assert recorder.inline == []


def test_force_review_clears_mute_and_resets_budget(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.MR_COMMENT_BUDGET", 30)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult("[LGTM]", "stop", 0, 0, 0.0),
    )
    seed(state, Ledger(muted=True, posted=30))
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: [fd("a.py")])
    pipeline.review_merge_request(1, 1, force=True)
    assert state["ledger"].muted is False
    assert len(recorder.notes) == 1


def test_unreadable_ledger_skips_the_run(harness, monkeypatch, caplog):
    """Fail closed. An empty-ledger fallback would re-review the whole MR."""
    recorder, state = harness
    from reviewer.ledger import LedgerUnavailable

    def _boom(mr):
        raise LedgerUnavailable("gitlab is down")

    monkeypatch.setattr("reviewer.pipeline.LedgerStore.load", staticmethod(_boom))
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: [fd("a.py")])
    with caplog.at_level(logging.ERROR):
        pipeline.review_merge_request(1, 1)
    assert recorder.notes == []
    assert recorder.inline == []
    # Silence alone proves nothing: an unhandled crash is silent too. The run
    # must have recognised the failure and declined on purpose.
    assert "review state unreadable" in caplog.text
    assert "Critical error" not in caplog.text


def test_a_rate_limited_run_records_nothing_and_reports_once(harness, monkeypatch):
    """A backend outage must cost neither a comment nor a budget slot.

    The files stay out of the ledger so the next push retries them, and no
    summary is posted: one note per push would mute the MR over an outage that
    fixes itself, and the command that lifts a mute is a human action.
    """
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult(
            "", "ratelimit", 0, 0, 0.0,
        ),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(5)],
    )
    pipeline.review_merge_request(1, 1)
    assert recorder.inline == []
    assert len(recorder.notes) == 1, "the human is told once that nothing worked"
    assert state["ledger"].hunks == (), "nothing settled, so nothing is recorded"

    # The same head must not be reported again on every webhook retry.
    pipeline.review_merge_request(1, 1)
    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 1, "one report per head, not one per webhook"
    assert state["ledger"].muted is False, "an outage must not mute the MR"


def test_force_does_not_bypass_the_file_ceiling(harness, monkeypatch):
    """The spec is explicit: /review must not re-enable a 300-file review.

    An override that does is the same foot-gun in a different shape.
    """
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 3)
    monkeypatch.setattr(
        "reviewer.pipeline.review_file",
        lambda *a: pytest.fail("an oversized MR must never be reviewed per file"),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(10)],
    )
    pipeline.review_merge_request(1, 1, force=True)
    assert recorder.inline == []
    assert len(recorder.notes) == 1, "one refusal, not ten reviews"


def test_force_on_an_oversized_mr_answers_instead_of_vanishing(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 3)
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(10)],
    )
    pipeline.review_merge_request(1, 1)
    pipeline.review_merge_request(1, 1, force=True)
    assert len(recorder.notes) == 2, "the oversized note, then an answer to /review"
    assert "нема чого дивитись" in recorder.notes[1].lower()


def test_force_on_a_fully_reviewed_mr_answers_instead_of_vanishing(harness, monkeypatch):
    """render_budget_exhausted tells the human to type /review. Answering
    that with total silence reads as a dead bot."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    diff = fd("a.py")
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: [diff])
    seed(state, Ledger().record([hunk_key("a.py", h) for h in diff.hunks]))
    pipeline.review_merge_request(1, 1, force=True)
    assert len(recorder.notes) == 1
    assert "нема чого дивитись" in recorder.notes[0].lower()


def test_a_failing_ledger_save_stops_the_run_before_it_posts(harness, monkeypatch):
    """A save that fails AFTER the comment is out leaves posted=0 in GitLab
    forever, so every later push starts from zero with no ceiling at all."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult(
            "**🔴 [BLOCKER]** boom", "stop", 0, 0, 0.0,
        ),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(10)],
    )

    def _boom(self):
        raise RuntimeError("gitlab said 422")

    monkeypatch.setattr("reviewer.pipeline.LedgerStore.save", _boom, raising=False)

    for _ in range(20):
        pipeline.review_merge_request(1, 1)

    assert recorder.inline == [], "nothing may be posted once the ledger is unwritable"
    assert recorder.notes == [], "not even the free outage note, which would repeat"


def test_a_refunded_slot_is_not_charged_twice(harness, monkeypatch):
    """A clean file is charged up front and must get the slot back."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr(
        "reviewer.pipeline.review_file",
        lambda mr, file_diff, context, voice, review_state: pipeline.FileOutcome(
            file_diff.new_path, "clean", "",
        ),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(5)],
    )
    pipeline.review_merge_request(1, 1)
    assert state["ledger"].posted == 1, "five clean files plus one summary note"


def _release(monkeypatch, commits):
    monkeypatch.setattr(FakeMR, "source_branch", "release/2026.08", raising=False)
    monkeypatch.setattr("reviewer.gitlab_client.fetch_commits", lambda mr: commits)
    monkeypatch.setattr(
        "reviewer.pipeline.summary_chat",
        lambda *a, **k: pipeline.ChatResult("роаст", "stop", 0, 0, 0.0),
    )
    monkeypatch.setattr("reviewer.pipeline.dedupe", pipeline.DedupeCache(maxsize=8))


def test_a_release_summary_is_not_reposted_after_a_restart(harness, monkeypatch):
    """The old dedupe lived in process memory, so every container restart
    re-posted the same roast. The ledger outlives the process."""
    recorder, _ = harness
    _release(monkeypatch, [{"short_id": "aaa", "title": "fix a", "author": "x"}])

    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 1

    monkeypatch.setattr("reviewer.pipeline.dedupe", pipeline.DedupeCache(maxsize=8))
    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 1, "a restart must not repeat the roast"


def test_a_release_mr_cannot_outrun_the_comment_budget(harness, monkeypatch):
    """Every path that posts is under the ceiling, the release path included."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MR_COMMENT_BUDGET", 3)
    commits = [{"short_id": "aaa", "title": "fix a", "author": "x"}]
    _release(monkeypatch, commits)

    for i in range(30):
        # New commits every push, so nothing is ever deduped away.
        commits[0] = {"short_id": f"c{i}", "title": f"fix {i}", "author": "x"}
        monkeypatch.setattr("reviewer.pipeline.dedupe", pipeline.DedupeCache(maxsize=8))
        pipeline.review_merge_request(1, 1)

    assert len(recorder.notes) <= config.MR_COMMENT_BUDGET, (
        f"30 pushes posted {len(recorder.notes)} notes against a budget of 3"
    )
    assert state["ledger"].posted == config.MR_COMMENT_BUDGET


def test_a_comment_that_may_have_posted_keeps_its_slot(harness, monkeypatch):
    """post_inline only catches GitlabError. A connection reset while reading
    the response of a discussion GitLab already created used to escape, refund
    the slot, and leave the comment public and unpaid — the ceiling silently
    disengaging, which is the whole failure this branch exists to prevent."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult(
            "**🔴 [BLOCKER]** boom", "stop", 0, 0, 0.0,
        ),
    )

    def _posts_then_dies(mr, path, line, body):
        recorder.inline.append((path, body))
        raise ConnectionError("connection reset after the discussion was created")

    monkeypatch.setattr("reviewer.gitlab_client.post_inline", _posts_then_dies)
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(10)],
    )
    for _ in range(20):
        pipeline.review_merge_request(1, 1)

    # Retried once, because the post may never have landed and losing a real
    # finding silently is worse than a duplicate — then recorded, so it
    # converges instead of replaying on every push.
    assert len(recorder.inline) == 20, "ten files, one retry each, then settled"
    before = len(recorder.inline)
    for _ in range(20):
        pipeline.review_merge_request(1, 1)
    assert len(recorder.inline) == before, "further pushes must cost nothing"


def test_two_broken_files_do_not_kill_review_of_the_healthy_ones(harness, monkeypatch):
    """select_files sorts deterministically, so a breaker that counts per-file
    errors puts the same two files first every run and silences the MR for
    good. Errors are content-dependent; only rate limits are the backend."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)

    def _chat(system, user, deadline_s):
        if "poison" in user:
            return pipeline.ChatResult("", "error", 0, 0, 0.0)
        return pipeline.ChatResult("**🔴 [BLOCKER]** boom", "stop", 0, 0, 0.0)

    monkeypatch.setattr("reviewer.pipeline.review_chat", _chat)
    diffs = [fd("poison0.py"), fd("poison1.py")] + [fd(f"ok{i}.py") for i in range(8)]
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: diffs)

    pipeline.review_merge_request(1, 1)

    assert len(recorder.inline) == 8, "the healthy files must still be reviewed"
    assert len(recorder.notes) == 1, "and the run still summarises itself"
    for i in range(2):
        assert hunk_key(f"poison{i}.py", diffs[i].hunks[0]) not in state["ledger"].hunks


def test_a_file_that_fails_before_posting_gives_its_slot_back(harness, monkeypatch):
    """Slots are charged before review_file runs, because a save that fails
    after the comment is out disengages the ceiling. A file that never got as
    far as posting must not keep the charge, or a flaky MR mutes itself."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)

    def _dies_early(mr, file_diff, context, voice, review_state):
        raise RuntimeError("blew up building the prompt")

    monkeypatch.setattr("reviewer.pipeline.review_file", _dies_early)
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(5)],
    )
    pipeline.review_merge_request(1, 1)

    assert recorder.inline == []
    assert len(recorder.notes) == 1, "one free report saying they all failed"
    assert state["ledger"].posted == 0, (
        "five charges, five refunds: a run that reviewed nothing costs nothing"
    )


def _dead_backend(monkeypatch, files=10):
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult("", "error", 0, 0, 0.0),
    )
    heads = {"n": 0}

    def _diffs(mr):
        heads["n"] += 1
        FakeMR.diff_refs = {"base_sha": "b", "start_sha": "s",
                            "head_sha": f"sha{heads['n']}"}
        return [fd(f"f{i}.py") for i in range(files)]

    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", _diffs)


def test_an_outage_costs_nothing_no_matter_how_many_pushes(harness, monkeypatch):
    """Developers push. Charging a slot per push burned the whole budget over a
    day-long outage and muted the MR — and a muted MR reviews nothing once the
    backend comes back, which is the opposite of what an outage should cost."""
    recorder, state = harness
    _dead_backend(monkeypatch)

    for _ in range(40):
        pipeline.review_merge_request(1, 1)

    assert len(recorder.notes) == 1, "told once, not once per push"
    assert state["ledger"].posted == 0, "an outage must not spend the budget"
    assert state["ledger"].muted is False, "and must never mute the MR"
    assert state["ledger"].hunks == (), "nothing settled, so nothing is recorded"


def test_the_mr_reviews_normally_once_the_backend_recovers(harness, monkeypatch):
    recorder, state = harness
    _dead_backend(monkeypatch)
    for _ in range(40):
        pipeline.review_merge_request(1, 1)

    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult(
            "**🔴 [BLOCKER]** boom", "stop", 0, 0, 0.0,
        ),
    )
    pipeline.review_merge_request(1, 1)
    assert len(recorder.inline) == 10, "every file the outage swallowed is retried"


def test_review_answers_during_an_outage(harness, monkeypatch):
    """render_nothing_to_do exists because silence reads as a crash. A human
    who types /review while the backend is down must not get nothing."""
    recorder, _ = harness
    _dead_backend(monkeypatch)
    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 1

    pipeline.review_merge_request(1, 1, force=True)
    assert len(recorder.notes) == 2, "/review always answers"


def test_an_undelivered_finding_is_not_reported_as_delivered(harness, monkeypatch):
    """Claiming a finding was posted when the post failed leaves no trace at
    all: the hunk is settled, the comment does not exist, and the note lies."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult(
            "**🔴 [BLOCKER]** boom", "stop", 0, 0, 0.0,
        ),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.post_inline", lambda *a: False,
    )

    def _note(mr, body):
        if body.startswith("### "):
            raise ConnectionError("gitlab hung up")
        recorder.notes.append(body)

    monkeypatch.setattr("reviewer.gitlab_client.post_note", _note)
    diff = fd("a.py")
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: [diff])

    pipeline.review_merge_request(1, 1)

    assert "Findings on" not in recorder.notes[0], "nothing was actually delivered"
    assert "could not be posted" in recorder.notes[0]
    assert hunk_key("a.py", diff.hunks[0]) not in state["ledger"].hunks, (
        "an undelivered finding must be retried, not settled forever"
    )


def test_a_refund_returns_exactly_what_was_charged(harness, monkeypatch):
    """One good file, four that die before posting. Refunding more than was
    charged inflates the ceiling; refunding less mutes the MR early."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    calls = {"n": 0}

    def _one_good_then_boom(mr, file_diff, context, voice, review_state):
        calls["n"] += 1
        if calls["n"] == 1:
            recorder.inline.append((file_diff.new_path, "finding"))
            return pipeline.FileOutcome(file_diff.new_path, "reviewed", "1 response(s)")
        raise RuntimeError("blew up before posting")

    monkeypatch.setattr("reviewer.pipeline.review_file", _one_good_then_boom)
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(5)],
    )
    pipeline.review_merge_request(1, 1)

    assert state["ledger"].posted == 2, (
        "one finding plus one summary note; the four failures are refunded whole"
    )


def test_a_second_outage_is_reported_again(harness, monkeypatch):
    """"Told once" must mean once per outage, not once per merge request:
    otherwise the first blip buys permanent silence for every later one."""
    recorder, state = harness
    _dead_backend(monkeypatch, files=2)
    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 1

    healthy = lambda system, user, deadline_s: pipeline.ChatResult(
        "**🔴 [BLOCKER]** boom", "stop", 0, 0, 0.0,
    )
    monkeypatch.setattr("reviewer.pipeline.review_chat", healthy)
    pipeline.review_merge_request(1, 1)
    assert state["ledger"].outage_reported is False, "recovery clears the flag"

    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult("", "error", 0, 0, 0.0),
    )
    # Fresh files, or the run would find nothing to do and stay silent for that
    # reason instead of the one under test.
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs", lambda mr: [fd("later.py")],
    )
    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 3, "the new outage earns its own report"


def _reviewed(path, settled):
    return pipeline.FileOutcome(path, "reviewed", "could not be posted", settled)


def test_the_budget_note_does_not_invite_review_when_nothing_landed():
    """The budget branch runs before the summary, so this note is the only
    thing such a run produces. Telling the reader to spend a /review here is
    telling them to replay a merge request that showed them nothing."""
    note = pipeline.render_budget_exhausted(
        [_reviewed("a.py", False), _reviewed("b.py", False)],
    )
    assert "`/review`" in note, "the command is still named, just not urged"
    assert "Полагодь звʼязок" in note
    assert "Розгреби те, що вже написав" not in note
    assert "`a.py`" in note and "`b.py`" in note


def test_the_budget_note_reads_normally_when_comments_did_land():
    note = pipeline.render_budget_exhausted(
        [_reviewed("a.py", True), _reviewed("b.py", False)],
    )
    assert "Розгреби те, що вже написав" in note
    assert "`b.py`" in note, "the one that failed is still named"
    assert "Полагодь звʼязок" not in note


def test_the_budget_note_is_unchanged_for_a_run_with_no_outcomes():
    note = pipeline.render_budget_exhausted()
    assert "Розгреби те, що вже написав" in note


def _always_fails_to_post(monkeypatch, recorder):
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult(
            "**🔴 [BLOCKER]** boom", "stop", 0, 0, 0.0,
        ),
    )
    monkeypatch.setattr("reviewer.gitlab_client.post_inline", lambda *a: False)

    def _note(mr, body):
        if body.startswith("### "):
            raise ConnectionError("gitlab hung up")
        recorder.notes.append(body)

    monkeypatch.setattr("reviewer.gitlab_client.post_note", _note)


def test_a_saturated_ledger_does_not_replay_the_merge_request(harness, monkeypatch):
    """Past LEDGER_MAX_HUNKS the ledger evicted its oldest keys — precisely the
    ones select_files reaches first — so every /review posted a fresh full
    round. Twenty-one of them reached six hundred comments."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.LEDGER_MAX_HUNKS", 4)
    monkeypatch.setattr(
        "reviewer.pipeline.review_file",
        lambda mr, file_diff, context, voice, review_state: pipeline.FileOutcome(
            file_diff.new_path, "clean", "",
        ),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(10)],
    )
    commands = 21
    for _ in range(commands):
        pipeline.review_merge_request(1, 1, force=True)

    assert state["ledger"].saturated is True
    # One answer per command is the /review contract. What must never happen is
    # a fresh round of file comments per command, which is what eviction bought
    # and what reached six hundred. The budget is not what stops it here: the
    # run never gets far enough to spend one.
    assert len(recorder.inline) == 0
    assert len(recorder.notes) <= commands + 1, (
        f"{len(recorder.notes)} notes from {commands} /review commands"
    )
    for note in recorder.notes[1:]:
        assert "Нема чого дивитись" in note


def test_the_budget_note_still_names_undelivered_on_a_second_review(
    harness, monkeypatch,
):
    """The give-up used to rewrite the outcome to settled before the note read
    it, so run two told the human to go read comments that never arrived."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.MR_COMMENT_BUDGET", 4)
    _always_fails_to_post(monkeypatch, recorder)
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(6)],
    )
    pipeline.review_merge_request(1, 1)
    pipeline.review_merge_request(1, 1, force=True)

    assert recorder.notes, "the budget note is the only thing such a run posts"
    for note in recorder.notes:
        assert "Розгреби те, що вже написав" not in note, (
            "nothing was ever delivered; there is nothing to go and read"
        )


def test_an_outage_that_raises_costs_nothing_either(harness, monkeypatch):
    """The returning-outage shape was covered; the raising one was not, and it
    skipped the save, so the last file's charge stayed and thirty pushes
    muted the merge request."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)

    def _raises(mr, file_diff, context, voice, review_state):
        raise RuntimeError("the client blew up")

    monkeypatch.setattr("reviewer.pipeline.review_file", _raises)
    heads = {"n": 0}

    def _diffs(mr):
        heads["n"] += 1
        FakeMR.diff_refs = {"base_sha": "b", "start_sha": "s",
                            "head_sha": f"sha{heads['n']}"}
        return [fd(f"f{i}.py") for i in range(5)]

    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", _diffs)

    for _ in range(40):
        pipeline.review_merge_request(1, 1)

    assert state["ledger"].posted == 0, "an outage must not spend the budget"
    assert state["ledger"].muted is False, "and must never mute the MR"
    # _charge writes the charge to GitLab before review_file runs, so the
    # refund has to be persisted per file too. Leaving it to the end of the run
    # means anything that kills the run first leaves the charge standing.
    assert state["saves"] / 40 >= 10, (
        f"{state['saves'] / 40} saves per push: the refund is not persisted "
        f"per file"
    )
