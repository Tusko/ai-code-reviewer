import pytest

from reviewer import pipeline
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.ledger import Ledger, hunk_key
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


@pytest.fixture
def harness(monkeypatch):
    """Wires review_merge_request to fakes and returns (recorder, state)."""
    recorder = Recorder()
    state = {"ledger": Ledger(), "saves": 0}
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
        def __init__(self):
            self.mr = mr

        @property
        def ledger(self):
            return state["ledger"]

        @ledger.setter
        def ledger(self, value):
            state["ledger"] = value

        @classmethod
        def load(cls, mr):
            return cls()

        def save(self):
            state["saves"] += 1

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
    state["ledger"] = Ledger().record(
        [hunk_key("a.py", h) for h in diff.hunks]
    )
    pipeline.review_merge_request(1, 1)
    assert recorder.notes == []
    assert recorder.inline == []


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
