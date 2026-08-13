import pytest
import gitlab.exceptions

from reviewer.diff_parser import FileDiff, Hunk
from reviewer.gitlab_client import (
    diff_fingerprint, fetch_file_content, post_inline, post_note, surgical_context,
)

HUNK = Hunk(old_start=1, new_start=5, lines=(" a", "+b"))
FD = FileDiff("x.py", "x.py", False, False, False, False, (HUNK,))


class FakeDiscussions:
    def __init__(self, fail=False):
        self.created = []
        self.fail = fail

    def create(self, payload):
        if self.fail:
            raise gitlab.exceptions.GitlabError("400 Bad Request")
        self.created.append(payload)


class FakeNotes:
    def __init__(self):
        self.created = []

    def create(self, payload):
        self.created.append(payload)


class FakeMR:
    def __init__(self, fail_inline=False):
        self.discussions = FakeDiscussions(fail=fail_inline)
        self.notes = FakeNotes()
        self.diff_refs = {"base_sha": "b1", "start_sha": "s1", "head_sha": "h1"}


def test_post_inline_builds_position():
    mr = FakeMR()
    assert post_inline(mr, "src/a.py", 42, "body") is True
    payload = mr.discussions.created[0]
    assert payload["body"] == "body"
    assert payload["position"]["new_line"] == 42
    assert payload["position"]["new_path"] == "src/a.py"
    assert payload["position"]["position_type"] == "text"
    assert payload["position"]["head_sha"] == "h1"


def test_post_inline_returns_false_when_gitlab_rejects():
    mr = FakeMR(fail_inline=True)
    assert post_inline(mr, "src/a.py", 42, "body") is False


def test_post_inline_returns_false_without_diff_refs():
    mr = FakeMR()
    mr.diff_refs = None
    assert post_inline(mr, "src/a.py", 42, "body") is False


def test_post_note_creates_note():
    mr = FakeMR()
    post_note(mr, "summary")
    assert mr.notes.created == [{"body": "summary"}]


def test_surgical_context_extracts_window_around_hunk():
    text = "\n".join(f"line{i}" for i in range(1, 21))
    out = surgical_context(text, [HUNK], window=2)
    assert "line5" in out
    assert "line20" not in out


def test_surgical_context_handles_hunk_past_end_of_file():
    out = surgical_context("only one line", [Hunk(1, 999, (" a",))], window=3)
    assert isinstance(out, str)


def test_surgical_context_merges_overlapping_windows():
    text = "\n".join(f"line{i}" for i in range(1, 21))
    hunks = [Hunk(1, 5, (" a",)), Hunk(1, 7, (" b",))]
    out = surgical_context(text, hunks, window=5)
    assert out.count("line5") == 1


def test_fetch_file_content_returns_empty_on_error():
    class Boom:
        class files:
            @staticmethod
            def get(**kwargs):
                raise RuntimeError("404")

    assert fetch_file_content(Boom, "a.py", "main") == ""


def test_diff_fingerprint_is_stable_and_content_sensitive():
    a = diff_fingerprint(1, 1, [FD])
    b = diff_fingerprint(1, 1, [FD])
    other = FileDiff("x.py", "x.py", False, False, False, False,
                     (Hunk(1, 5, (" a", "+c")),))
    assert a == b
    assert diff_fingerprint(1, 1, [other]) != a


def test_diff_fingerprint_distinguishes_path_and_line_boundaries():
    # Regression test for hash collision bug:
    # Without delimiters, "x1" + "2 a" and "x" + "12 a" both hash the same.
    a = FileDiff('x1', 'x1', False, False, False, False, (Hunk(1, 2, (' a',)),))
    b = FileDiff('x', 'x', False, False, False, False, (Hunk(1, 12, (' a',)),))
    assert diff_fingerprint(1, 1, [a]) != diff_fingerprint(1, 1, [b])


def test_diff_fingerprint_distinguishes_project_and_mr_identity():
    # I1 regression: DedupeCache is a single process-wide instance shared
    # across every project the bot serves. Two merge requests with
    # byte-identical diffs (backports, cross-project cherry-picks, a
    # re-created MR) must not collide.
    assert diff_fingerprint(1, 1, [FD]) != diff_fingerprint(2, 1, [FD])
    assert diff_fingerprint(1, 1, [FD]) != diff_fingerprint(1, 2, [FD])


def test_post_inline_propagates_malformed_diff_refs():
    mr = FakeMR()
    del mr.diff_refs["head_sha"]
    with pytest.raises(KeyError):
        post_inline(mr, "src/a.py", 42, "body")
