import reviewer.pipeline as pipeline
from reviewer.diff_parser import FileDiff, Hunk

FD = FileDiff("a.py", "a.py", False, False, False, False, (Hunk(1, 1, (" x", "+y")),))


class FakeMR:
    def __init__(self, branch="feature/x"):
        self.source_branch = branch
        self.diff_refs = {"base_sha": "b", "start_sha": "s", "head_sha": "h"}
        self.notes_posted = []

    def changes(self):
        return {"changes": []}


def test_release_branches_are_skipped():
    assert pipeline.should_skip_branch("release/2026.08") is True
    assert pipeline.should_skip_branch("feature/x") is False
    assert pipeline.should_skip_branch("") is False
    # Prefix match only — "release/" appearing mid-branch-name must not trigger.
    assert pipeline.should_skip_branch("feature/prerelease/fix") is False
    assert pipeline.should_skip_branch("chore/prerelease/beta") is False
    assert pipeline.should_skip_branch("hotfix/release/patch") is False


def test_identical_diff_is_reviewed_once(monkeypatch):
    calls = []
    mr = FakeMR()

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "fetch_file_diffs", lambda m: [FD])
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: calls.append(body))
    monkeypatch.setattr(pipeline, "review_file",
                        lambda mr_, fd, ctx: pipeline.FileOutcome(fd.new_path, "clean", ""))
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    assert len(calls) == 1

    pipeline.review_merge_request(1, 1)
    assert len(calls) == 1   # second identical diff costs nothing


def test_changed_diff_is_reviewed_again(monkeypatch):
    calls = []
    mr = FakeMR()
    diffs = [FD]

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "fetch_file_diffs", lambda m: diffs)
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: calls.append(body))
    monkeypatch.setattr(pipeline, "review_file",
                        lambda mr_, fd, ctx: pipeline.FileOutcome(fd.new_path, "clean", ""))
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    diffs[0] = FileDiff("a.py", "a.py", False, False, False, False,
                        (Hunk(1, 1, (" x", "+z")),))
    pipeline.review_merge_request(1, 1)
    assert len(calls) == 2


def test_failed_review_does_not_suppress_retry(monkeypatch):
    calls = []
    review_calls = []
    mr = FakeMR()

    def fake_post_note(m, body):
        calls.append(body)
        if len(calls) == 1:
            raise RuntimeError("network blip")

    def fake_review_file(mr_, fd, ctx):
        review_calls.append(fd.new_path)
        return pipeline.FileOutcome(fd.new_path, "clean", "")

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "fetch_file_diffs", lambda m: [FD])
    monkeypatch.setattr(pipeline.gitlab_client, "post_note", fake_post_note)
    monkeypatch.setattr(pipeline, "review_file", fake_review_file)
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    assert len(review_calls) == 1   # attempted, but post_note raised

    pipeline.review_merge_request(1, 1)   # same diff — must not be suppressed by dedupe
    assert len(review_calls) == 2


def test_release_branch_mr_posts_nothing(monkeypatch):
    calls = []
    mr = FakeMR(branch="release/2026.08")

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "fetch_file_diffs", lambda m: [FD])
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: calls.append(body))
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    assert calls == []
