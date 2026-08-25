import reviewer.pipeline as pipeline
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.ollama_client import ChatResult

FD = FileDiff("a.py", "a.py", False, False, False, False, (Hunk(1, 1, (" x", "+y")),))


class FakeCommit:
    def __init__(self, short_id="abc1234", title="fix feed", author_name="Ivan"):
        self.short_id = short_id
        self.id = short_id
        self.title = title
        self.message = title
        self.author_name = author_name


class FakeMR:
    def __init__(self, branch="feature/x", commits=None):
        self.source_branch = branch
        self.diff_refs = {"base_sha": "b", "start_sha": "s", "head_sha": "h"}
        self.notes_posted = []
        self._commits = commits if commits is not None else [FakeCommit()]

    def changes(self):
        return {"changes": []}

    def commits(self):
        return list(self._commits)


def _sidorovich(text="Опять реліз без рев'ю, їб вашу мать."):
    return ChatResult(text=text, done_reason="stop",
                      prompt_eval_count=0, eval_count=0, elapsed_s=0.1)


def test_release_and_hotfix_branches_are_skipped():
    assert pipeline.should_skip_branch("release/2026.08") is True
    assert pipeline.should_skip_branch("hotfix/urgent-fix") is True
    assert pipeline.should_skip_branch("feature/x") is False
    assert pipeline.should_skip_branch("") is False
    # Prefix match only — "release/" appearing mid-branch-name must not trigger.
    assert pipeline.should_skip_branch("feature/prerelease/fix") is False
    assert pipeline.should_skip_branch("chore/prerelease/beta") is False
    assert pipeline.should_skip_branch("hotfix/release/patch") is True


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


def test_forced_review_runs_even_when_fingerprint_cached(monkeypatch):
    calls = []
    review_calls = []
    mr = FakeMR()

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "fetch_file_diffs", lambda m: [FD])
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: calls.append(body))

    def fake_review_file(mr_, fd, ctx, voice=None):
        review_calls.append(fd.new_path)
        return pipeline.FileOutcome(fd.new_path, "clean", "")

    monkeypatch.setattr(pipeline, "review_file", fake_review_file)
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    assert len(review_calls) == 1
    assert len(calls) == 1

    # Same diff, dedupe would normally skip it — force=True (manual /review) must
    # bypass the dedupe gate and re-run.
    pipeline.review_merge_request(1, 1, force=True)
    assert len(review_calls) == 2
    assert len(calls) == 2


def test_failed_review_does_not_suppress_retry(monkeypatch):
    calls = []
    review_calls = []
    mr = FakeMR()

    def fake_post_note(m, body):
        calls.append(body)
        if len(calls) == 1:
            raise RuntimeError("network blip")

    def fake_review_file(mr_, fd, ctx, voice=None):
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


def test_skipped_branch_posts_sidorovich_summary(monkeypatch):
    notes = []
    review_calls = []
    mr = FakeMR(branch="release/2026.08")

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "fetch_file_diffs",
                        lambda m: (_ for _ in ()).throw(AssertionError("no full review")))
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: notes.append(body))
    monkeypatch.setattr(pipeline, "review_file",
                        lambda *a, **k: review_calls.append(1))
    monkeypatch.setattr(pipeline, "summary_chat",
                        lambda *a, **k: _sidorovich("hotfix знову без пекла, сука."))
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    assert review_calls == []
    assert notes == ["hotfix знову без пекла, сука."]

    notes.clear()
    monkeypatch.setattr(
        pipeline.gitlab_client, "fetch_mr",
        lambda p, i: (object(), FakeMR(branch="hotfix/urgent-fix")),
    )
    pipeline.review_merge_request(1, 2)
    assert notes == ["hotfix знову без пекла, сука."]
    assert review_calls == []


def test_sidorovich_summary_retries_when_russian(monkeypatch):
    notes = []
    users = []
    replies = [
        _sidorovich("Опять этот недоделанный высер в репозиторий закинули без спроса."),
        _sidorovich("Опять цей недолугий висер у репозиторій закинули без спросу."),
    ]

    def fake_chat(system, user, deadline_s, history=()):
        users.append(user)
        return replies.pop(0)

    mr = FakeMR(branch="release/2026.08")
    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: notes.append(body))
    monkeypatch.setattr(pipeline, "summary_chat", fake_chat)
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    assert notes == ["Опять цей недолугий висер у репозиторій закинули без спросу."]
    assert "російською" in users[1]


def test_sidorovich_summary_is_deduped_until_commits_change(monkeypatch):
    notes = []
    mr = FakeMR(branch="release/2026.08", commits=[FakeCommit("aaa", "fix a")])

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: notes.append(body))
    monkeypatch.setattr(pipeline, "summary_chat", lambda *a, **k: _sidorovich())
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    pipeline.review_merge_request(1, 1)
    assert len(notes) == 1

    mr._commits = [FakeCommit("bbb", "fix b")]
    pipeline.review_merge_request(1, 1)
    assert len(notes) == 2


def test_sidorovich_failure_does_not_suppress_retry(monkeypatch):
    notes = []
    chats = []
    mr = FakeMR(branch="hotfix/boom")

    def fake_chat(*a, **k):
        chats.append(1)
        if len(chats) == 1:
            return ChatResult(text="boom", done_reason="error",
                              prompt_eval_count=0, eval_count=0, elapsed_s=0.1)
        return _sidorovich("тепер хоч щось")

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: notes.append(body))
    monkeypatch.setattr(pipeline, "summary_chat", fake_chat)
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    assert notes == []
    pipeline.review_merge_request(1, 1)
    assert notes == ["тепер хоч щось"]


def test_summary_chat_uses_openrouter_when_keyed(monkeypatch):
    monkeypatch.setattr(pipeline.config, "OPENROUTER_API_KEY", "sk-or-test")
    called = []
    monkeypatch.setattr(
        pipeline.openrouter_client, "chat",
        lambda *a, **k: called.append(("or", k.get("temperature"))) or _sidorovich(),
    )
    monkeypatch.setattr(
        pipeline.ollama_client, "chat",
        lambda *a, **k: called.append(("ollama", None)) or _sidorovich(),
    )
    result = pipeline.summary_chat("sys", "user", 30)
    assert called == [("or", 1.0)]
    assert result.text.startswith("Опять")


def test_summary_chat_without_key_reports_voice_unavailable(monkeypatch):
    """Default: no OpenRouter means no roast. A coder model cannot do surzhyk."""
    monkeypatch.setattr(pipeline.config, "OPENROUTER_API_KEY", None)
    monkeypatch.setattr(pipeline.config, "SIDOROVICH_OLLAMA_FALLBACK", False)
    monkeypatch.setattr(
        pipeline.ollama_client, "chat",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call Ollama")),
    )
    result = pipeline.summary_chat("sys", "user", 30)
    assert result.done_reason == "unavailable"
    assert result.text == ""


def test_summary_chat_falls_back_to_ollama_without_key(monkeypatch):
    monkeypatch.setattr(pipeline.config, "OPENROUTER_API_KEY", None)
    monkeypatch.setattr(pipeline.config, "SIDOROVICH_OLLAMA_FALLBACK", True)
    called = []
    monkeypatch.setattr(
        pipeline.openrouter_client, "chat",
        lambda *a, **k: called.append("or") or _sidorovich(),
    )
    monkeypatch.setattr(
        pipeline.ollama_client, "chat",
        lambda *a, **k: called.append("ollama") or _sidorovich(),
    )
    pipeline.summary_chat("sys", "user", 30)
    assert called == ["ollama"]


def test_summary_chat_falls_back_to_ollama_when_openrouter_fails(monkeypatch):
    monkeypatch.setattr(pipeline.config, "OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(pipeline.config, "OLLAMA_MODEL", "qwen2.5-coder:7b")
    monkeypatch.setattr(pipeline.config, "SIDOROVICH_OLLAMA_FALLBACK", True)
    called = []
    monkeypatch.setattr(
        pipeline.openrouter_client, "chat",
        lambda *a, **k: called.append("or") or ChatResult(
            text="boom", done_reason="error",
            prompt_eval_count=0, eval_count=0, elapsed_s=0.1,
        ),
    )
    monkeypatch.setattr(
        pipeline.ollama_client, "chat",
        lambda *a, **k: called.append("ollama") or _sidorovich("з ollama"),
    )
    result = pipeline.summary_chat("sys", "user", 30)
    assert called == ["or", "ollama"]
    assert result.text == "з ollama"


def test_summary_chat_falls_back_to_ollama_when_openrouter_empty(monkeypatch):
    monkeypatch.setattr(pipeline.config, "OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(pipeline.config, "SIDOROVICH_OLLAMA_FALLBACK", True)
    monkeypatch.setattr(
        pipeline.openrouter_client, "chat",
        lambda *a, **k: ChatResult(
            text="  ", done_reason="stop",
            prompt_eval_count=0, eval_count=0, elapsed_s=0.1,
        ),
    )
    monkeypatch.setattr(
        pipeline.ollama_client, "chat",
        lambda *a, **k: _sidorovich("з ollama"),
    )
    result = pipeline.summary_chat("sys", "user", 30)
    assert result.text == "з ollama"


def test_summary_chat_rate_limit_does_not_reach_ollama_by_default(monkeypatch):
    monkeypatch.setattr(pipeline.config, "OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(pipeline.config, "SIDOROVICH_OLLAMA_FALLBACK", False)
    called = []
    monkeypatch.setattr(pipeline.ollama_client, "chat", lambda *a, **k: called.append(1))
    monkeypatch.setattr(
        pipeline.openrouter_client, "chat",
        lambda *a, **k: ChatResult(
            text="rate limited", done_reason="ratelimit",
            prompt_eval_count=0, eval_count=0, elapsed_s=0.1,
        ),
    )
    # "ratelimit", not "unavailable": a rate limit is transient, and reporting it
    # as "no voice model here" makes the caller dedupe the MR forever.
    assert pipeline.summary_chat("sys", "user", 30).done_reason == "ratelimit"
    assert called == []


def test_rate_limited_summary_is_not_deduped(monkeypatch):
    notes = []
    mr = FakeMR(branch="release/2026.08", commits=[FakeCommit("aaa", "MONO-1 fix a")])

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: notes.append(body))
    attempts = []

    def limited(*a, **k):
        attempts.append(1)
        return ChatResult(text="", done_reason="ratelimit",
                          prompt_eval_count=0, eval_count=0, elapsed_s=0.0)

    monkeypatch.setattr(pipeline, "summary_chat", limited)
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 908)
    assert notes == []

    # A rate limit must leave the MR retryable: the next webhook tries again
    # instead of finding the commits already deduped.
    pipeline.review_merge_request(1, 908)
    assert notes == []
    assert len(attempts) == 2


def test_unavailable_voice_posts_plain_commit_digest(monkeypatch):
    notes = []
    mr = FakeMR(branch="release/2026.08", commits=[FakeCommit("aaa", "MONO-1 fix a")])

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: notes.append(body))
    monkeypatch.setattr(
        pipeline, "summary_chat",
        lambda *a, **k: ChatResult(text="", done_reason="unavailable",
                                   prompt_eval_count=0, eval_count=0, elapsed_s=0.0),
    )
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    assert len(notes) == 1
    assert "MONO-1 fix a" in notes[0]

    # Not a transient failure: it is deduped, not retried on every webhook.
    pipeline.review_merge_request(1, 1)
    assert len(notes) == 1
