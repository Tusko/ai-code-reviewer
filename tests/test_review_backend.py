import pytest

from reviewer import openrouter_client


class FakeResponse:
    ok = True
    status_code = 200
    headers: dict = {}

    def __init__(self, content="**🔴 [BLOCKER]** boom"):
        self._content = content

    def json(self):
        return {
            "choices": [
                {"message": {"content": self._content}, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "poolside/laguna-s-2.1",
        }


@pytest.fixture
def captured(monkeypatch):
    """Captures the request body openrouter_client would have sent."""
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent["url"] = url
        sent["body"] = json
        return FakeResponse()

    monkeypatch.setattr("reviewer.openrouter_client.requests.post", fake_post)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", "sk-test")
    return sent


def test_review_chat_uses_review_model(captured, monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_REVIEW_MODEL", "poolside/laguna-s-2.1")
    monkeypatch.setattr("reviewer.config.OPENROUTER_MODEL", "voice/model")
    openrouter_client.review_chat("sys", "user", 30)
    assert captured["body"]["model"] == "poolside/laguna-s-2.1"


def test_review_chat_uses_review_output_cap(captured, monkeypatch):
    monkeypatch.setattr("reviewer.config.REVIEW_MAX_OUTPUT_TOKENS", 4096)
    monkeypatch.setattr("reviewer.config.OPENROUTER_MAX_TOKENS", 512)
    openrouter_client.review_chat("sys", "user", 30)
    assert captured["body"]["max_tokens"] == 4096


def test_review_chat_omits_models_key_when_no_fallbacks(captured, monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_REVIEW_FALLBACK_MODELS", [])
    openrouter_client.review_chat("sys", "user", 30)
    assert "models" not in captured["body"]


def test_review_chat_sends_fallback_chain(captured, monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_REVIEW_MODEL", "poolside/laguna-s-2.1")
    monkeypatch.setattr(
        "reviewer.config.OPENROUTER_REVIEW_FALLBACK_MODELS", ["poolside/laguna-xs-2.1"],
    )
    openrouter_client.review_chat("sys", "user", 30)
    assert captured["body"]["models"] == [
        "poolside/laguna-s-2.1", "poolside/laguna-xs-2.1",
    ]


def test_review_chat_is_low_temperature(captured):
    openrouter_client.review_chat("sys", "user", 30)
    assert captured["body"]["temperature"] == 0.1


def test_voice_chat_still_uses_voice_model(captured, monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_MODEL", "voice/model")
    monkeypatch.setattr("reviewer.config.OPENROUTER_REVIEW_MODEL", "poolside/laguna-s-2.1")
    openrouter_client.chat("sys", "user", 30)
    assert captured["body"]["model"] == "voice/model"


def test_review_chat_without_key_reports_error(monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    result = openrouter_client.review_chat("sys", "user", 30)
    assert result.failed is True


from reviewer import pipeline
from reviewer.chat_types import LGTM_TEXT, ChatResult
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.pipeline import ReviewState


def _fd(path="a.py"):
    return FileDiff(
        old_path=path, new_path=path, is_new=False, is_deleted=False,
        is_renamed=False, is_binary=False,
        hunks=(Hunk(1, 1, (" ctx", "+added")),),
    )


class _MR:
    pass


def test_review_state_trips_after_two_ratelimits():
    state = ReviewState()
    assert state.open is True
    state.record("ratelimit")
    assert state.open is True
    state.record("ratelimit")
    assert state.open is False


def test_review_state_resets_on_success():
    state = ReviewState()
    state.record("ratelimit")
    state.record("stop")
    state.record("ratelimit")
    assert state.open is True


def test_review_file_calls_openrouter_not_ollama(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: (
            calls.append("openrouter"),
            ChatResult(LGTM_TEXT, "stop", 0, 0, 0.0),
        )[1],
    )

    def _boom(*args, **kwargs):
        raise AssertionError("ollama_client.chat must not be called for review")

    monkeypatch.setattr("reviewer.ollama_client.chat", _boom)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    outcome = pipeline.review_file(_MR(), _fd(), "")
    assert calls == ["openrouter"]
    assert outcome.status == "clean"


def test_review_file_records_ratelimit_on_state(monkeypatch):
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: ChatResult("", "ratelimit", 0, 0, 0.0),
    )
    state = ReviewState()
    outcome = pipeline.review_file(_MR(), _fd(), "", None, state)
    assert outcome.status == "error"
    assert state.ratelimits == 1
