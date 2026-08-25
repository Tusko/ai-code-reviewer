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
