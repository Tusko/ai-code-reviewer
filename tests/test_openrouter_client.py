import json

from reviewer.ollama_client import ChatResult
from reviewer.openrouter_client import chat


class FakeResponse:
    def __init__(self, payload, status=200, raw=""):
        self._payload = payload
        self.status_code = status
        self.text = raw or json.dumps(payload)
        self.ok = status == 200

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


def _ok(content="Єбать, знову реліз.", finish="stop"):
    return {
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish,
        }],
        "usage": {"prompt_tokens": 80, "completion_tokens": 40},
    }


def test_openrouter_chat_extracts_content(monkeypatch):
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
    monkeypatch.setattr(
        "reviewer.openrouter_client.requests.post",
        lambda *a, **k: FakeResponse(_ok()),
    )
    result = chat("sys", "user", deadline_s=30)
    assert isinstance(result, ChatResult)
    assert result.failed is False
    assert "Єбать" in result.text
    assert result.prompt_eval_count == 80
    assert result.eval_count == 40


def test_openrouter_sends_model_and_auth(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResponse(_ok())

    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_MAX_TOKENS", 512)
    monkeypatch.setattr("reviewer.openrouter_client.requests.post", fake_post)

    chat("sys", "user", deadline_s=30, temperature=1.0)
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["json"]["model"] == "google/gemma-4-26b-a4b-it:free"
    assert captured["json"]["temperature"] == 1.0
    assert captured["headers"]["Authorization"] == "Bearer sk-or-test"
    assert captured["headers"]["X-Title"] == "Sidorovich"


def test_openrouter_http_error_is_failed(monkeypatch):
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "reviewer.openrouter_client.requests.post",
        lambda *a, **k: FakeResponse(
            {"error": {"message": "rate limited"}}, status=429,
        ),
    )
    result = chat("sys", "user", deadline_s=30)
    assert result.failed is True
    assert "rate limited" in result.text


def test_openrouter_error_payload_on_200_is_failed(monkeypatch):
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "reviewer.openrouter_client.requests.post",
        lambda *a, **k: FakeResponse({"error": {"message": "no provider"}}),
    )
    result = chat("sys", "user", deadline_s=30)
    assert result.failed is True
    assert "no provider" in result.text


def test_openrouter_empty_content_is_incomplete(monkeypatch):
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "reviewer.openrouter_client.requests.post",
        lambda *a, **k: FakeResponse(_ok(content="")),
    )
    result = chat("sys", "user", deadline_s=30)
    assert result.done_reason == "incomplete"
    assert result.text == ""


def test_openrouter_missing_key_fails_without_http(monkeypatch):
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_API_KEY", None)
    called = []
    monkeypatch.setattr(
        "reviewer.openrouter_client.requests.post",
        lambda *a, **k: called.append(1) or FakeResponse(_ok()),
    )
    result = chat("sys", "user", deadline_s=30)
    assert result.failed is True
    assert called == []


def test_openrouter_transport_error_is_failed(monkeypatch):
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_API_KEY", "sk-or-test")

    def boom(*a, **k):
        raise ConnectionError("refused")

    monkeypatch.setattr("reviewer.openrouter_client.requests.post", boom)
    result = chat("sys", "user", deadline_s=30)
    assert result.failed is True
    assert "refused" in result.text
