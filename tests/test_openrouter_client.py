import json

from reviewer.ollama_client import ChatResult
from reviewer.openrouter_client import chat


class FakeResponse:
    def __init__(self, payload, status=200, raw="", headers=None):
        self._payload = payload
        self.status_code = status
        self.text = raw or json.dumps(payload)
        self.ok = status == 200
        self.headers = headers or {}

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
    monkeypatch.setattr("reviewer.openrouter_client.time.sleep", lambda s: None)
    monkeypatch.setattr(
        "reviewer.openrouter_client.requests.post",
        lambda *a, **k: FakeResponse(
            {"error": {"message": "rate limited"}}, status=429,
        ),
    )
    result = chat("sys", "user", deadline_s=30)
    assert result.failed is True
    assert result.done_reason == "ratelimit"
    assert "rate limited" in result.text


def test_openrouter_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_API_KEY", "sk-or-test")
    slept = []
    monkeypatch.setattr("reviewer.openrouter_client.time.sleep", slept.append)
    calls = []

    def post(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return FakeResponse(
                {"error": {"message": "Provider returned error"}}, status=429,
            )
        return FakeResponse(_ok("Ну шо, реліз."))

    monkeypatch.setattr("reviewer.openrouter_client.requests.post", post)
    result = chat("sys", "user", deadline_s=30)
    assert result.failed is False
    assert result.text == "Ну шо, реліз."
    assert len(calls) == 2
    assert slept == [2.0]


def test_openrouter_honours_retry_after_header(monkeypatch):
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_API_KEY", "sk-or-test")
    slept = []
    monkeypatch.setattr("reviewer.openrouter_client.time.sleep", slept.append)
    calls = []

    def post(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return FakeResponse(
                {"error": {"message": "slow down"}}, status=429,
                headers={"Retry-After": "7"},
            )
        return FakeResponse(_ok())

    monkeypatch.setattr("reviewer.openrouter_client.requests.post", post)
    chat("sys", "user", deadline_s=60)
    assert slept == [7.0]


def test_openrouter_does_not_retry_past_deadline(monkeypatch):
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_API_KEY", "sk-or-test")
    slept = []
    monkeypatch.setattr("reviewer.openrouter_client.time.sleep", slept.append)
    monkeypatch.setattr(
        "reviewer.openrouter_client.requests.post",
        lambda *a, **k: FakeResponse({"error": {"message": "nope"}}, status=429),
    )
    result = chat("sys", "user", deadline_s=3)
    assert result.done_reason == "ratelimit"
    assert slept == []


def test_openrouter_does_not_retry_client_error(monkeypatch):
    monkeypatch.setattr("reviewer.openrouter_client.config.OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("reviewer.openrouter_client.time.sleep", lambda s: None)
    calls = []

    def post(*a, **k):
        calls.append(1)
        return FakeResponse({"error": {"message": "bad key"}}, status=401)

    monkeypatch.setattr("reviewer.openrouter_client.requests.post", post)
    result = chat("sys", "user", deadline_s=30)
    assert result.done_reason == "error"
    assert len(calls) == 1


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
