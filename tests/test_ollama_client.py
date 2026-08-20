import json

import pytest

from reviewer.ollama_client import ChatResult, chat, clean_response


class FakeResponse:
    def __init__(self, payloads, status=200):
        self._payloads = payloads
        self.status_code = status
        self.closed = False

    @property
    def ok(self):
        return self.status_code == 200

    def iter_lines(self):
        for payload in self._payloads:
            yield json.dumps(payload).encode()

    def close(self):
        self.closed = True

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")

    @property
    def text(self):
        return "error body"


def _stream(chunks, done_reason="stop", eval_count=120):
    payloads = [{"message": {"content": c}, "done": False} for c in chunks]
    payloads.append({
        "message": {"content": ""},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 900,
        "eval_count": eval_count,
    })
    return payloads


def test_streamed_chunks_are_concatenated(monkeypatch):
    monkeypatch.setattr(
        "reviewer.ollama_client.requests.post",
        lambda *a, **kw: FakeResponse(_stream(["Hello ", "world"])),
    )
    result = chat("sys", "user", deadline_s=30)
    assert result.text == "Hello world"
    assert result.eval_count == 120
    assert result.prompt_eval_count == 900


def test_short_lgtm_is_not_treated_as_truncated(monkeypatch):
    # Regression for B2: eval_count of 8 must not trigger a retry.
    monkeypatch.setattr(
        "reviewer.ollama_client.requests.post",
        lambda *a, **kw: FakeResponse(_stream(["[LGTM]"], eval_count=8)),
    )
    result = chat("sys", "user", deadline_s=30)
    assert result.truncated is False


def test_length_done_reason_is_truncated(monkeypatch):
    monkeypatch.setattr(
        "reviewer.ollama_client.requests.post",
        lambda *a, **kw: FakeResponse(_stream(["partial"], done_reason="length")),
    )
    result = chat("sys", "user", deadline_s=30)
    assert result.truncated is True


def test_stream_with_no_terminal_payload_is_incomplete(monkeypatch):
    # C1 regression: connection ends (OOM-killed model, host sleep, daemon
    # restart, proxy reset) without ever sending {"done": true}. iter_lines()
    # just stops — no exception. This must not be mistaken for a clean result.
    payloads = [{"message": {"content": c}, "done": False} for c in ["par", "tial"]]
    monkeypatch.setattr(
        "reviewer.ollama_client.requests.post",
        lambda *a, **kw: FakeResponse(payloads),
    )
    result = chat("sys", "user", deadline_s=30)
    assert result.done_reason == "incomplete"
    assert result.truncated is True


def test_deadline_aborts_and_closes_connection(monkeypatch):
    clock = {"t": 0.0}

    def fake_monotonic():
        clock["t"] += 50.0
        return clock["t"]

    response = FakeResponse(_stream(["a", "b", "c", "d"]))
    monkeypatch.setattr("reviewer.ollama_client.requests.post", lambda *a, **kw: response)
    monkeypatch.setattr("reviewer.ollama_client.time.monotonic", fake_monotonic)

    result = chat("sys", "user", deadline_s=60)
    assert result.done_reason == "timeout"
    assert result.truncated is True
    assert response.closed is True


def test_num_batch_512_is_sent(monkeypatch):
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured.update(json)
        return FakeResponse(_stream(["ok"]))

    monkeypatch.setattr("reviewer.ollama_client.requests.post", fake_post)
    chat("sys", "user", deadline_s=30)
    # Regression for B1: the old hardcoded 128 slowed prompt eval 2-4x on Metal.
    assert captured["options"]["num_batch"] == 512
    assert captured["stream"] is True
    assert captured["options"]["temperature"] == 0.1
    assert captured["options"]["seed"] == 42


def test_chat_honours_temperature_and_omits_seed(monkeypatch):
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured.update(json)
        return FakeResponse(_stream(["ok"]))

    monkeypatch.setattr("reviewer.ollama_client.requests.post", fake_post)
    chat("sys", "user", deadline_s=30, temperature=0.85, seed=None)
    assert captured["options"]["temperature"] == 0.85
    assert "seed" not in captured["options"]


def test_transport_error_returns_error_result(monkeypatch):
    def boom(*a, **kw):
        raise ConnectionError("refused")

    monkeypatch.setattr("reviewer.ollama_client.requests.post", boom)
    result = chat("sys", "user", deadline_s=30)
    assert result.done_reason == "error"
    assert "refused" in result.text


def test_clean_response_strips_harmony_tokens():
    assert clean_response("<|channel|>final<|message|>real text") == "final real text"


def test_clean_response_normalises_bare_lgtm():
    out = clean_response("[LGTM]")
    assert out == "LGTM. The changes are clean and follow best practices."


def test_clean_response_keeps_findings_alongside_lgtm():
    text = "**🔴 [BLOCKER]**\nbad\n[LGTM]"
    assert "[BLOCKER]" in clean_response(text)


def test_clean_response_swaps_the_for_meme():
    from reviewer.memes import meme_phrases
    assert clean_response("The") in meme_phrases
