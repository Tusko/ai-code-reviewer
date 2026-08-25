from reviewer import voice
from reviewer.chat_types import ChatResult
from reviewer.voice import VoiceState, preserves_findings


def _result(text, done_reason="stop"):
    return ChatResult(text, done_reason, 0, 0, 0.0)


def test_voice_state_opens_closed_after_limit():
    state = VoiceState()
    assert state.enabled is True
    for _ in range(voice.VOICE_FAILURE_LIMIT):
        state.record_failure()
    assert state.enabled is False


def test_voice_state_success_resets_failures():
    state = VoiceState()
    state.record_failure()
    state.record_success()
    assert state.enabled is True


def test_preserves_findings_rejects_dropped_tag():
    original = "**🔴 [BLOCKER]** boom"
    assert preserves_findings(original, "нема нічого") is False


def test_preserves_findings_rejects_invented_fence():
    original = "**🔵 [NIT]** cast it"
    assert preserves_findings(original, "**🔵 [NIT]** хуйня\n```py\nx = 1\n```") is False


def test_preserves_findings_accepts_identical_fences():
    original = "**🟡 [SUGGESTION]** fix\n```py\nx = 1\n```"
    flavored = "**🟡 [SUGGESTION]** блять, полагодь\n```py\nx = 1\n```"
    assert preserves_findings(original, flavored) is True


def test_flavor_review_returns_text_unchanged_without_key(monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    assert voice.flavor_review("**🔴 [BLOCKER]** boom") == "**🔴 [BLOCKER]** boom"


def test_flavor_review_keeps_dry_when_voice_disabled(monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr("reviewer.config.SNARK", True)
    state = VoiceState()
    for _ in range(voice.VOICE_FAILURE_LIMIT):
        state.record_failure()
    assert voice.flavor_review("**🔴 [BLOCKER]** boom", state) == "**🔴 [BLOCKER]** boom"
