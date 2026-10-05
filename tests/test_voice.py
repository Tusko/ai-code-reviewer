from reviewer import voice
from reviewer.chat_types import ChatResult
from reviewer.memes import review_opener_for
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


def test_flavor_review_hands_the_model_a_rotated_opener(monkeypatch):
    """Without this the rewrite opened almost every finding on "Якого хуя"."""
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr("reviewer.config.SNARK", True)
    sent = {}

    def fake_chat(system, user, **kwargs):
        sent["user"] = user
        return _result("**🔴 [BLOCKER]** та шо ж ти робиш")

    monkeypatch.setattr(voice.openrouter_client, "chat", fake_chat)
    dry = "**🔴 [BLOCKER]** boom"
    voice.flavor_review(dry)
    assert f"«{review_opener_for(dry)}»" in sent["user"]
    assert dry in sent["user"]


def test_flavor_review_replays_the_same_opener_on_the_ukrainian_retry(monkeypatch):
    """A retry that re-rolled the opener would read as a different person."""
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr("reviewer.config.SNARK", True)
    users = []
    replies = [
        _result("**🔴 [BLOCKER]** опять этот высер"),
        _result("**🔴 [BLOCKER]** та шо ж ти робиш"),
    ]

    def fake_chat(system, user, **kwargs):
        history = kwargs.get("history") or ()
        users.append(history[0]["content"] if history else user)
        return replies.pop(0)

    monkeypatch.setattr(voice.openrouter_client, "chat", fake_chat)
    voice.flavor_review("**🔴 [BLOCKER]** boom")
    assert len(users) == 2
    assert users[0] == users[1]


def test_flavor_review_keeps_dry_when_voice_disabled(monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr("reviewer.config.SNARK", True)
    state = VoiceState()
    for _ in range(voice.VOICE_FAILURE_LIMIT):
        state.record_failure()
    assert voice.flavor_review("**🔴 [BLOCKER]** boom", state) == "**🔴 [BLOCKER]** boom"
