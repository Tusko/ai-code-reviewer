from reviewer.chat_types import (
    FINDING_TAGS, LGTM_TEXT, ChatResult, clean_response,
)


def test_chat_result_failed_covers_ratelimit():
    result = ChatResult("", "ratelimit", 0, 0, 0.0)
    assert result.failed is True


def test_chat_result_truncated_covers_length():
    result = ChatResult("x", "length", 0, 0, 0.0)
    assert result.truncated is True
    assert result.failed is False


def test_clean_response_normalises_bare_lgtm():
    assert clean_response("**LGTM.**") == LGTM_TEXT


def test_clean_response_drops_degenerate_reply():
    assert clean_response("the") == ""


def test_clean_response_preserves_fenced_indentation():
    text = "**🔴 [BLOCKER]** broken\n\n*Fix:*\n```py\nif x:\n    pass\n```"
    assert "\n    pass\n" in clean_response(text)


def test_finding_tags_exported():
    assert FINDING_TAGS == ("[BLOCKER]", "[SUGGESTION]", "[NIT]")
