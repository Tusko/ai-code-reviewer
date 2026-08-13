from reviewer.diff_parser import Hunk
from reviewer.prompt import (
    SYSTEM_PROMPT, build_file_prompt, estimate_tokens, fits,
    input_token_budget, render_hunk,
)

HUNK = Hunk(
    old_start=10,
    new_start=10,
    lines=(" def handler(req):", "-    q = req.q", "+    q = sanitize(req.q)", "+    return run(q)"),
)


def test_render_hunk_omits_removed_lines():
    rendered = render_hunk(HUNK)
    assert "q = req.q" not in rendered
    assert "sanitize(req.q)" in rendered


def test_render_hunk_numbers_lines_from_new_start():
    rendered = render_hunk(HUNK)
    lines = rendered.splitlines()
    assert lines[0].strip().startswith("10")
    assert "11 +" in lines[1]
    assert "12 +" in lines[2]


def test_render_hunk_marks_added_lines():
    rendered = render_hunk(HUNK)
    added = [line for line in rendered.splitlines() if " + " in line]
    assert len(added) == 2


def test_build_file_prompt_includes_path_and_hunk():
    prompt = build_file_prompt("src/api.py", [HUNK])
    assert "src/api.py" in prompt
    assert "sanitize(req.q)" in prompt


def test_build_file_prompt_omits_context_block_when_empty():
    prompt = build_file_prompt("src/api.py", [HUNK])
    assert "FILE CONTEXT" not in prompt


def test_build_file_prompt_includes_context_block_when_given():
    prompt = build_file_prompt("src/api.py", [HUNK], context="def run(q): ...")
    assert "FILE CONTEXT" in prompt
    assert "def run(q): ..." in prompt


def test_system_prompt_keeps_ban_rules():
    assert "NEVER complain about" in SYSTEM_PROMPT
    assert "[LGTM]" in SYSTEM_PROMPT
    assert "[BLOCKER]" in SYSTEM_PROMPT


def test_estimate_tokens_is_never_zero():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcdef") == 2


def test_input_token_budget_leaves_room_for_output(monkeypatch):
    budget = input_token_budget()
    assert budget > 0
    assert budget < 8192


def test_fits_rejects_oversized_prompt():
    assert fits("x") is True
    assert fits("x" * 10_000_000) is False
