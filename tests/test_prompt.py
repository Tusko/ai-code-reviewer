from reviewer.diff_parser import Hunk
from reviewer.prompt import (
    SYSTEM_PROMPT, SIDOROVICH_SYSTEM_PROMPT, SIDOROVICH_REVIEW_VOICE_PROMPT,
    build_commit_summary_prompt,
    build_file_prompt, estimate_tokens, extract_ticket_key, fits,
    input_token_budget, looks_too_russian, render_hunk,
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


def test_sidorovich_prompt_keeps_format_rules():
    assert "Сідорович" in SIDOROVICH_SYSTEM_PROMPT
    assert "Ось короткий огляд" in SIDOROVICH_SYSTEM_PROMPT
    assert "MONO-123" in SIDOROVICH_SYSTEM_PROMPT
    assert "100–150" in SIDOROVICH_SYSTEM_PROMPT
    assert "Не пиши російською" in SIDOROVICH_SYSTEM_PROMPT
    assert "этот" in SIDOROVICH_SYSTEM_PROMPT


def test_sidorovich_review_voice_keeps_findings_intact():
    assert "Сідорович" in SIDOROVICH_REVIEW_VOICE_PROMPT
    assert "Не додавай і не викидай знахідок" in SIDOROVICH_REVIEW_VOICE_PROMPT
    assert "*Fix:*" in SIDOROVICH_REVIEW_VOICE_PROMPT
    assert "[BLOCKER]" in SIDOROVICH_REVIEW_VOICE_PROMPT
    assert "этот" in SIDOROVICH_REVIEW_VOICE_PROMPT


def test_looks_too_russian_catches_pure_russian_sidorovich():
    assert looks_too_russian(
        "Опять этот недоделанный высер в репозиторий закинули без спроса."
    )
    assert not looks_too_russian(
        "Опять цей недолугий висер у репозиторій закинули без спросу."
    )


def test_extract_ticket_key_from_title():
    assert extract_ticket_key("MONO-123 fix feed crash") == "MONO-123"
    assert extract_ticket_key("[MONO-456] auth hotfix") == "MONO-456"
    assert extract_ticket_key("feat: ABC-7 do thing") == "ABC-7"
    assert extract_ticket_key("no ticket here") is None


def test_build_commit_summary_prompt_lists_commits():
    text = build_commit_summary_prompt([
        {"short_id": "abc1234", "author": "Ivan", "title": "MONO-123 fix feed crash"},
        {"short_id": "def5678", "author": "Oksana", "title": "auth hotfix"},
    ])
    assert "MONO-123: fix feed crash (Ivan)" in text
    assert "- auth hotfix (Oksana)" in text
    assert "abc1234" not in text


def test_build_commit_summary_prompt_caps_long_lists():
    commits = [
        {"short_id": f"{i:07d}", "author": "dev", "title": f"c{i}"}
        for i in range(45)
    ]
    text = build_commit_summary_prompt(commits)
    assert "- c0 (dev)" in text
    assert "- c39 (dev)" in text
    assert "- c44 (dev)" not in text
    assert "ще 5 коміт" in text
