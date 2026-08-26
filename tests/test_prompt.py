from reviewer import config
from reviewer import prompt as prompt_mod
from reviewer.diff_parser import Hunk
from reviewer.memes import (
    closer_for, opener_for, sidorovich_closers, sidorovich_openers,
)
from reviewer.prompt import (
    SYSTEM_PROMPT, SIDOROVICH_SYSTEM_PROMPT, SIDOROVICH_REVIEW_VOICE_PROMPT,
    build_commit_summary_prompt,
    build_file_prompt, estimate_tokens, extract_ticket_key, fits,
    has_foreign_script, input_token_budget, looks_too_russian,
    render_hunk, unusable_language,
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
    assert budget <= config.REVIEW_CONTEXT_TOKENS - config.REVIEW_MAX_OUTPUT_TOKENS


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


def test_looks_too_russian_ignores_fenced_code():
    """`если` inside a *Fix:* block is the reviewed code, not Sidorovich."""
    assert not looks_too_russian(
        "Тут ти обісрався з валідацією.\n"
        "*Fix:*\n"
        "```python\n"
        'if lang == "если": raise ValueError("это")\n'
        "```"
    )


def test_input_token_budget_follows_review_context(monkeypatch):
    monkeypatch.setattr("reviewer.config.REVIEW_CONTEXT_TOKENS", 262144)
    monkeypatch.setattr("reviewer.config.REVIEW_MAX_OUTPUT_TOKENS", 4096)
    monkeypatch.setattr("reviewer.config.PROMPT_TOKEN_BUFFER", 128)
    budget = prompt_mod.input_token_budget()
    assert budget > 250_000


def test_input_token_budget_ignores_ollama_context(monkeypatch):
    monkeypatch.setattr("reviewer.config.REVIEW_CONTEXT_TOKENS", 262144)
    monkeypatch.setattr("reviewer.config.REVIEW_MAX_OUTPUT_TOKENS", 4096)
    before = prompt_mod.input_token_budget()
    monkeypatch.setattr("reviewer.config.OLLAMA_NUM_CTX", 512)
    assert prompt_mod.input_token_budget() == before


def test_input_token_budget_floors_at_256(monkeypatch):
    monkeypatch.setattr("reviewer.config.REVIEW_CONTEXT_TOKENS", 100)
    monkeypatch.setattr("reviewer.config.REVIEW_MAX_OUTPUT_TOKENS", 90)
    assert prompt_mod.input_token_budget() == 256


def test_the_prompt_no_longer_mandates_a_swear_opener():
    """"Починай з матюка" collapsed every release roast onto the same word:
    the prompt is byte-identical each call, so the first token went to the
    mode every time."""
    assert "Починай з матюка" not in SIDOROVICH_SYSTEM_PROMPT
    # The literal moral example was copied verbatim into the output too.
    assert "шукайте собі нову хату" not in SIDOROVICH_SYSTEM_PROMPT


def test_the_opener_is_injected_and_quoted_once():
    text = build_commit_summary_prompt(
        [{"short_id": "a", "title": "MONO-1 fix", "author": "x"}],
        opener="Оце номер.",
    )
    assert "«Оце номер.»" in text
    assert "MONO-1" in text


def test_no_opener_leaves_the_prompt_alone():
    text = build_commit_summary_prompt(
        [{"short_id": "a", "title": "MONO-1 fix", "author": "x"}],
    )
    assert "зачин" not in text.lower()


def test_opener_for_is_deterministic_and_spread():
    """Deterministic because the Ukrainian retry re-sends the same request:
    an opener that changed between attempts reads as a different person."""
    assert opener_for("abc") == opener_for("abc")
    seen = {opener_for(f"fingerprint-{i}") for i in range(400)}
    assert len(seen) > len(sidorovich_openers) * 0.8, (
        f"only {len(seen)} of {len(sidorovich_openers)} openers ever chosen"
    )


def test_no_opener_starts_with_the_word_it_replaced():
    assert not any(o.lower().startswith("блять") for o in sidorovich_openers)


def test_the_language_rule_no_longer_donates_a_roast_opener():
    """The "Добре:" example was a complete opening sentence, and the model
    copied it verbatim: three roasts in a sampled ten opened on it."""
    assert "недолугий висер у репозиторій" not in SIDOROVICH_SYSTEM_PROMPT
    assert "чайник" in SIDOROVICH_SYSTEM_PROMPT, "the minimal pair must survive"
    assert "этот" in SIDOROVICH_SYSTEM_PROMPT, "it still has to teach the failure"


def test_the_worn_out_tics_are_banned():
    for tic in ("без нормального рев'ю", "пішли всі нахуй",
                "шукайте собі нову роботу"):
        assert tic in SIDOROVICH_SYSTEM_PROMPT, f"{tic!r} must be named as banned"
    assert 'Не починай останнє речення зі слова "Якщо"' in SIDOROVICH_SYSTEM_PROMPT


def test_the_closer_mode_is_injected():
    text = build_commit_summary_prompt(
        [{"short_id": "a", "title": "MONO-1 fix", "author": "x"}],
        opener="Тю.", closer="ультиматум з дедлайном",
    )
    assert "ультиматум з дедлайном" in text
    assert "«Тю.»" in text


def test_opener_and_closer_rotate_independently():
    """Sharing one hash byte would pin every opener to one closer forever."""
    pairs = {(opener_for(f"s{i}"), closer_for(f"s{i}")) for i in range(600)}
    assert len(pairs) > len(sidorovich_openers) * 3, (
        f"only {len(pairs)} distinct opener/closer pairs"
    )


def test_looks_too_russian_catches_the_words_real_roasts_shipped():
    """Every one of these went out to a merge request unchallenged."""
    for bad in ("тепер хоч не буде різати глаза",
                "якщо це все развалиться на проді",
                "намагаються оптимизувати автотести",
                "випихали цей дебильний дефіс",
                "хоч це и так хуйня"):
        assert looks_too_russian(bad), bad


def test_looks_too_russian_still_allows_the_surzhyk_it_is_meant_to_keep():
    for good in ("Опять цей недолугий висер закинули без спросу",
                 "накодили якоїсь хуйні в авторизації",
                 "оптимізація автотестів"):
        assert not looks_too_russian(good), good


def test_has_foreign_script_catches_the_alphabet_leaks():
    """Both of these reached a real merge request."""
    assert has_foreign_script("MONO-1532: هاي ця фігня з лоуеркейсом")
    assert has_foreign_script("MONO-1536: наፈላли повідомлень про домени")


def test_has_foreign_script_leaves_emoji_and_latin_alone():
    assert not has_foreign_script("### 📄 `a.py`\n\n**🔴 [BLOCKER]** boom")
    assert not has_foreign_script("Ну шо, TypeScript вам не поміг — 200 OK і все.")


def test_has_foreign_script_ignores_fenced_code():
    """A *Fix:* block may legitimately quote a string in any language, and
    losing the finding's code over it is worse than the leak."""
    assert not has_foreign_script(
        'Полагодь це:\n```python\nGREETING = "مرحبا"\n```\nІ не пхай більше.'
    )


def test_unusable_language_covers_both_failure_modes():
    assert unusable_language("это провал")
    assert unusable_language("наፈላли")
    assert not unusable_language("накодили якоїсь хуйні")
