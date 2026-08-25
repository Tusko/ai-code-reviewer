import pytest

from reviewer import ledger as ledger_mod
from reviewer.diff_parser import Hunk
from reviewer.ledger import Ledger, LedgerUnavailable, hunk_key


def test_hunk_key_ignores_line_number_shift():
    """A rebase moves hunks without changing them. The key must not move."""
    lines = (" ctx", "+added", " tail")
    assert hunk_key("a.py", Hunk(1, 1, lines)) == hunk_key("a.py", Hunk(90, 90, lines))


def test_hunk_key_changes_with_content():
    a = hunk_key("a.py", Hunk(1, 1, (" ctx", "+added")))
    b = hunk_key("a.py", Hunk(1, 1, (" ctx", "+altered")))
    assert a != b


def test_hunk_key_changes_with_path():
    lines = (" ctx", "+added")
    assert hunk_key("a.py", Hunk(1, 1, lines)) != hunk_key("b.py", Hunk(1, 1, lines))


def test_marker_round_trip():
    original = Ledger(
        head="abc1234", posted=7, muted=True, oversized=True, hunks=("aa", "bb"),
    )
    assert ledger_mod.parse_marker(ledger_mod.to_marker(original)) == original


def test_parse_marker_returns_fresh_ledger_when_absent():
    assert ledger_mod.parse_marker("just a normal comment") == Ledger()


def test_parse_marker_raises_on_broken_json():
    body = f"<!-- {ledger_mod.MARKER_PREFIX} {{not json}} -->"
    with pytest.raises(LedgerUnavailable):
        ledger_mod.parse_marker(body)


def test_record_is_idempotent():
    value = Ledger().record(["aa", "bb"]).record(["bb", "cc"])
    assert value.hunks == ("aa", "bb", "cc")


def test_record_evicts_oldest_past_cap(monkeypatch):
    monkeypatch.setattr("reviewer.config.LEDGER_MAX_HUNKS", 3)
    value = Ledger().record(["a", "b", "c", "d"])
    assert value.hunks == ("b", "c", "d")


def test_remaining_counts_down_from_budget(monkeypatch):
    monkeypatch.setattr("reviewer.config.MR_COMMENT_BUDGET", 30)
    assert Ledger(posted=28).remaining() == 2


def test_remaining_never_negative(monkeypatch):
    monkeypatch.setattr("reviewer.config.MR_COMMENT_BUDGET", 30)
    assert Ledger(posted=99).remaining() == 0


def test_unmute_and_reset_clears_both():
    value = Ledger(posted=30, muted=True).unmute_and_reset()
    assert value.posted == 0
    assert value.muted is False


def test_render_note_embeds_a_parseable_marker():
    value = Ledger(head="abc1234", posted=3, hunks=("aa",))
    assert ledger_mod.parse_marker(ledger_mod.render_note(value)) == value


def test_render_note_is_human_readable():
    body = ledger_mod.render_note(Ledger(head="abc1234", posted=3))
    assert "Сідорович" in body
    assert "abc1234" in body
