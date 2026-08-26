import json

import pytest

from reviewer import ledger as ledger_mod
from reviewer.diff_parser import Hunk
from reviewer.ledger import (
    Ledger, LedgerStore, LedgerUnavailable, hunk_key, parse_marker, render_note,
)


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


def test_parse_marker_raises_when_hunks_is_not_a_list():
    payload = {"hunks": "not-a-list"}
    body = f"<!-- {ledger_mod.MARKER_PREFIX} {json.dumps(payload)} -->"
    with pytest.raises(LedgerUnavailable):
        ledger_mod.parse_marker(body)


def test_parse_marker_raises_when_hunks_contains_non_string():
    payload = {"hunks": ["aa", 5]}
    body = f"<!-- {ledger_mod.MARKER_PREFIX} {json.dumps(payload)} -->"
    with pytest.raises(LedgerUnavailable):
        ledger_mod.parse_marker(body)


def test_parse_marker_raises_when_muted_is_a_string():
    payload = {"muted": "false"}
    body = f"<!-- {ledger_mod.MARKER_PREFIX} {json.dumps(payload)} -->"
    with pytest.raises(LedgerUnavailable):
        ledger_mod.parse_marker(body)


def test_parse_marker_raises_when_posted_is_a_string():
    payload = {"posted": "abc"}
    body = f"<!-- {ledger_mod.MARKER_PREFIX} {json.dumps(payload)} -->"
    with pytest.raises(LedgerUnavailable):
        ledger_mod.parse_marker(body)


def test_parse_marker_raises_when_posted_is_a_bool():
    payload = {"posted": True}
    body = f"<!-- {ledger_mod.MARKER_PREFIX} {json.dumps(payload)} -->"
    with pytest.raises(LedgerUnavailable):
        ledger_mod.parse_marker(body)


def test_parse_marker_raises_when_head_is_a_number():
    payload = {"head": 123}
    body = f"<!-- {ledger_mod.MARKER_PREFIX} {json.dumps(payload)} -->"
    with pytest.raises(LedgerUnavailable):
        ledger_mod.parse_marker(body)


def test_parse_marker_accepts_a_fully_valid_marker_unchanged():
    original = Ledger(
        head="abc1234", posted=7, muted=True, oversized=True, hunks=("aa", "bb"),
    )
    payload = {
        "head": original.head,
        "posted": original.posted,
        "muted": original.muted,
        "oversized": original.oversized,
        "hunks": list(original.hunks),
    }
    body = f"<!-- {ledger_mod.MARKER_PREFIX} {json.dumps(payload)} -->"
    assert ledger_mod.parse_marker(body) == original


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


class FakeNote:
    def __init__(self, body=""):
        self.body = body
        self.saves = 0

    def save(self):
        self.saves += 1


class FakeNotes:
    def __init__(self, notes=(), raises=False):
        self._notes = list(notes)
        self._raises = raises
        self.created = []

    def list(self, iterator=False):
        if self._raises:
            raise RuntimeError("gitlab is down")
        return list(self._notes)

    def create(self, payload):
        note = FakeNote(payload["body"])
        self._notes.append(note)
        self.created.append(payload["body"])
        return note


class FakeMR:
    def __init__(self, notes=(), raises=False):
        self.notes = FakeNotes(notes, raises)


def test_store_load_fresh_when_no_marker():
    store = LedgerStore.load(FakeMR([FakeNote("unrelated chatter")]))
    assert store.ledger == Ledger()


def test_store_load_reads_existing_marker():
    body = ledger_mod.render_note(Ledger(head="abc1234", posted=4, hunks=("aa",)))
    store = LedgerStore.load(FakeMR([FakeNote("noise"), FakeNote(body)]))
    assert store.ledger.posted == 4
    assert store.ledger.hunks == ("aa",)


def test_store_load_raises_when_api_fails():
    with pytest.raises(LedgerUnavailable):
        LedgerStore.load(FakeMR(raises=True))


def test_store_load_raises_on_broken_marker():
    mr = FakeMR([FakeNote(f"<!-- {ledger_mod.MARKER_PREFIX} {{broken}} -->")])
    with pytest.raises(LedgerUnavailable):
        LedgerStore.load(mr)


def test_store_save_creates_note_once_then_edits():
    mr = FakeMR()
    store = LedgerStore.load(mr)
    store.ledger = store.ledger.spend(1)
    store.save()
    assert len(mr.notes.created) == 1

    store.ledger = store.ledger.spend(1)
    store.save()
    assert len(mr.notes.created) == 1, "second save must edit, not post again"
    assert ledger_mod.parse_marker(mr.notes._notes[0].body).posted == 2


def test_every_field_survives_the_marker_round_trip():
    """A field dropped from to_marker never persists in production, and the
    suite stayed green through it because the fake store held the object."""
    from dataclasses import fields
    full = Ledger(
        head="abc1234", posted=7, muted=True, oversized=True,
        outage_reported=True, hunks=("aaaaaaaaaaaa",), retried=("bbbbbbbbbbbb",),
    )
    back = parse_marker(render_note(full))
    for f in fields(Ledger):
        assert getattr(back, f.name) == getattr(full, f.name), f.name


def test_the_new_fields_fail_closed_on_a_wrong_shape():
    """Same rule as every other field: a present-but-wrong value must raise,
    never be coerced into something that looks like a valid empty ledger."""
    for payload, field in (('{"outage_reported": "yes"}', "outage_reported"),
                           ('{"outage_reported": 1}', "outage_reported"),
                           ('{"retried": "not-a-list"}', "retried"),
                           ('{"retried": [1, 2]}', "retried")):
        body = f"<!-- sidorovich-state:v1 {payload} -->"
        with pytest.raises(LedgerUnavailable):
            parse_marker(body)
