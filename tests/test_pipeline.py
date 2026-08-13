import pytest

from reviewer.diff_parser import FileDiff, Hunk
from reviewer.pipeline import (
    FileOutcome, build_prompt_ladder, render_summary, select_files,
)


def fd(path, added=1, **kwargs):
    lines = [" ctx"] + [f"+line{i}" for i in range(added)]
    return FileDiff(
        old_path=path, new_path=path,
        is_new=False,
        is_deleted=kwargs.get("is_deleted", False),
        is_renamed=False,
        is_binary=kwargs.get("is_binary", False),
        hunks=kwargs.get("hunks", (Hunk(1, 1, tuple(lines)),)),
    )


def test_select_files_orders_smallest_first():
    kept, _ = select_files([fd("big.py", added=50), fd("small.py", added=2)])
    assert [f.new_path for f in kept] == ["small.py", "big.py"]


def test_select_files_reports_filtered_files():
    kept, outcomes = select_files([fd("src/a.py"), fd("package-lock.json")])
    assert [f.new_path for f in kept] == ["src/a.py"]
    assert outcomes[0].path == "package-lock.json"
    assert outcomes[0].status == "skipped"
    assert outcomes[0].detail == "lockfile"


def test_select_files_caps_at_max_files(monkeypatch):
    monkeypatch.setattr("reviewer.config.MAX_FILES", 2)
    kept, outcomes = select_files([fd(f"f{i}.py", added=i + 1) for i in range(5)])
    assert len(kept) == 2
    over = [o for o in outcomes if o.detail == "over MAX_FILES limit"]
    assert len(over) == 3


def test_ladder_starts_at_l1_when_context_disabled(monkeypatch):
    monkeypatch.setattr("reviewer.config.INCLUDE_FILE_CONTEXT", False)
    ladder = build_prompt_ladder("a.py", [Hunk(1, 1, (" a", "+b"))], context="ctx text")
    assert ladder[0][0] == "L1"
    assert "ctx text" not in ladder[0][1]


def test_ladder_starts_at_l0_when_context_enabled(monkeypatch):
    monkeypatch.setattr("reviewer.config.INCLUDE_FILE_CONTEXT", True)
    ladder = build_prompt_ladder("a.py", [Hunk(1, 1, (" a", "+b"))], context="ctx text")
    assert ladder[0][0] == "L0"
    assert "ctx text" in ladder[0][1]


def test_ladder_adds_per_hunk_level_for_multi_hunk_files(monkeypatch):
    monkeypatch.setattr("reviewer.config.INCLUDE_FILE_CONTEXT", False)
    hunks = [Hunk(1, 1, (" a", "+b")), Hunk(1, 40, (" c", "+d"))]
    ladder = build_prompt_ladder("a.py", hunks, context="")
    assert [level for level, _ in ladder] == ["L1", "L2", "L2"]


def test_ladder_has_no_l2_for_single_hunk_files(monkeypatch):
    monkeypatch.setattr("reviewer.config.INCLUDE_FILE_CONTEXT", False)
    ladder = build_prompt_ladder("a.py", [Hunk(1, 1, (" a", "+b"))], context="")
    assert [level for level, _ in ladder] == ["L1"]


def test_render_summary_lists_every_category():
    summary = render_summary([
        FileOutcome("a.py", "reviewed", "2 findings"),
        FileOutcome("b.py", "clean", ""),
        FileOutcome("huge.py", "skipped", "single hunk exceeds context budget"),
        FileOutcome("c.py", "error", "timeout"),
    ])
    assert "a.py" in summary
    assert "huge.py" in summary
    assert "single hunk exceeds context budget" in summary
    assert "c.py" in summary
    assert "timeout" in summary


def test_render_summary_of_all_clean_says_lgtm():
    summary = render_summary([FileOutcome("a.py", "clean", "")])
    assert "LGTM" in summary
