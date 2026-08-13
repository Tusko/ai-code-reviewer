import pytest

from reviewer import pipeline
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.ollama_client import ChatResult
from reviewer.pipeline import (
    FileOutcome, build_prompt_ladder, render_summary, review_file,
    review_merge_request, select_files,
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


class FakeMR:
    """Placeholder MR object. review_file never reads its attributes directly —
    gitlab_client calls that would need them (post_inline, post_note) are mocked
    out in the tests below."""


def _chat_result(text, done_reason="stop"):
    return ChatResult(
        text=text, done_reason=done_reason,
        prompt_eval_count=0, eval_count=0, elapsed_s=0.1,
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


def test_review_file_skips_when_no_ladder_level_fits(monkeypatch):
    monkeypatch.setattr(pipeline.prompt_mod, "fits", lambda text: False)
    outcome = review_file(FakeMR(), fd("a.py"), context="")
    assert outcome.status == "skipped"
    assert outcome.detail


def test_review_file_reports_error_on_chat_failure(monkeypatch):
    monkeypatch.setattr(
        pipeline, "chat",
        lambda system, user, deadline_s: _chat_result("boom", done_reason="error"),
    )
    outcome = review_file(FakeMR(), fd("a.py"), context="")
    assert outcome.status == "error"


def test_review_file_reports_error_on_timeout(monkeypatch):
    monkeypatch.setattr(
        pipeline, "chat",
        lambda system, user, deadline_s: _chat_result("partial", done_reason="timeout"),
    )
    outcome = review_file(FakeMR(), fd("a.py"), context="")
    assert outcome.status == "error"
    assert "timeout" in outcome.detail


def test_review_file_reports_error_on_incomplete_stream(monkeypatch):
    # C1 regression: an interrupted stream (no terminal payload) must never
    # be reported as "clean" — an absence of signal is not a positive result.
    monkeypatch.setattr(
        pipeline, "chat",
        lambda system, user, deadline_s: _chat_result("", done_reason="incomplete"),
    )
    outcome = review_file(FakeMR(), fd("a.py"), context="")
    assert outcome.status == "error"
    assert outcome.detail == "no response from model"


def test_review_file_reports_error_on_empty_response(monkeypatch):
    # C1 regression: done_reason == "stop" with empty text is also an absence
    # of signal, not a clean verdict.
    monkeypatch.setattr(
        pipeline, "chat",
        lambda system, user, deadline_s: _chat_result("", done_reason="stop"),
    )
    outcome = review_file(FakeMR(), fd("a.py"), context="")
    assert outcome.status == "error"
    assert outcome.detail == "no response from model"


def test_review_file_is_clean_when_response_starts_with_lgtm(monkeypatch):
    monkeypatch.setattr(
        pipeline, "chat",
        lambda system, user, deadline_s: _chat_result("LGTM. Looks fine."),
    )
    inline_calls = []
    note_calls = []
    monkeypatch.setattr(pipeline.gitlab_client, "post_inline",
                         lambda *a, **k: inline_calls.append((a, k)) or True)
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                         lambda *a, **k: note_calls.append((a, k)))

    outcome = review_file(FakeMR(), fd("a.py"), context="")

    assert outcome.status == "clean"
    assert inline_calls == []
    assert note_calls == []


def test_review_file_posts_inline_when_finding_and_position_accepted(monkeypatch):
    monkeypatch.setattr(
        pipeline, "chat",
        lambda system, user, deadline_s: _chat_result("**🔴 [BLOCKER]**\nSomething bad."),
    )
    inline_calls = []
    note_calls = []
    monkeypatch.setattr(pipeline.gitlab_client, "post_inline",
                         lambda mr, path, line, body: inline_calls.append((mr, path, line, body)) or True)
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                         lambda mr, body: note_calls.append((mr, body)))

    file_diff = fd("a.py")
    outcome = review_file(FakeMR(), file_diff, context="")

    assert outcome.status == "reviewed"
    assert len(inline_calls) == 1
    assert inline_calls[0][2] == file_diff.hunks[0].first_added_line()
    assert note_calls == []


def test_review_file_falls_back_to_note_when_inline_rejected(monkeypatch):
    monkeypatch.setattr(
        pipeline, "chat",
        lambda system, user, deadline_s: _chat_result("**🔴 [BLOCKER]**\nSomething bad."),
    )
    inline_calls = []
    note_calls = []
    monkeypatch.setattr(pipeline.gitlab_client, "post_inline",
                         lambda mr, path, line, body: inline_calls.append((mr, path, line, body)) or False)
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                         lambda mr, body: note_calls.append((mr, body)))

    outcome = review_file(FakeMR(), fd("a.py"), context="")

    assert outcome.status == "reviewed"
    assert len(inline_calls) == 1
    assert len(note_calls) == 1
    assert note_calls[0][1] == inline_calls[0][3]


def test_review_file_marks_truncated_response_but_still_posts(monkeypatch):
    monkeypatch.setattr(
        pipeline, "chat",
        lambda system, user, deadline_s: _chat_result(
            "**🔴 [BLOCKER]**\nSomething bad but cut off", done_reason="length",
        ),
    )
    note_calls = []
    monkeypatch.setattr(pipeline.gitlab_client, "post_inline", lambda *a, **k: False)
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                         lambda mr, body: note_calls.append(body))

    outcome = review_file(FakeMR(), fd("a.py"), context="")

    assert outcome.status == "reviewed"
    assert "truncated" in outcome.detail
    assert len(note_calls) == 1
    assert ("_⚠️ This review was truncated at the output token limit "
            "and may be incomplete._") in note_calls[0]


def test_review_file_reports_partial_l2_coverage_when_hunk_exceeds_budget(monkeypatch):
    # I3 regression: at L2, hunks that don't fit must be named in the outcome
    # detail rather than silently vanishing.
    ladder = [("L1", "l1-text"), ("L2", "hunk1-text"), ("L2", "hunk2-text")]
    monkeypatch.setattr(pipeline, "build_prompt_ladder", lambda path, hunks, context: ladder)
    monkeypatch.setattr(pipeline.prompt_mod, "fits",
                         lambda text: text in ("hunk1-text",))
    monkeypatch.setattr(
        pipeline, "chat",
        lambda system, user, deadline_s: _chat_result("**🔴 [BLOCKER]**\nbad"),
    )
    monkeypatch.setattr(pipeline.gitlab_client, "post_inline", lambda *a, **k: True)
    monkeypatch.setattr(pipeline.gitlab_client, "post_note", lambda *a, **k: None)

    two_hunks = (Hunk(1, 1, (" x", "+y")), Hunk(1, 40, (" z", "+w")))
    outcome = review_file(FakeMR(), fd("a.py", hunks=two_hunks), context="")

    assert outcome.status == "reviewed"
    assert outcome.detail == "1 of 2 hunks reviewed; 1 hunk exceeds context budget"


def test_review_merge_request_marks_all_files_skipped_when_deadline_passed(monkeypatch):
    monkeypatch.setattr(pipeline.config, "MR_TIMEOUT_S", 0)

    mr = FakeMR()
    project = object()
    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr",
                         lambda project_id, mr_iid: (project, mr))
    monkeypatch.setattr(pipeline.gitlab_client, "fetch_file_diffs",
                         lambda mr: [fd("a.py"), fd("b.py")])

    posted = []
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                         lambda mr, body: posted.append(body))

    def fail_review_file(mr, file_diff, context):
        raise AssertionError("review_file should not run once the MR deadline has passed")
    monkeypatch.setattr(pipeline, "review_file", fail_review_file)

    review_merge_request(project_id=1, mr_iid=2)

    assert len(posted) == 1
    summary = posted[0]
    assert "a.py" in summary
    assert "b.py" in summary
    assert summary.count("MR deadline of 0s reached") == 2
