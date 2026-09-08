import pytest

from review_server import (
    build_hygiene_comment,
    check_mr_hygiene,
    has_matching_hygiene_note,
    hygiene_marker,
    is_ignored_branch,
    is_valid_mr_title,
)


# ---------- title format ----------

@pytest.mark.parametrize("title", [
    "MONO-1628: reuse browser tabs for performance",
    "ABC-1: bump deps",
    "Draft: MONO-1628: reuse browser tabs",
    "WIP: MONO-1628: reuse browser tabs",
])
def test_accepts_ticket_prefixed_titles(title):
    assert is_valid_mr_title(title) is True


@pytest.mark.parametrize("title", [
    "fix(MONO-1628): reuse browser tabs for performance",
    "MONO-1628 reuse browser tabs",
    "reuse browser tabs",
    "mono-1628: reuse browser tabs",
    "MONO-1628:",
    "",
])
def test_rejects_titles_without_bare_ticket_prefix(title):
    assert is_valid_mr_title(title) is False


# ---------- ignored branches ----------

@pytest.mark.parametrize("branch", ["release/1.2.0", "hotfix/MONO-1", "release/", "hotfix/"])
def test_ignores_release_and_hotfix_branches(branch):
    assert is_ignored_branch(branch) is True


@pytest.mark.parametrize("branch", [
    "feature/MONO-1628",
    "feat/release/tabs",
    "my-hotfix/thing",
    "release",
    "",
])
def test_does_not_ignore_other_branches(branch):
    assert is_ignored_branch(branch) is False


# ---------- hygiene checks ----------

def test_clean_mr_has_no_issues():
    assert check_mr_hygiene("MONO-1628: reuse tabs", ["dev"], []) == []


def test_bad_title_reported():
    assert check_mr_hygiene("fix(MONO-1628): reuse tabs", ["dev"], []) == ["title"]


def test_no_assignee_and_no_reviewer_reported():
    assert check_mr_hygiene("MONO-1628: reuse tabs", [], []) == ["assign"]


def test_reviewer_alone_satisfies_assignment():
    assert check_mr_hygiene("MONO-1628: reuse tabs", [], ["reviewer"]) == []


def test_both_issues_reported_in_stable_order():
    assert check_mr_hygiene("bad title", [], []) == ["title", "assign"]


# ---------- comment rendering ----------

def test_marker_lists_codes_in_stable_order():
    assert hygiene_marker(["title", "assign"]) == "<!-- ai-reviewer:hygiene:title,assign -->"


def test_comment_embeds_marker_and_offending_title():
    body = build_hygiene_comment(["title"], "fix(MONO-1628): reuse tabs")
    assert hygiene_marker(["title"]) in body
    assert "fix(MONO-1628): reuse tabs" in body
    assert "MONO-1628: reuse tabs" in body


def test_assign_comment_mentions_assignee_and_reviewer():
    body = build_hygiene_comment(["assign"], "MONO-1628: reuse tabs")
    assert "assignee" in body.lower()
    assert "reviewer" in body.lower()


# ---------- dedupe ----------

def test_matching_marker_in_existing_notes_suppresses_repeat():
    notes = ["## 🤖 AI Code Review\n\nLGTM.", build_hygiene_comment(["title"], "bad")]
    assert has_matching_hygiene_note(notes, ["title"]) is True


def test_different_issue_set_is_not_suppressed():
    notes = [build_hygiene_comment(["title"], "bad")]
    assert has_matching_hygiene_note(notes, ["title", "assign"]) is False


def test_no_hygiene_notes_means_nothing_to_suppress():
    assert has_matching_hygiene_note(["just a comment"], ["title"]) is False
