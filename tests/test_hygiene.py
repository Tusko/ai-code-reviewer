import pytest

from reviewer import config, hygiene, pipeline
from reviewer.ledger import Ledger, LedgerUnavailable, parse_marker, to_marker


# ---------- title format ----------

@pytest.mark.parametrize("title", [
    "MONO-1628: reuse browser tabs for performance",
    "ABC-1: bump deps",
    "Draft: MONO-1628: reuse browser tabs",
    "WIP: MONO-1628: reuse browser tabs",
])
def test_accepts_ticket_prefixed_titles(title):
    assert hygiene.is_valid_title(title) is True


@pytest.mark.parametrize("title", [
    "fix(MONO-1628): reuse browser tabs for performance",
    "MONO-1628 reuse browser tabs",
    "reuse browser tabs",
    "mono-1628: reuse browser tabs",
    "MONO-1628:",
    "",
])
def test_rejects_titles_without_a_bare_ticket_prefix(title):
    assert hygiene.is_valid_title(title) is False


def test_title_pattern_is_configurable(monkeypatch):
    monkeypatch.setattr(hygiene, "TITLE_RE", hygiene.re.compile(r"^ticket \d+$"))
    assert hygiene.is_valid_title("ticket 7") is True
    assert hygiene.is_valid_title("MONO-1628: x") is False


# ---------- issue detection ----------

def test_clean_mr_has_no_issues():
    assert hygiene.check("MONO-1628: reuse tabs", ["dev"], []) == []


def test_bad_title_is_reported():
    assert hygiene.check("fix(MONO-1628): reuse tabs", ["dev"], []) == ["title"]


def test_missing_assignee_and_reviewer_is_reported():
    assert hygiene.check("MONO-1628: reuse tabs", [], []) == ["assign"]


def test_a_reviewer_alone_satisfies_assignment():
    assert hygiene.check("MONO-1628: reuse tabs", [], ["reviewer"]) == []


def test_both_issues_are_reported_in_a_stable_order():
    assert hygiene.check("bad title", [], []) == ["title", "assign"]


def test_issue_key_order_does_not_depend_on_input_order():
    assert hygiene.issue_key(["assign", "title"]) == hygiene.issue_key(["title", "assign"])
    assert hygiene.issue_key(["title", "assign"]) == "title,assign"
    assert hygiene.issue_key([]) == ""


# ---------- rendering ----------

def test_comment_shows_the_offending_and_the_expected_title():
    body = hygiene.render(["title"], "fix(MONO-1628): reuse tabs")
    assert "fix(MONO-1628): reuse tabs" in body
    assert "MONO-1628: reuse tabs" in body


def test_comment_about_assignment_names_both_roles():
    body = hygiene.render(["assign"], "MONO-1628: reuse tabs")
    assert "assignee" in body.lower()
    assert "reviewer" in body.lower()


def test_comment_carries_snark_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "SNARK", True)
    monkeypatch.setattr(hygiene, "snark", lambda: "Сука, руль вирвало")
    assert "Сука, руль вирвало" in hygiene.render(["title"], "bad")


def test_comment_stays_dry_when_snark_is_off(monkeypatch):
    monkeypatch.setattr(config, "SNARK", False)
    monkeypatch.setattr(hygiene, "snark", lambda: "Сука, руль вирвало")
    assert "Сука, руль вирвало" not in hygiene.render(["title"], "bad")


# ---------- ledger field ----------

def test_ledger_remembers_the_reported_issue_set():
    assert Ledger().report_hygiene("title,assign").hygiene == "title,assign"


def test_hygiene_survives_a_marker_round_trip():
    value = Ledger().report_hygiene("title")
    assert parse_marker(to_marker(value)).hygiene == "title"


def test_a_marker_without_hygiene_reads_as_never_reported():
    assert parse_marker(to_marker(Ledger())).hygiene == ""


def test_a_wrongly_typed_hygiene_field_is_refused():
    with pytest.raises(LedgerUnavailable):
        parse_marker('<!-- sidorovich-state:v1 {"hygiene":["title"]} -->')


# ---------- pipeline behaviour ----------

class FakeMR:
    source_branch = "feature/x"

    def __init__(self, title="MONO-1: ok", assignees=(), reviewers=()):
        self.title = title
        self.assignees = list(assignees)
        self.reviewers = list(reviewers)


class FakeStore:
    def __init__(self, ledger=None):
        self.mr = FakeMR()
        self.ledger = ledger or Ledger()
        self.saves = 0

    def save(self):
        self.saves += 1


@pytest.fixture
def posted(monkeypatch):
    bodies = []
    monkeypatch.setattr(
        pipeline.gitlab_client, "post_note", lambda mr, body: bodies.append(body),
    )
    return bodies


def test_a_dirty_mr_is_nagged_once_and_charged(posted):
    store = FakeStore()
    pipeline.nag_hygiene(1, FakeMR(title="fix(MONO-1): x"), store)
    assert len(posted) == 1
    assert store.ledger.posted == 1
    assert store.ledger.hygiene == "title,assign"


def test_the_same_issues_are_not_nagged_twice(posted):
    store = FakeStore(Ledger().report_hygiene("title,assign"))
    pipeline.nag_hygiene(1, FakeMR(title="fix(MONO-1): x"), store)
    assert posted == []
    assert store.ledger.posted == 0


def test_a_changed_issue_set_is_nagged_again(posted):
    store = FakeStore(Ledger().report_hygiene("title,assign"))
    pipeline.nag_hygiene(1, FakeMR(title="fix(MONO-1): x", reviewers=["dev"]), store)
    assert len(posted) == 1
    assert store.ledger.hygiene == "title"


def test_a_clean_mr_is_silent_and_forgets_the_old_issues(posted):
    store = FakeStore(Ledger().report_hygiene("title"))
    pipeline.nag_hygiene(1, FakeMR(title="MONO-1: ok", assignees=["dev"]), store)
    assert posted == []
    assert store.ledger.hygiene == ""


def test_a_clean_mr_with_nothing_recorded_writes_nothing(posted):
    store = FakeStore()
    pipeline.nag_hygiene(1, FakeMR(title="MONO-1: ok", assignees=["dev"]), store)
    assert posted == []
    assert store.saves == 0


def test_an_exhausted_budget_suppresses_the_nag(posted, monkeypatch):
    monkeypatch.setattr(config, "MR_COMMENT_BUDGET", 2)
    store = FakeStore(Ledger().spend(1))
    pipeline.nag_hygiene(1, FakeMR(title="fix(MONO-1): x"), store)
    assert posted == []
    assert store.ledger.hygiene == ""


def test_a_failed_post_is_retried_on_the_next_push(posted, monkeypatch):
    def boom(mr, body):
        raise RuntimeError("gitlab is down")

    monkeypatch.setattr(pipeline.gitlab_client, "post_note", boom)
    store = FakeStore()
    pipeline.nag_hygiene(1, FakeMR(title="fix(MONO-1): x"), store)
    # The slot stays charged (the note may have landed), but the issue set is
    # not recorded, so the next push says it again rather than going silent.
    assert store.ledger.posted == 1
    assert store.ledger.hygiene == ""
