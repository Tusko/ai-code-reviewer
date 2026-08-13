import pytest

import review_server
from review_server import app, should_review


@pytest.fixture
def client(monkeypatch):
    # review_server reads config.WEBHOOK_SECRET at request time, so patching
    # the config module attribute is sufficient.
    monkeypatch.setattr("reviewer.config.WEBHOOK_SECRET", "s3cret")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _mr_hook(action, oldrev=None):
    attrs = {"action": action, "iid": 7}
    if oldrev:
        attrs["oldrev"] = oldrev
    return {"project": {"id": 3}, "object_attributes": attrs}


def test_open_action_is_reviewed():
    assert should_review("Merge Request Hook", _mr_hook("open")) == (3, 7, False)


def test_reopen_action_is_reviewed():
    assert should_review("Merge Request Hook", _mr_hook("reopen")) == (3, 7, False)


def test_update_with_new_commits_is_reviewed():
    assert should_review("Merge Request Hook", _mr_hook("update", oldrev="abc123")) == (3, 7, False)


def test_update_without_oldrev_is_ignored():
    # Regression for B7: title/description/label edits carry no oldrev.
    assert should_review("Merge Request Hook", _mr_hook("update")) is None


def test_merge_action_is_ignored():
    assert should_review("Merge Request Hook", _mr_hook("merge")) is None


def test_review_comment_triggers_review():
    data = {
        "project": {"id": 3},
        "merge_request": {"iid": 7},
        "object_attributes": {"noteable_type": "MergeRequest", "note": "please /review this"},
    }
    assert should_review("Note Hook", data) == (3, 7, True)


def test_unrelated_comment_is_ignored():
    data = {
        "project": {"id": 3},
        "merge_request": {"iid": 7},
        "object_attributes": {"noteable_type": "MergeRequest", "note": "nice work"},
    }
    assert should_review("Note Hook", data) is None


def test_note_on_issue_is_ignored():
    data = {
        "project": {"id": 3},
        "object_attributes": {"noteable_type": "Issue", "note": "/review"},
    }
    assert should_review("Note Hook", data) is None


def test_webhook_rejects_bad_token(client):
    resp = client.post("/webhook", json=_mr_hook("open"),
                       headers={"X-Gitlab-Token": "wrong", "X-Gitlab-Event": "Merge Request Hook"})
    assert resp.status_code == 403


def test_webhook_enqueues_and_returns_202(client, monkeypatch):
    submitted = []
    monkeypatch.setattr(review_server.review_queue, "submit",
                        lambda key, job: submitted.append((key, job)) or True)
    resp = client.post("/webhook", json=_mr_hook("open"),
                       headers={"X-Gitlab-Token": "s3cret", "X-Gitlab-Event": "Merge Request Hook"})
    assert resp.status_code == 202
    assert submitted[0][1] == (3, 7, False)


def test_webhook_returns_503_when_queue_full(client, monkeypatch):
    monkeypatch.setattr(review_server.review_queue, "submit", lambda key, job: False)
    resp = client.post("/webhook", json=_mr_hook("open"),
                       headers={"X-Gitlab-Token": "s3cret", "X-Gitlab-Event": "Merge Request Hook"})
    assert resp.status_code == 503


def test_health_endpoint(client):
    assert client.get("/health").status_code == 200
