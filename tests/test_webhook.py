import pytest

import review_server
from review_server import ReviewJob, app, queue_key, should_review
from reviewer import gitlab_client


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
    assert should_review("Merge Request Hook", _mr_hook("open")) == ReviewJob(3, 7, False, "review")


def test_reopen_action_is_reviewed():
    assert should_review("Merge Request Hook", _mr_hook("reopen")) == ReviewJob(3, 7, False, "review")


def test_update_with_new_commits_is_reviewed():
    assert should_review("Merge Request Hook", _mr_hook("update", oldrev="abc123")) == ReviewJob(3, 7, False, "review")


def test_update_without_oldrev_is_ignored():
    # Regression for B7: title/description/label edits carry no oldrev.
    assert should_review("Merge Request Hook", _mr_hook("update")) is None


def test_merge_action_is_ignored():
    assert should_review("Merge Request Hook", _mr_hook("merge")) is None


@pytest.fixture(autouse=True)
def _known_bot(monkeypatch):
    """should_review now needs to know which account is Sidorovich's."""
    monkeypatch.setattr(gitlab_client, "bot_username", lambda: "sidorovich-bot")


def _note_hook(note, author="human"):
    return {
        "project": {"id": 3},
        "merge_request": {"iid": 7},
        "user": {"username": author},
        "object_attributes": {"noteable_type": "MergeRequest", "note": note},
    }


def test_review_comment_triggers_review():
    assert should_review("Note Hook", _note_hook("please /review this")) == ReviewJob(3, 7, True, "review")


def test_sidorovichs_own_review_mention_does_not_retrigger_him():
    """The budget note says "кинь `/review`" and GitLab webhooks it back to us.

    Obeying it would run unmute_and_reset, clearing the mute and zeroing the
    comment counter — the hard stop would re-arm itself and never bind.
    """
    note = "**Ліміт вичерпано.** Далі мовчу. Кинь `/review` — лічильник обнулиться."
    assert should_review("Note Hook", _note_hook(note, author="sidorovich-bot")) is None


def test_review_is_ignored_when_authorship_is_unknown(monkeypatch):
    """Fail closed: an unattributable /review might be the bot's own."""
    monkeypatch.setattr(gitlab_client, "bot_username", lambda: "")
    assert should_review("Note Hook", _note_hook("/review")) is None


def test_kill_switch_ignores_every_event(monkeypatch):
    monkeypatch.setattr("reviewer.config.SIDOROVICH_ENABLED", False)
    assert should_review("Merge Request Hook", _mr_hook("open")) is None
    assert should_review("Note Hook", _note_hook("/review")) is None
    assert should_review("Note Hook", _note_hook("/sidorovich stop")) is None


def test_stop_command_produces_a_mute_job():
    job = should_review("Note Hook", _note_hook("/sidorovich stop"))
    assert job == ReviewJob(3, 7, False, "mute")


def test_stop_command_wins_over_review():
    job = should_review("Note Hook", _note_hook("/review then /sidorovich stop"))
    assert job.command == "mute"


def test_mute_and_review_do_not_coalesce():
    """A queued mute must not be replaced by a review for the same MR."""
    mute = should_review("Note Hook", _note_hook("/sidorovich stop"))
    review = should_review("Note Hook", _note_hook("/review"))
    assert queue_key(mute) != queue_key(review)


def test_sidorovich_cannot_stop_himself():
    """The bot must never be able to trigger its own kill switch.

    Same hole as the /review self-trigger (ad5a986): if a bot-authored note
    could carry /sidorovich stop, any text Sidorovich itself posts containing
    that phrase — including a future note that merely quotes the command in
    documentation or an error message — would silence the bot with no human
    involved at all. The author filter must cover this command too.
    """
    job = should_review(
        "Note Hook", _note_hook("/sidorovich stop", author="sidorovich-bot"),
    )
    assert job is None


def test_stop_command_is_ignored_when_authorship_is_unknown(monkeypatch):
    """Fail closed: an unattributable /sidorovich stop might be the bot's own."""
    monkeypatch.setattr(gitlab_client, "bot_username", lambda: "")
    assert should_review("Note Hook", _note_hook("/sidorovich stop")) is None


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
    assert submitted[0][1] == ReviewJob(3, 7, False, "review")


def test_webhook_returns_503_when_queue_full(client, monkeypatch):
    monkeypatch.setattr(review_server.review_queue, "submit", lambda key, job: False)
    resp = client.post("/webhook", json=_mr_hook("open"),
                       headers={"X-Gitlab-Token": "s3cret", "X-Gitlab-Event": "Merge Request Hook"})
    assert resp.status_code == 503


def test_health_endpoint(client):
    assert client.get("/health").status_code == 200
