import logging
import re
from dataclasses import dataclass, replace

from flask import Flask, jsonify, request

from reviewer import config, gitlab_client
from reviewer.pipeline import mute_merge_request, review_merge_request
from reviewer.queue import start_worker

config.configure_logging()

app = Flask(__name__)

_review_llm = f"openrouter:{config.OPENROUTER_REVIEW_MODEL}"
if config.OPENROUTER_REVIEW_FALLBACK_MODELS:
    _review_llm += " -> " + " -> ".join(config.OPENROUTER_REVIEW_FALLBACK_MODELS)
if not config.OPENROUTER_API_KEY:
    _review_llm += "  [NO API KEY — every file will error]"
logging.info(
    "Review LLM: %s ctx=%s max_out=%s per_file_timeout=%ss mr_timeout=%ss "
    "max_files=%s include_context=%s",
    _review_llm, config.REVIEW_CONTEXT_TOKENS, config.REVIEW_MAX_OUTPUT_TOKENS,
    config.PER_FILE_TIMEOUT_S, config.MR_TIMEOUT_S, config.MAX_FILES,
    config.INCLUDE_FILE_CONTEXT,
)
if config.SIDOROVICH_OLLAMA_FALLBACK:
    logging.info(
        "Ollama voice fallback: host=%s model=%s num_ctx=%s num_predict=%s "
        "num_batch=%s", config.OLLAMA_HOST, config.OLLAMA_MODEL,
        config.OLLAMA_NUM_CTX, config.OLLAMA_NUM_PREDICT, config.OLLAMA_NUM_BATCH,
    )
if config.OPENROUTER_API_KEY:
    _sidorovich_llm = f"openrouter:{config.OPENROUTER_MODEL}"
    if config.OPENROUTER_FALLBACK_MODELS:
        _sidorovich_llm += " -> " + " -> ".join(config.OPENROUTER_FALLBACK_MODELS)
elif config.SIDOROVICH_OLLAMA_FALLBACK:
    _sidorovich_llm = f"ollama:{config.OLLAMA_MODEL} (fallback)"
else:
    _sidorovich_llm = "disabled (no OPENROUTER_API_KEY)"
logging.info("Sidorovich LLM: %s", _sidorovich_llm)
logging.info(
    "Guardrails: enabled=%s max_mr_files=%s comment_budget=%s ledger_max_hunks=%s",
    config.SIDOROVICH_ENABLED, config.MAX_MR_FILES,
    config.MR_COMMENT_BUDGET, config.LEDGER_MAX_HUNKS,
)


# A command has to be typed, not merely mentioned. Rejecting a preceding word
# character or slash keeps a path like app/review/service.py from resetting the
# budget; rejecting a preceding backtick keeps a documentation mention from
# doing it -- including Sidorovich's own budget note, quote-replied by a human,
# which the author filter cannot catch because a human really did write it.
REVIEW_RE = re.compile(r"(?<![\w/`])/review\b", re.IGNORECASE)
# Trailing \b so "/sidorovich stopwatch" is not a kill switch.
STOP_RE = re.compile(r"(?<![\w/`])/sidorovich\s+stop\b", re.IGNORECASE)


def strip_quotes(note: str) -> str:
    """Drops blockquoted lines: quoting a command is not issuing one.

    Without this, replying to a comment that says /review re-runs the review,
    and quoting Sidorovich's budget note lifts the mute he just set.
    """
    return "\n".join(
        line for line in note.splitlines() if not line.lstrip().startswith(">")
    )


@dataclass(frozen=True)
class ReviewJob:
    project_id: int
    mr_iid: int
    force: bool
    command: str      # "review" | "mute"


def queue_key(job: ReviewJob) -> str:
    """Coalescing key. The command is part of it so a queued mute is never
    replaced by a review for the same merge request."""
    return f"{job.project_id}:{job.mr_iid}:{job.command}"


def _handle(job: ReviewJob) -> None:
    if job.command == "mute":
        mute_merge_request(job.project_id, job.mr_iid)
        return
    review_merge_request(job.project_id, job.mr_iid, force=job.force)


review_queue = start_worker(_handle)


def should_review(event_type: str, data: dict) -> ReviewJob | None:
    """Returns the job this event warrants, or None.

    force is True for a manual '/review' comment, which must re-run even though
    the diff itself is by definition unchanged. It is False for the automatic
    Merge Request Hook paths, which defer to the ledger.

    The kill switch (config.SIDOROVICH_ENABLED) is checked first, before any
    GitLab call: it must work even when GitLab itself is unreachable.
    """
    if not config.SIDOROVICH_ENABLED:
        # Logged, not silent: "the bot stopped commenting" must be answerable
        # from the container log without anyone guessing at the config.
        logging.info("SIDOROVICH_ENABLED is false; ignoring %s", event_type)
        return None

    attrs = data.get("object_attributes", {})

    if event_type == "Note Hook":
        if attrs.get("noteable_type") != "MergeRequest":
            return None
        note = strip_quotes(attrs.get("note") or "")
        if not (STOP_RE.search(note) or REVIEW_RE.search(note)):
            return None
        # The budget note ends with "кинь `/review`", and GitLab fires a Note
        # Hook for notes the bot creates through the API. Without this check
        # a bot-authored note containing either command re-triggers itself:
        # /review would clear its own mute via unmute_and_reset, and a
        # /sidorovich stop the bot could issue to itself would be a kill
        # switch an attacker (or a bug) could pull the bot's own strings on.
        # Fail closed when authorship cannot be established.
        bot = gitlab_client.bot_username()
        if not bot:
            logging.error("Ignoring note command: cannot tell whose comment this is")
            return None
        author = ((data.get("user") or {}).get("username") or "")
        if author.lower() == bot.lower():
            logging.info("Ignoring note command: it is Sidorovich's own comment")
            return None
        # Read after the author check: a malformed hook we were going to
        # ignore anyway must not raise KeyError and fail the whole webhook.
        project_id = data["project"]["id"]
        mr_iid = data["merge_request"]["iid"]
        # Checked before /review so a comment containing both silences the bot.
        if STOP_RE.search(note):
            return ReviewJob(project_id, mr_iid, False, "mute")
        return ReviewJob(project_id, mr_iid, True, "review")

    if event_type == "Merge Request Hook":
        action = attrs.get("action")
        if action in ("open", "reopen"):
            return ReviewJob(data["project"]["id"], attrs["iid"], False, "review")
        # 'update' fires on title, description and label edits too. GitLab sets
        # oldrev only when new commits arrived, so it is our new-commits signal.
        if action == "update" and attrs.get("oldrev"):
            return ReviewJob(data["project"]["id"], attrs["iid"], False, "review")
        return None

    return None


@app.route("/webhook", methods=["POST"])
def webhook():
    token = request.headers.get("X-Gitlab-Token")
    if config.WEBHOOK_SECRET and token != config.WEBHOOK_SECRET:
        return jsonify({"error": "Invalid token"}), 403

    job = should_review(request.headers.get("X-Gitlab-Event"), request.json or {})
    if not job:
        return jsonify({"message": "Ignored event"}), 200

    if job.command == "mute":
        if review_queue.drop(f"{job.project_id}:{job.mr_iid}:review"):
            logging.info("Dropped the queued review for MR !%s in favour of the "
                         "mute", job.mr_iid)
    else:
        queued = review_queue.peek(queue_key(job))
        if getattr(queued, "force", False) and not job.force:
            # Coalescing keeps the newer payload, so a push landing behind a
            # human's /review would otherwise silently downgrade it and the
            # ledger would then suppress everything.
            job = replace(job, force=True)

    if review_queue.submit(queue_key(job), job):
        return jsonify({"message": "Review queued", "depth": review_queue.size()}), 202
    return jsonify({"error": "Review queue full"}), 503


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "queue_depth": review_queue.size()}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
