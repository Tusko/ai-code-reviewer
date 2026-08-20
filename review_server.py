import logging

from flask import Flask, jsonify, request

from reviewer import config
from reviewer.pipeline import review_merge_request
from reviewer.queue import start_worker

config.configure_logging()

app = Flask(__name__)

logging.info(
    "Ollama config: host=%s model=%s num_ctx=%s num_predict=%s num_batch=%s "
    "include_context=%s per_file_timeout=%ss mr_timeout=%ss max_files=%s",
    config.OLLAMA_HOST, config.OLLAMA_MODEL, config.OLLAMA_NUM_CTX,
    config.OLLAMA_NUM_PREDICT, config.OLLAMA_NUM_BATCH, config.INCLUDE_FILE_CONTEXT,
    config.PER_FILE_TIMEOUT_S, config.MR_TIMEOUT_S, config.MAX_FILES,
)
if config.OPENROUTER_API_KEY:
    _sidorovich_llm = f"openrouter:{config.OPENROUTER_MODEL}"
elif config.SIDOROVICH_OLLAMA_FALLBACK:
    _sidorovich_llm = f"ollama:{config.OLLAMA_MODEL} (fallback)"
else:
    _sidorovich_llm = "disabled (no OPENROUTER_API_KEY)"
logging.info("Sidorovich LLM: %s", _sidorovich_llm)


def _handle(job: tuple) -> None:
    project_id, mr_iid, force = job
    review_merge_request(project_id, mr_iid, force=force)


review_queue = start_worker(_handle)


def should_review(event_type: str, data: dict) -> tuple[int, int, bool] | None:
    """Returns (project_id, mr_iid, force) if this event warrants a review, else None.

    force is True for a manual '/review' comment, which must re-run even
    though the diff itself is by definition unchanged. It is False for the
    automatic Merge Request Hook paths, which defer to the dedupe cache.
    """
    attrs = data.get("object_attributes", {})

    if event_type == "Note Hook":
        if attrs.get("noteable_type") != "MergeRequest":
            return None
        if "/review" not in (attrs.get("note") or "").lower():
            return None
        return data["project"]["id"], data["merge_request"]["iid"], True

    if event_type == "Merge Request Hook":
        action = attrs.get("action")
        if action in ("open", "reopen"):
            return data["project"]["id"], attrs["iid"], False
        # 'update' fires on title, description and label edits too. GitLab sets
        # oldrev only when new commits arrived, so it is our new-commits signal.
        if action == "update" and attrs.get("oldrev"):
            return data["project"]["id"], attrs["iid"], False
        return None

    return None


@app.route("/webhook", methods=["POST"])
def webhook():
    token = request.headers.get("X-Gitlab-Token")
    if config.WEBHOOK_SECRET and token != config.WEBHOOK_SECRET:
        return jsonify({"error": "Invalid token"}), 403

    target = should_review(request.headers.get("X-Gitlab-Event"), request.json or {})
    if not target:
        return jsonify({"message": "Ignored event"}), 200

    project_id, mr_iid, force = target
    if review_queue.submit(f"{project_id}:{mr_iid}", (project_id, mr_iid, force)):
        return jsonify({"message": "Review queued", "depth": review_queue.size()}), 202
    return jsonify({"error": "Review queue full"}), 503


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "queue_depth": review_queue.size()}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
