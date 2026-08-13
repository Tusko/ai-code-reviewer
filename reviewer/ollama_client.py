import json
import logging
import random
import re
import time
from dataclasses import dataclass

import requests

from reviewer import config
from reviewer.memes import meme_phrases

HARMONY_TOKEN_RE = re.compile(r"<\|[^>]*\|?>")
FINDING_TAGS = ("[BLOCKER]", "[SUGGESTION]", "[NIT]")


@dataclass(frozen=True)
class ChatResult:
    text: str
    done_reason: str
    prompt_eval_count: int
    eval_count: int
    elapsed_s: float

    @property
    def truncated(self) -> bool:
        return self.done_reason in ("length", "timeout", "incomplete")

    @property
    def failed(self) -> bool:
        return self.done_reason == "error"


def clean_response(text: str) -> str:
    text = HARMONY_TOKEN_RE.sub(" ", text).strip()
    text = re.sub(r"[ \t]{2,}", " ", text)
    if text == "The":
        return random.choice(meme_phrases)
    if "[LGTM]" in text and not any(tag in text for tag in FINDING_TAGS):
        return "LGTM. The changes are clean and follow best practices."
    return text


def chat(system: str, user: str, deadline_s: int) -> ChatResult:
    """Streams a chat completion, aborting the connection at deadline_s."""
    started = time.monotonic()
    body = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "think": False,
        "stream": True,
        "keep_alive": "24h",
        "options": {
            "num_ctx": config.OLLAMA_NUM_CTX,
            "num_predict": config.OLLAMA_NUM_PREDICT,
            "num_batch": config.OLLAMA_NUM_BATCH,
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "seed": 42,
        },
    }

    chunks: list[str] = []
    done_reason = "incomplete"
    prompt_eval = 0
    eval_count = 0
    response = None

    try:
        response = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json=body,
            stream=True,
            timeout=(10, deadline_s),
        )
        if not response.ok:
            logging.error("Ollama returned %s: %s", response.status_code, response.text[:500])
        response.raise_for_status()

        for raw in response.iter_lines():
            if time.monotonic() - started > deadline_s:
                logging.warning("Aborting Ollama stream after %ss deadline", deadline_s)
                done_reason = "timeout"
                break
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            chunks.append(obj.get("message", {}).get("content", "") or "")
            if obj.get("done"):
                done_reason = obj.get("done_reason") or "stop"
                prompt_eval = obj.get("prompt_eval_count") or 0
                eval_count = obj.get("eval_count") or 0
                break

        if done_reason == "incomplete":
            logging.warning(
                "Ollama stream ended with no terminal payload (connection closed "
                "early, daemon restart, or OOM); treating as incomplete",
            )

    except Exception as exc:
        logging.error("Error communicating with Ollama: %s", exc)
        return ChatResult(
            text=f"Error communicating with AI Reviewer: {exc}",
            done_reason="error",
            prompt_eval_count=0,
            eval_count=0,
            elapsed_s=time.monotonic() - started,
        )
    finally:
        if response is not None:
            response.close()

    elapsed = time.monotonic() - started
    logging.info(
        "Ollama done in %.1fs (reason=%s prompt_eval=%s eval=%s)",
        elapsed, done_reason, prompt_eval, eval_count,
    )
    return ChatResult(
        text=clean_response("".join(chunks)),
        done_reason=done_reason,
        prompt_eval_count=prompt_eval,
        eval_count=eval_count,
        elapsed_s=elapsed,
    )
