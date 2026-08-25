import json
import logging
import time
from typing import Sequence

import requests

from reviewer import config
from reviewer.chat_types import ChatResult, clean_response, LGTM_TEXT


def chat(
    system: str,
    user: str,
    deadline_s: int,
    *,
    temperature: float = 0.1,
    seed: int | None = 42,
    history: Sequence[dict] = (),
) -> ChatResult:
    """Streams a chat completion, aborting the connection at deadline_s.

    `history` is inserted between the system prompt and `user`, so a retry can
    show the model the reply it is being asked to correct.
    """
    started = time.monotonic()
    options = {
        "num_ctx": config.OLLAMA_NUM_CTX,
        "num_predict": config.OLLAMA_NUM_PREDICT,
        "num_batch": config.OLLAMA_NUM_BATCH,
        "temperature": temperature,
        "top_p": 0.9,
        "repeat_penalty": 1.05,
    }
    if seed is not None:
        options["seed"] = seed
    body = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": user},
        ],
        "think": False,
        "stream": True,
        "keep_alive": "24h",
        "options": options,
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
