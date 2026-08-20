import logging
import time
from typing import Sequence

import requests

from reviewer import config
from reviewer.ollama_client import ChatResult, clean_response

# 429 on a free-tier model and 5xx from an upstream provider are both transient:
# OpenRouter routes the next attempt to a different provider more often than not.
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3
BASE_BACKOFF_S = 2.0
# Never wait longer than this on a Retry-After header; the review has a deadline.
MAX_BACKOFF_S = 15.0


def chat(
    system: str,
    user: str,
    deadline_s: int,
    *,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    history: Sequence[dict] = (),
) -> ChatResult:
    """One-shot OpenRouter chat completion for Sidorovich voice.

    `history` is inserted between the system prompt and `user`, so a retry can
    show the model the reply it is being asked to correct.

    Transient failures (429, 5xx) are retried with backoff for as long as
    `deadline_s` leaves room; `deadline_s` bounds the whole call, not one attempt.
    """
    started = time.monotonic()
    if not config.OPENROUTER_API_KEY:
        return ChatResult(
            text="OPENROUTER_API_KEY is not set",
            done_reason="error",
            prompt_eval_count=0,
            eval_count=0,
            elapsed_s=0.0,
        )

    body = {
        "model": config.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens or config.OPENROUTER_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": config.GITLAB_URL or "http://localhost",
        "X-Title": "Sidorovich",
    }

    result = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        remaining = deadline_s - (time.monotonic() - started)
        if remaining <= 1:
            break
        result, retry_after = _attempt(body, headers, int(remaining), started)
        if retry_after is None:
            return result

        backoff = _backoff(attempt, retry_after)
        spent = time.monotonic() - started
        # Only sleep if a further attempt can still finish inside the deadline.
        if attempt == MAX_ATTEMPTS or spent + backoff + 2 >= deadline_s:
            break
        logging.warning(
            "OpenRouter attempt %s/%s failed (%s); retrying in %.1fs",
            attempt, MAX_ATTEMPTS, result.done_reason, backoff,
        )
        time.sleep(backoff)

    return result


def _attempt(
    body: dict, headers: dict, read_timeout: int, started: float,
) -> tuple[ChatResult, float | None]:
    """One HTTP call. Second element is a retry hint (seconds, or None to stop)."""
    try:
        response = requests.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            json=body,
            headers=headers,
            timeout=(10, read_timeout),
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": {"message": response.text[:500]}}

        if not response.ok:
            message = _error_message(payload) or response.text[:500]
            # Free-tier models rate-limit hard. Tag it distinctly so callers can
            # stop hammering OpenRouter for the rest of the merge request.
            reason = "ratelimit" if response.status_code == 429 else "error"
            logging.error("OpenRouter returned %s: %s", response.status_code, message)
            failed = ChatResult(
                text=f"Error communicating with OpenRouter: {message}",
                done_reason=reason,
                prompt_eval_count=0,
                eval_count=0,
                elapsed_s=time.monotonic() - started,
            )
            if response.status_code in RETRYABLE_STATUSES:
                return failed, _retry_after(response)
            return failed, None

        err = _error_message(payload)
        if err:
            logging.error("OpenRouter error payload: %s", err)
            return ChatResult(
                text=f"Error communicating with OpenRouter: {err}",
                done_reason="error",
                prompt_eval_count=0,
                eval_count=0,
                elapsed_s=time.monotonic() - started,
            ), None

        choice = (payload.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        finish = choice.get("finish_reason") or "stop"
        done_reason = "length" if finish == "length" else "stop"
        usage = payload.get("usage") or {}
        prompt_eval = usage.get("prompt_tokens") or 0
        eval_count = usage.get("completion_tokens") or 0

        if not text:
            logging.warning("OpenRouter returned empty content (finish=%s)", finish)
            return ChatResult(
                text="",
                done_reason="incomplete" if finish != "length" else "length",
                prompt_eval_count=prompt_eval,
                eval_count=eval_count,
                elapsed_s=time.monotonic() - started,
            ), None

        elapsed = time.monotonic() - started
        logging.info(
            "OpenRouter done in %.1fs (model=%s reason=%s prompt=%s eval=%s)",
            elapsed, config.OPENROUTER_MODEL, done_reason, prompt_eval, eval_count,
        )
        return ChatResult(
            text=clean_response(text),
            done_reason=done_reason,
            prompt_eval_count=prompt_eval,
            eval_count=eval_count,
            elapsed_s=elapsed,
        ), None

    except requests.Timeout as exc:
        logging.error("OpenRouter timed out: %s", exc)
        return ChatResult(
            text=f"Error communicating with OpenRouter: {exc}",
            done_reason="error",
            prompt_eval_count=0,
            eval_count=0,
            elapsed_s=time.monotonic() - started,
        ), 0.0
    except Exception as exc:
        logging.error("Error communicating with OpenRouter: %s", exc)
        return ChatResult(
            text=f"Error communicating with OpenRouter: {exc}",
            done_reason="error",
            prompt_eval_count=0,
            eval_count=0,
            elapsed_s=time.monotonic() - started,
        ), None


def _retry_after(response) -> float:
    """Seconds the server asked us to wait, or 0 to use plain backoff."""
    raw = (response.headers.get("Retry-After") or "").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _backoff(attempt: int, retry_after: float) -> float:
    return min(MAX_BACKOFF_S, max(retry_after, BASE_BACKOFF_S * attempt))


def _error_message(payload: dict) -> str:
    err = payload.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err)
    if err:
        return str(err)
    return ""
