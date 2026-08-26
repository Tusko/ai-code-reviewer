import logging
import re
from typing import Sequence

from reviewer import config, openrouter_client, prompt as prompt_mod
from reviewer.chat_types import FINDING_TAGS, ChatResult

VOICE_DEADLINE_S = 20
# Consecutive failed voice calls before the rest of the MR stays dry. Free-tier
# OpenRouter models rate-limit mid-review; retrying every file just makes half
# the comments Sidorovich and half of them dry.
VOICE_FAILURE_LIMIT = 2
FENCE_BODY_RE = re.compile(r"```(?:\w*)\n?(.*?)```", re.DOTALL)


class VoiceState:
    """Per-MR circuit breaker for the Sidorovich voice rewrite."""

    def __init__(self) -> None:
        self.failures = 0

    @property
    def enabled(self) -> bool:
        return self.failures < VOICE_FAILURE_LIMIT

    def record_failure(self) -> None:
        self.failures += 1
        if not self.enabled:
            logging.warning(
                "Sidorovich voice disabled for the rest of this MR after %s "
                "consecutive failures", self.failures,
            )

    def record_success(self) -> None:
        self.failures = 0


def _fence_bodies(text: str) -> list[str]:
    return FENCE_BODY_RE.findall(text)


def preserves_findings(original: str, flavored: str) -> bool:
    """True when the rewrite kept every finding tag and every Fix code block."""
    for tag in FINDING_TAGS:
        if original.count(tag) != flavored.count(tag):
            return False
    original_fences = _fence_bodies(original)
    flavored_fences = _fence_bodies(flavored)
    if original_fences:
        return flavored_fences == original_fences
    # No fences to preserve, but the rewrite must not invent a *Fix:* block —
    # code Sidorovich made up is worse than no code at all.
    return not flavored_fences


def _voice_chat(user: str, history: Sequence[dict] = ()) -> ChatResult:
    return openrouter_client.chat(
        prompt_mod.SIDOROVICH_REVIEW_VOICE_PROMPT,
        user,
        deadline_s=VOICE_DEADLINE_S,
        # A rewrite must stay faithful to the finding. High temperature here is
        # what makes Sidorovich invent bugs that are not in the diff.
        temperature=0.5,
        max_tokens=config.OPENROUTER_VOICE_MAX_TOKENS,
        history=history,
    )


def flavor_review(text: str, voice: "VoiceState | None" = None) -> str:
    """Rewrite a dry finding as Sidorovich. Voice is extra: never block or replace."""
    if not config.SNARK or not config.OPENROUTER_API_KEY:
        return text
    if voice is not None and not voice.enabled:
        return text

    result = prefer_ukrainian(
        _voice_chat(text),
        lambda bad: _voice_chat(
            prompt_mod.SIDOROVICH_UKRAINIAN_RETRY,
            history=(
                {"role": "user", "content": text},
                {"role": "assistant", "content": bad},
            ),
        ),
    )
    flavored = (result.text or "").strip()

    def keep_dry(why: str) -> str:
        logging.warning("Sidorovich voice rewrite skipped (%s); posting dry review", why)
        if voice is not None:
            voice.record_failure()
        return text

    if result.failed or result.done_reason == "length" or not flavored:
        return keep_dry(result.done_reason)
    if not preserves_findings(text, flavored):
        return keep_dry("rewrite changed findings")
    if prompt_mod.unusable_language(flavored):
        return keep_dry("still Russian after retry")

    if voice is not None:
        voice.record_success()
    return flavored


def prefer_ukrainian(result: ChatResult, retry) -> ChatResult:
    """One retry when Sidorovich slipped into Russian.

    `retry` is called with the offending reply so it can be replayed to the
    model as an assistant turn; scolding a model that cannot see what it wrote
    mostly reproduces the same mistake.
    """
    if result.failed or not prompt_mod.unusable_language(result.text):
        return result
    logging.warning("Sidorovich broke the language rule; retrying")
    retried = retry(result.text)
    if retried.failed or not (retried.text or "").strip():
        return result
    return retried
