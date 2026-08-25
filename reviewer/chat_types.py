import re
from dataclasses import dataclass

HARMONY_TOKEN_RE = re.compile(r"<\|[^>]*\|?>")
FENCE_SPLIT_RE = re.compile(r"(```[\s\S]*?```)")
FINDING_TAGS = ("[BLOCKER]", "[SUGGESTION]", "[NIT]")
LGTM_TEXT = "LGTM. The changes are clean and follow best practices."
# `LGTM`, `LGTM.`, `[LGTM]`, `**LGTM.**` — any bare clean verdict.
BARE_LGTM_RE = re.compile(r"^[*\[\s]*LGTM[\]\s*.!]*$", re.IGNORECASE)
# Single-token replies the model emits when it gives up mid-generation.
DEGENERATE_REPLIES = frozenset({"the", "ok", "okay", "none", "n/a", "-", "..."})


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
        return self.done_reason in ("error", "ratelimit")


def clean_response(text: str) -> str:
    text = HARMONY_TOKEN_RE.sub(" ", text).strip()
    pieces = []
    for i, part in enumerate(FENCE_SPLIT_RE.split(text)):
        if i % 2 == 0:
            pieces.append(re.sub(r"[ \t]{2,}", " ", part))
        else:
            pieces.append(part)
    text = "".join(pieces)
    has_finding = any(tag in text for tag in FINDING_TAGS)
    # The model bailed and emitted a stray token. Returning anything readable
    # here would be posted verbatim as a review finding, so return nothing and
    # let the caller report an error.
    if not has_finding and text.strip().lower() in DEGENERATE_REPLIES:
        return ""
    if not has_finding and BARE_LGTM_RE.match(text.strip()):
        return LGTM_TEXT
    if "[LGTM]" in text and not has_finding:
        return LGTM_TEXT
    return text
