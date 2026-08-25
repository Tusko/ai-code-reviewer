import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Iterable

from reviewer import config
from reviewer.diff_parser import Hunk

MARKER_PREFIX = "sidorovich-state:v1"
MARKER_RE = re.compile(
    r"<!--\s*" + re.escape(MARKER_PREFIX) + r"\s*(\{.*?\})\s*-->", re.DOTALL,
)


class LedgerUnavailable(Exception):
    """The MR's review state could not be read. The run must not proceed.

    Treating an unreadable ledger as an empty one would re-review the whole MR,
    which is the exact flood this module exists to prevent.
    """


def hunk_key(path: str, hunk: Hunk) -> str:
    """Content hash of one hunk, stable across rebases.

    `new_start` and `old_start` are deliberately excluded: line numbers shift
    whenever an unrelated earlier hunk changes, but the content does not.
    """
    digest = hashlib.sha256()
    digest.update(path.encode())
    digest.update(b"\x00")
    for line in hunk.lines:
        digest.update(line.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:12]


@dataclass(frozen=True)
class Ledger:
    head: str = ""
    posted: int = 0
    muted: bool = False
    oversized: bool = False
    hunks: tuple[str, ...] = ()

    def remaining(self) -> int:
        return max(0, config.MR_COMMENT_BUDGET - self.posted)

    def record(self, keys: Iterable[str]) -> "Ledger":
        merged = list(self.hunks)
        known = set(self.hunks)
        for key in keys:
            if key not in known:
                merged.append(key)
                known.add(key)
        if len(merged) > config.LEDGER_MAX_HUNKS:
            merged = merged[len(merged) - config.LEDGER_MAX_HUNKS:]
        return replace(self, hunks=tuple(merged))

    def spend(self, n: int = 1) -> "Ledger":
        return replace(self, posted=self.posted + n)

    def mute(self) -> "Ledger":
        return replace(self, muted=True)

    def unmute_and_reset(self) -> "Ledger":
        return replace(self, muted=False, posted=0)

    def mark_oversized(self) -> "Ledger":
        return replace(self, oversized=True)

    def at_head(self, head: str) -> "Ledger":
        return replace(self, head=head)


def to_marker(value: Ledger) -> str:
    payload = {
        "head": value.head,
        "posted": value.posted,
        "muted": value.muted,
        "oversized": value.oversized,
        "hunks": list(value.hunks),
    }
    return f"<!-- {MARKER_PREFIX} {json.dumps(payload, separators=(',', ':'))} -->"


def parse_marker(body: str) -> Ledger:
    """Reads a ledger out of a note body. No marker means a fresh MR."""
    match = MARKER_RE.search(body or "")
    if not match:
        return Ledger()
    try:
        payload = json.loads(match.group(1))
    except ValueError as exc:
        raise LedgerUnavailable(f"state marker is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LedgerUnavailable("state marker is not a JSON object")
    try:
        return Ledger(
            head=str(payload.get("head") or ""),
            posted=int(payload.get("posted") or 0),
            muted=bool(payload.get("muted")),
            oversized=bool(payload.get("oversized")),
            hunks=tuple(str(key) for key in payload.get("hunks") or ()),
        )
    except (TypeError, ValueError) as exc:
        raise LedgerUnavailable(f"state marker has bad field types: {exc}") from exc


def render_note(value: Ledger) -> str:
    head = value.head or "—"
    status = "заглушений" if value.muted else "активний"
    return (
        "🔒 **Сідорович — стан рев'ю**\n\n"
        f"Коментарів: {value.posted}/{config.MR_COMMENT_BUDGET} · "
        f"останній head: `{head}` · {status}\n\n"
        "_Цю нотатку я редагую, а не пишу заново. Не чіпай._\n\n"
        + to_marker(value)
    )
