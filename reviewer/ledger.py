import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Iterable

from reviewer import config, gitlab_client
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

    def refund(self, n: int = 1) -> "Ledger":
        """Gives back a slot charged for a comment that was never posted."""
        return replace(self, posted=max(0, self.posted - n))

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
    """Reads a ledger out of a note body. No marker means a fresh MR.

    Fields are type-checked explicitly rather than coerced. A field that is
    merely absent still falls back to the Ledger default, but a field that is
    PRESENT with the wrong shape raises LedgerUnavailable instead of being
    silently coerced — a coerced wrong-type value (e.g. a `hunks` string
    iterated into single characters, or `bool("false") == True`) can make a
    corrupt marker look like a valid, empty ledger, which is exactly the
    "re-review the whole MR" flood this module exists to prevent.
    """
    match = MARKER_RE.search(body or "")
    if not match:
        return Ledger()
    try:
        payload = json.loads(match.group(1))
    except ValueError as exc:
        raise LedgerUnavailable(f"state marker is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LedgerUnavailable("state marker is not a JSON object")

    head = payload.get("head", "")
    if not isinstance(head, str):
        raise LedgerUnavailable(
            f"state marker field 'head' must be a string, got {type(head).__name__}",
        )

    posted = payload.get("posted", 0)
    if isinstance(posted, bool) or not isinstance(posted, int):
        raise LedgerUnavailable(
            f"state marker field 'posted' must be an int, got {type(posted).__name__}",
        )

    muted = payload.get("muted", False)
    if not isinstance(muted, bool):
        raise LedgerUnavailable(
            f"state marker field 'muted' must be a bool, got {type(muted).__name__}",
        )

    oversized = payload.get("oversized", False)
    if not isinstance(oversized, bool):
        raise LedgerUnavailable(
            f"state marker field 'oversized' must be a bool, got {type(oversized).__name__}",
        )

    hunks = payload.get("hunks", [])
    if not isinstance(hunks, list) or not all(isinstance(key, str) for key in hunks):
        raise LedgerUnavailable(
            "state marker field 'hunks' must be a list of strings",
        )

    return Ledger(
        head=head,
        posted=posted,
        muted=muted,
        oversized=oversized,
        hunks=tuple(hunks),
    )


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


class LedgerStore:
    """Owns the GitLab state note. `Ledger` itself stays a pure value.

    The note object is cached so a save is one API write, not a re-listing of
    every note on the merge request.
    """

    def __init__(self, mr, note, value: Ledger) -> None:
        self.mr = mr
        self._note = note
        self.ledger = value

    @classmethod
    def load(cls, mr) -> "LedgerStore":
        try:
            note = gitlab_client.find_note_with(mr, MARKER_PREFIX)
        except LedgerUnavailable:
            raise
        except Exception as exc:
            raise LedgerUnavailable(f"could not list MR notes: {exc}") from exc
        if note is None:
            return cls(mr, None, Ledger())
        return cls(mr, note, parse_marker(getattr(note, "body", None) or ""))

    def save(self) -> None:
        """Writes the current value. Failures propagate.

        A failed save means posted comments went unrecorded; continuing would
        post them again on the next push, so the run must stop instead.
        """
        body = render_note(self.ledger)
        if self._note is None:
            self._note = gitlab_client.create_note(self.mr, body)
        else:
            gitlab_client.update_note(self._note, body)
