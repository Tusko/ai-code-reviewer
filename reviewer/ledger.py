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
    outage_reported: bool = False
    # Set once the ledger can no longer track this MR hunk by hunk. Evicting
    # the oldest keys looked harmless and was not: select_files sorts
    # deterministically, so the evicted keys are exactly the ones needed first
    # next run, and every /review replayed the whole merge request.
    saturated: bool = False
    hunks: tuple[str, ...] = ()
    # Hunks whose comment failed to post once. A second failure records them
    # like any settled hunk: retrying for ever means every /review replays the
    # whole merge request, and /review resets the budget that would cap it.
    retried: tuple[str, ...] = ()
    # The hygiene issues most recently reported on this MR, as an issue_key.
    # Kept here rather than scanned out of the comments so that fixing one of
    # two problems costs exactly one new nag, and fixing both costs none.
    hygiene: str = ""

    def remaining(self) -> int:
        return max(0, config.MR_COMMENT_BUDGET - self.posted)

    def record(self, keys: Iterable[str]) -> "Ledger":
        keys = list(keys)
        merged = list(self.hunks)
        known = set(self.hunks)
        for key in keys:
            if key not in known:
                merged.append(key)
                known.add(key)
        # A hunk that settled needs no retry slot, so stop paying to store it.
        settling = set(keys)
        retried = tuple(k for k in self.retried if k not in settling)
        if len(merged) > config.LEDGER_MAX_HUNKS:
            return replace(
                self, hunks=tuple(merged[: config.LEDGER_MAX_HUNKS]),
                retried=retried, saturated=True,
            )
        return replace(self, hunks=tuple(merged), retried=retried)

    def spend(self, n: int = 1) -> "Ledger":
        return replace(self, posted=self.posted + n)

    def refund(self, n: int = 1) -> "Ledger":
        """Gives back a slot charged for a comment that was never posted."""
        return replace(self, posted=max(0, self.posted - n))

    def mute(self) -> "Ledger":
        return replace(self, muted=True)

    def unmute_and_reset(self) -> "Ledger":
        return replace(self, muted=False, posted=0)

    def mark_retried(self, keys: Iterable[str]) -> "Ledger":
        merged = list(self.retried)
        known = set(self.retried)
        for key in keys:
            if key not in known:
                merged.append(key)
                known.add(key)
        if len(merged) > config.LEDGER_MAX_HUNKS:
            return replace(
                self, retried=tuple(merged[: config.LEDGER_MAX_HUNKS]),
                saturated=True,
            )
        return replace(self, retried=tuple(merged))

    def already_retried(self, keys: Iterable[str]) -> bool:
        """True once every one of these hunks has had its retry.

        A saturated ledger answers True for everything: it can no longer prove
        a hunk has not been tried, and guessing "not yet" is what turns a
        failing post into an unbounded replay.
        """
        if self.saturated:
            return True
        known = set(self.retried)
        return all(key in known for key in keys)

    def keep_only(self, live: set) -> "Ledger":
        """Drops keys for hunks the diff no longer contains.

        hunk_key is content-based and record only ever appends, so a
        long-lived merge request accumulated a key for every hunk that ever
        existed in it. LEDGER_MAX_HUNKS then stopped being a ceiling on how big
        an MR may be and became a countdown on how old it may get: ten files of
        five hunks reached 1450 keys in a hundred pushes on fifty live hunks.
        Pruning to the current diff also gives saturation an exit.
        """
        if (len(self.hunks) + len(self.retried) < config.LEDGER_MAX_HUNKS
                and not self.saturated):
            # Absent from one diff is not the same as never existed. A rebase
            # onto a main that already carries some of your commits drops a
            # file for a single push, and pruning eagerly made every finding in
            # it post again on the next one — measured at ten re-posts for a
            # file that flapped ten times. Collect only under pressure: the
            # duplicate is worth paying when the alternative is going blind,
            # and not before.
            return self
        hunks = tuple(k for k in self.hunks if k in live)
        retried = tuple(k for k in self.retried if k in live)
        saturated = self.saturated and len(hunks) >= config.LEDGER_MAX_HUNKS
        return replace(self, hunks=hunks, retried=retried, saturated=saturated)

    def report_outage(self) -> "Ledger":
        """Marks that the human has been told the backend produced nothing."""
        return replace(self, outage_reported=True)

    def clear_outage(self) -> "Ledger":
        return replace(self, outage_reported=False)

    def mark_oversized(self) -> "Ledger":
        return replace(self, oversized=True)

    def report_hygiene(self, key: str) -> "Ledger":
        """Records which hygiene issues the MR has been told about. Empty
        clears the record, so a relapse is nagged about again."""
        return replace(self, hygiene=key)

    def at_head(self, head: str) -> "Ledger":
        return replace(self, head=head)


def to_marker(value: Ledger) -> str:
    payload = {
        "head": value.head,
        "posted": value.posted,
        "muted": value.muted,
        "oversized": value.oversized,
        "outage_reported": value.outage_reported,
        "saturated": value.saturated,
        "retried": list(value.retried),
        "hunks": list(value.hunks),
        "hygiene": value.hygiene,
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

    outage_reported = payload.get("outage_reported", False)
    if not isinstance(outage_reported, bool):
        raise LedgerUnavailable(
            "state marker field 'outage_reported' must be a bool, got "
            f"{type(outage_reported).__name__}",
        )

    saturated = payload.get("saturated", False)
    if not isinstance(saturated, bool):
        raise LedgerUnavailable(
            f"state marker field 'saturated' must be a bool, got "
            f"{type(saturated).__name__}",
        )

    hunks = payload.get("hunks", [])
    if not isinstance(hunks, list) or not all(isinstance(key, str) for key in hunks):
        raise LedgerUnavailable(
            "state marker field 'hunks' must be a list of strings",
        )

    retried = payload.get("retried", [])
    if not isinstance(retried, list) or not all(isinstance(k, str) for k in retried):
        raise LedgerUnavailable(
            "state marker field 'retried' must be a list of strings",
        )

    hygiene = payload.get("hygiene", "")
    if not isinstance(hygiene, str):
        raise LedgerUnavailable(
            "state marker field 'hygiene' must be a string, got "
            f"{type(hygiene).__name__}",
        )

    return Ledger(
        head=head,
        posted=posted,
        muted=muted,
        oversized=oversized,
        outage_reported=outage_reported,
        saturated=saturated,
        hunks=tuple(hunks),
        retried=tuple(retried),
        hygiene=hygiene,
    )


def render_note(value: Ledger) -> str:
    head = value.head or "—"
    if value.saturated:
        # The one status the reader must not have to guess at: the bot is not
        # quiet because there is nothing to say, it is quiet because it has
        # lost track of what it has already said.
        status = "переріс памʼять — не рев'ю"
    elif value.muted:
        status = "заглушений"
    else:
        status = "активний"
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
