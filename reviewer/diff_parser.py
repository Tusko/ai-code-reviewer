import re
from dataclasses import dataclass

# Both sides may omit the count when the range is a single line:
#   @@ -12 +12,3 @@
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Hunk:
    old_start: int
    new_start: int
    lines: tuple[str, ...]

    def added_lines(self) -> list[tuple[int, str]]:
        """Returns (new_file_line_number, text) for every added line."""
        out: list[tuple[int, str]] = []
        lineno = self.new_start
        for line in self.lines:
            if line.startswith("-"):
                continue
            if line.startswith("+"):
                out.append((lineno, line[1:]))
            lineno += 1
        return out

    def first_added_line(self) -> int | None:
        added = self.added_lines()
        return added[0][0] if added else None


@dataclass(frozen=True)
class FileDiff:
    old_path: str
    new_path: str
    is_new: bool
    is_deleted: bool
    is_renamed: bool
    is_binary: bool
    hunks: tuple[Hunk, ...]

    @property
    def total_lines(self) -> int:
        return sum(len(h.added_lines()) for h in self.hunks)


def _looks_binary(diff_text: str) -> bool:
    if "\x00" in diff_text:
        return True
    return diff_text.lstrip().startswith("Binary files ")


def parse_hunks(diff_text: str) -> tuple[Hunk, ...]:
    if not diff_text or _looks_binary(diff_text):
        return ()

    hunks: list[Hunk] = []
    old_start = new_start = 0
    body: list[str] | None = None

    def flush() -> None:
        if body is not None:
            hunks.append(Hunk(old_start, new_start, tuple(body)))

    for line in diff_text.splitlines():
        match = HUNK_RE.match(line)
        if match:
            flush()
            old_start = int(match.group(1))
            new_start = int(match.group(3))
            body = []
            continue
        if body is None:
            # Preamble: ---, +++, index, diff --git. Ignored.
            continue
        if line.startswith("\\"):
            # "\ No newline at end of file" is metadata, not content.
            continue
        if line == "":
            # Some producers emit a bare empty line for an empty context line.
            body.append(" ")
            continue
        if line[0] not in "+- ":
            continue
        body.append(line)

    flush()
    return tuple(hunks)


def file_diff_from_change(change: dict) -> FileDiff:
    """Builds a FileDiff from one entry of python-gitlab's mr.changes()['changes']."""
    diff_text = change.get("diff") or ""
    return FileDiff(
        old_path=change.get("old_path") or "",
        new_path=change.get("new_path") or "",
        is_new=bool(change.get("new_file")),
        is_deleted=bool(change.get("deleted_file")),
        is_renamed=bool(change.get("renamed_file")),
        is_binary=_looks_binary(diff_text),
        hunks=parse_hunks(diff_text),
    )
