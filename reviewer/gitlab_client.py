import hashlib
import logging
from typing import Sequence

import gitlab

from reviewer import config
from reviewer.diff_parser import FileDiff, Hunk, file_diff_from_change

_client = None


def get_client():
    global _client
    if _client is None and config.GITLAB_TOKEN:
        _client = gitlab.Gitlab(config.GITLAB_URL, private_token=config.GITLAB_TOKEN)
    if _client is None:
        logging.warning("GITLAB_TOKEN not provided; GitLab calls will fail")
    return _client


def fetch_mr(project_id: int, mr_iid: int):
    client = get_client()
    project = client.projects.get(project_id)
    return project, project.mergerequests.get(mr_iid)


def fetch_file_diffs(mr) -> list[FileDiff]:
    changes = mr.changes().get("changes", [])
    return [file_diff_from_change(change) for change in changes]


def fetch_file_content(project, path: str, ref: str) -> str:
    try:
        blob = project.files.get(file_path=path, ref=ref)
        return blob.decode().decode("utf-8")
    except Exception as exc:
        logging.warning("Could not fetch %s@%s: %s", path, ref, exc)
        return ""


def surgical_context(full_text: str, hunks: Sequence[Hunk], window: int) -> str:
    """Returns non-overlapping context windows around each hunk, in file order."""
    lines = full_text.splitlines()
    if not lines:
        return ""

    ranges: list[list[int]] = []
    for hunk in hunks:
        start = max(0, hunk.new_start - 1 - window)
        end = min(len(lines), hunk.new_start - 1 + window)
        if start >= end:
            continue
        if ranges and start <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])

    blocks = [
        f"Lines {start + 1}-{end}:\n" + "\n".join(lines[start:end])
        for start, end in ranges
    ]
    return "\n...\n".join(blocks)


def post_inline(mr, path: str, new_line: int, body: str) -> bool:
    """Posts an inline discussion. Returns False if GitLab rejects the position."""
    refs = getattr(mr, "diff_refs", None)
    if not refs:
        return False

    # Build position dict outside try so KeyError and AttributeError propagate
    position = {
        "base_sha": refs["base_sha"],
        "start_sha": refs["start_sha"],
        "head_sha": refs["head_sha"],
        "position_type": "text",
        "new_path": path,
        "old_path": path,
        "new_line": new_line,
    }

    try:
        mr.discussions.create({"body": body, "position": position})
        return True
    except gitlab.exceptions.GitlabError as exc:
        logging.info("Inline discussion rejected for %s:%s (%s)", path, new_line, exc)
        return False


def post_note(mr, body: str) -> None:
    mr.notes.create({"body": body})


def diff_fingerprint(project_id: int, mr_iid: int, file_diffs: Sequence[FileDiff]) -> str:
    """Stable hash of project + MR identity and diff content, used for dedupe.

    DedupeCache is a single process-wide instance shared across every project
    the bot serves, so project_id and mr_iid must be part of the digest —
    otherwise two merge requests with byte-identical diffs (backports,
    cross-project cherry-picks, a re-created MR) collide and the second is
    silently skipped.
    """
    digest = hashlib.sha256()
    digest.update(str(project_id).encode())
    digest.update(b"\x00")
    digest.update(str(mr_iid).encode())
    digest.update(b"\x00")
    for fd in file_diffs:
        digest.update(fd.new_path.encode())
        digest.update(b"\x00")
        for hunk in fd.hunks:
            digest.update(str(hunk.new_start).encode())
            digest.update(b"\x00")
            for line in hunk.lines:
                digest.update(line.encode())
                digest.update(b"\x00")
        digest.update(b"\x00")
    return digest.hexdigest()
