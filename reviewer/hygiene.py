"""MR hygiene: the title format and whether anyone is on the hook for the MR.

This is not a code review — it never calls a model. The checks are cheap and
deterministic, so the nag is a local template with a meme on top.
"""

import re

from reviewer import config
from reviewer.memes import snark

TITLE_RE = re.compile(config.MR_TITLE_PATTERN)
TICKET_RE = re.compile(r"[A-Z][A-Z0-9]+-\d+")

# The order issues are reported and keyed in. The key goes into the ledger, so
# it must not depend on the order the checks happened to run in.
ISSUE_ORDER = ("title", "assign")


def is_valid_title(title: str) -> bool:
    return bool(TITLE_RE.match(title or ""))


def check(title: str, assignees, reviewers) -> list[str]:
    """Issue codes for one MR, in ISSUE_ORDER."""
    issues = []
    if not is_valid_title(title):
        issues.append("title")
    if not assignees and not reviewers:
        issues.append("assign")
    return issues


def issue_key(issues) -> str:
    """Canonical key for a set of issues, stored in the ledger to dedupe nags."""
    return ",".join(code for code in ISSUE_ORDER if code in issues)


def suggest_title(title: str) -> str:
    """Rebuilds a compliant title from a malformed one, so the nag can show it."""
    raw = (title or "").strip()
    match = TICKET_RE.search(raw)
    ticket = match.group(0) if match else "MONO-0000"

    if ":" in raw:
        description = raw.rsplit(":", 1)[1].strip()
    else:
        description = raw.replace(ticket, "").strip(" -:()[]")

    return f"{ticket}: {description or '<опис змін>'}"


def render(issues, title: str) -> str:
    lines = []
    if config.SNARK:
        lines.append(f"_{snark()}_\n")

    if "title" in issues:
        lines.append(
            "**Заголовок MR ні до чого не привʼязаний.**\n\n"
            f"Зараз: `{title}`\n\n"
            f"Треба: `{suggest_title(title)}`\n\n"
            "Ключ задачі має стояти на початку голим — без `fix(...)` і `feat(...)`. "
            "Інакше GitLab не звʼяже MR із задачею, і шукати його потім нема як.\n"
        )
    if "assign" in issues:
        lines.append(
            "**MR ні на кому не висить.**\n\n"
            "Ні assignee, ні reviewer. Постав хоч когось — інакше він тут "
            "стоятиме, доки не протухне.\n"
        )
    return "\n".join(lines)
