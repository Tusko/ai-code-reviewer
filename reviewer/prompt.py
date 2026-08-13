import logging
from typing import Sequence

from reviewer import config
from reviewer.diff_parser import Hunk

SYSTEM_PROMPT = """Act as a strict Principal Software Engineer reviewing the changed
lines of ONE file. Lines marked `+` were added or modified. Unmarked lines are
unchanged context, shown only so you can understand the change. Deleted lines are
not shown to you at all.

Find ONLY logic errors, security vulnerabilities (SQL injection, XSS, exposed
credentials), or severe performance bugs in the lines marked `+`.

### ABSOLUTE BANS (CRITICAL TO OBEY):
1. NEVER complain about "unused", "undeclared", or "missing" variables, methods, or imports. You see a fragment; assume they exist elsewhere.
2. NEVER complain about "duplicated" methods or blocks. The diff repeats context.
3. NEVER flag formatting, docstrings, naming conventions, or style.
4. NEVER comment on lines not marked `+`.

### OUTPUT
Markdown. No preamble, no closing remarks. Use only these headings, each followed by
the problem in plain prose and then a `*Fix:*` fenced code block:

**🔴 [BLOCKER]** — crash risk, data loss, or severe security flaw
**🟡 [SUGGESTION]** — logic bug, unhandled edge case, or N+1 query
**🔵 [NIT]** — data validation, safer SQL, or stricter type casting ONLY

If and ONLY if the code has no logic or security issues, output EXACTLY:
[LGTM]
"""


def estimate_tokens(text: str) -> int:
    """Conservative char-to-token estimate for code-heavy prompts."""
    return max(1, len(text) // 3)


def input_token_budget() -> int:
    """Tokens available for the user prompt after system prompt and output reserve."""
    budget = (
        config.OLLAMA_NUM_CTX
        - config.OLLAMA_NUM_PREDICT
        - estimate_tokens(SYSTEM_PROMPT)
        - config.PROMPT_TOKEN_BUFFER
    )
    if budget < 256:
        logging.warning(
            "OLLAMA_NUM_CTX=%s leaves only ~%s input tokens; raise num_ctx or "
            "lower OLLAMA_NUM_PREDICT",
            config.OLLAMA_NUM_CTX, budget,
        )
    return max(256, budget)


def fits(prompt: str) -> bool:
    return estimate_tokens(prompt) <= input_token_budget()


def render_hunk(hunk: Hunk) -> str:
    """Renders added and context lines with new-file line numbers. Removed lines are dropped."""
    out = []
    lineno = hunk.new_start
    for line in hunk.lines:
        if line.startswith("-"):
            continue
        marker = "+" if line.startswith("+") else " "
        out.append(f"{lineno:>6} {marker} {line[1:]}")
        lineno += 1
    return "\n".join(out)


def build_file_prompt(path: str, hunks: Sequence[Hunk], context: str = "") -> str:
    parts = []
    if context:
        parts.append(
            f"=== START FILE CONTEXT: {path} ===\n{context}\n=== END FILE CONTEXT ===\n"
        )
    body = "\n...\n".join(render_hunk(h) for h in hunks)
    parts.append(
        f"=== START CHANGED LINES: {path} ===\n{body}\n=== END CHANGED LINES ===\n"
    )
    return "\n".join(parts)
