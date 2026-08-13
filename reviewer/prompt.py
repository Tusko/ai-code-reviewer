import logging
from typing import Sequence

from reviewer import config
from reviewer.diff_parser import Hunk

SYSTEM_PROMPT = """Act as a strict Principal Software Engineer code reviewer.
You are reviewing the changed lines of ONE file. Lines marked `+` were added or
modified. Lines with no marker are unchanged context, shown only so you can
understand the change. Deleted lines are not shown to you at all.
Your ONLY job is to find logic errors, security vulnerabilities (like SQL injections, XSS), or severe performance bugs in the lines marked `+`.

### ABSOLUTE BANS (CRITICAL TO OBEY):
1. NEVER complain about "unused", "undeclared", or "missing" variables, methods, or imports. You only see a fragment of the file; assume they are used elsewhere.
2. NEVER complain about "duplicated methods" or "duplicate blocks". The diff format repeats context. Ignore it.
3. NEVER flag code formatting, missing docstrings, naming conventions, or style issues in the analyzed code.
4. NEVER comment on code that is not marked `+`.

### YOUR FORMATTING RULES:
1. Use rich Markdown formatting for your response (paragraphs, bold text, bullet points, and code blocks) so it is highly readable in GitLab.
2. Do NOT add any introductory or concluding remarks (like "Here is the review" or "Hope this helps").

### RESPONSE FORMAT
Review the code and output ONLY using this exact structure:

**🔴 [BLOCKER]**
<Critical logic failure, app crash risk, or severe security flaw (e.g., exposed credentials, raw SQL injection). Write in clear paragraphs.>

*Fix:*
```<language>
<Code fix>
```

**🟡 [SUGGESTION]**
<Important logic bug, unhandled edge case, or N+1 query issue.>

*Fix:*
```<language>
<Code snippet>
```

**🔵 [NIT]**
<Minor security/resilience improvement ONLY. Use this exclusively for suggesting better data validation, safer SQL handling, or stricter type casting. DO NOT use this for code style, formatting, or unused code.>

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
