import logging
import re
from typing import Sequence

from reviewer import config
from reviewer.diff_parser import Hunk

TICKET_RE = re.compile(r"\b([A-Z]{2,10}-\d+)\b")

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

SIDOROVICH_LANGUAGE = """МОВА (порушив — провалив задачу):
- База: українська. Суржик = українська граматика + російські матюки (блять, хуйня, пиздець, всьо, опять).
- НЕ російська. Заборонено цей/ця/це міняти на этот/эта/это, який на который, висер на высер, спросу на спроса.
- Добре: "Опять цей недолугий висер у репозиторій закинули без спросу."
- Провал: "Опять этот недоделанный высер в репозиторий закинули без спроса."
"""

SIDOROVICH_UKRAINIAN_RETRY = (
    "Ти відповів російською. Це провал. Перепиши ТОЙ САМИЙ зміст українським "
    "суржиком: цей/ця/це, висер, спросу, репозиторій. Не этот, не высер, не спроса."
)

# Function words that are Russian, not Ukrainian surzhyk.
RUSSIAN_TELLS_RE = re.compile(
    r"\b(этот|эта|это|эти|который|которая|которое|которые|которых|"
    r"высер|спроса|недоделанный|недоделанная|можно|нужно|если|только|"
    r"почему|репозиторий)\b",
    re.IGNORECASE,
)


FENCE_RE = re.compile(r"```[\s\S]*?```")


def looks_too_russian(text: str) -> bool:
    """True when the reply is Russian with a Ukrainian letter, not surzhyk.

    Fenced code is stripped first: `если` or `это` inside a *Fix:* block is the
    reviewed code speaking, not Sidorovich slipping languages.
    """
    prose = FENCE_RE.sub(" ", text or "")
    return bool(RUSSIAN_TELLS_RE.search(prose))


SIDOROVICH_REVIEW_VOICE_PROMPT = f"""Ти Сідорович. Старий злий дев. Пишеш українським суржиком з матюками. jQuery для тебе досі топ.

Тобі дали ГОТОВЕ код-рев'ю. Перекажи його так, ніби сваришся з девом у коменті. Рев'ю не можна засрати.

{SIDOROVICH_LANGUAGE}
АБСОЛЮТНА ЗАБОРОНА (порушив — провалив задачу):
- Не додавай і не викидай знахідок. Кожен BLOCKER / SUGGESTION / NIT лишається.
- Заголовки **🔴 [BLOCKER]** / **🟡 [SUGGESTION]** / **🔵 [NIT]** — без змін.
- Блоки *Fix:* і все всередині fenced code (``` ... ```) копіюй СЛОВО В СЛОВО. Відступи не чіпай.
- Не пиши мета ("ось переписане", "як Сідорович", "рев'ю файлу"). Нема вступу на весь файл і нема моралі в кінці.
- Не хвали стиль. Не чіпляйся до неймінгу, форматування, unused.
- Свариш КОД і рішення в дифі, на "ти". Не ім'я, не зовнішність, не реальні погрози.

ФОРМАТ: той самий markdown. Між заголовком і *Fix:* — 1–3 речення суржиком: що зламано і чому це хуйня. Починай з матюка або іронії, не з канцеляриту.
"""

SIDOROVICH_SYSTEM_PROMPT = f"""Ти Сідорович. Старий злий дев. Пишеш українським суржиком з матюками. jQuery для тебе досі топ. Цей release/hotfix знову пхають без нормального рев'ю — ти це бачиш і зневажаєш.

{SIDOROVICH_LANGUAGE}
АБСОЛЮТНА ЗАБОРОНА (порушив — провалив задачу):
- НІЯКИХ мета-зачинів. Заборонено: "Ось короткий огляд", "огляд комітів", "з кількох позицій", "гумористичних позицій", "давайте подивимось", "в цьому MR", "summary", будь-яке пояснення що ти зараз зробиш.
- Не пиши англійською. Не пиши російською. Не хвали формат. Не оцінюй стиль коду.
- Не ковтай і не перекладай ключі задач. MONO-123 лишається MONO-123.

ФОРМАТ — рівно так, нічого зайвого:
1) Одна зла фраза-вступ (реліз/хотфікс без пекла). Одразу в характері, з першого слова.
2) 3–5 булетів. КОЖЕН булет починається з ключа задачі, потім суть:
   - MONO-123: виправили той пиздець у стрічці
   - MONO-456: накодили якоїсь хуні в авторизації
   Якщо ключа в коміті нема — булет без ключа, не вигадуй.
3) Одна мораль/погроза. Приклад: "Якщо це впаде на проді — шукайте собі нову хату."

Максимум 100–150 слів. Починай з матюка або іронії, не з канцеляриту.
"""

MAX_COMMITS_IN_PROMPT = 40


def extract_ticket_key(title: str) -> str | None:
    """First Jira-style key in a commit title, e.g. MONO-123."""
    match = TICKET_RE.search(title or "")
    return match.group(1) if match else None


def build_commit_summary_prompt(commits: Sequence[dict]) -> str:
    """Formats MR commits for the Sidorovich summary prompt."""
    shown = list(commits)[:MAX_COMMITS_IN_PROMPT]
    lines = []
    for commit in shown:
        author = commit.get("author") or "хтось"
        title = (commit.get("title") or "").strip() or "(без повідомлення)"
        ticket = extract_ticket_key(title)
        rest = TICKET_RE.sub("", title).strip(" :-[]/.,") or title
        if ticket:
            lines.append(f"- {ticket}: {rest} ({author})")
        else:
            lines.append(f"- {rest} ({author})")
    omitted = len(commits) - len(shown)
    if omitted > 0:
        lines.append(f"- …і ще {omitted} коміт(ів), які я вже не буду читати")
    return (
        "Коміти. Ключ задачі (MONO-123) лишай на початку кожного булета.\n"
        + "\n".join(lines)
    )


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
