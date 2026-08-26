import logging
import re
import unicodedata
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
- База: українська. Суржик = українська граматика + російська лайка.
- НЕ російська. Заборонено цей/ця/це міняти на этот/эта/это, який на который, висер на высер, спросу на спроса.
- Приклади нижче — ПРО ЧАЙНИК, і це навмисне. Вони вчать граматики, а не дають
  тобі готову фразу. Не тягни їхні слова у відповідь.
- Добре: "Цей чайник знову не гріє, бо хтось його спалив нахуй."
- Провал: "Этот чайник опять не греет, потому что кто-то его сжёг."
"""

SIDOROVICH_UKRAINIAN_RETRY = (
    "Ти зіпсував мову: або з'їхав у російську, або вкинув літери з чужої "
    "абетки. Перепиши ТОЙ САМИЙ зміст українським суржиком: цей/ця/це, висер, "
    "спросу, репозиторій. Не этот, не высер, не спроса. Тільки кирилиця й "
    "латиниця — жодних інших абеток."
)

# Function words that are Russian, not Ukrainian surzhyk.
RUSSIAN_TELLS_RE = re.compile(
    r"\b(этот|эта|это|эти|который|которая|которое|которые|которых|"
    r"высер|спроса|недоделанный|недоделанная|можно|нужно|если|только|"
    r"почему|репозиторий|"
    # Seen in shipped roasts, all waved through by the list above: "різати
    # глаза", "все развалиться", "оптимизація автотестів", "дебильний дефіс",
    # and a bare "и" where Ukrainian wants "і"/"й".
    r"глаз|глаза|глазах|развалит\w*|развалят\w*|оптимиз\w*|дебильн\w*|и)\b",
    re.IGNORECASE,
)

# Letters from a script that is neither Latin nor Cyrillic. Real roasts have
# come back with "هاي ця фігня" and "наፈላли повідомлень" — the model dropping a
# token from another alphabet mid-word. Only LETTERS count: emoji, box drawing
# and punctuation are category S/P and must stay legal, since findings are
# rendered with 📄 and 🔴.
def _is_foreign_letter(ch: str) -> bool:
    if not ch.isalpha():
        return False
    name = unicodedata.name(ch, "")
    return not name.startswith(("LATIN", "CYRILLIC", "GREEK"))


FENCE_RE = re.compile(r"```[\s\S]*?```")


def has_foreign_script(text: str) -> bool:
    """True when the reply contains letters from a third alphabet.

    Fenced code is stripped first: a *Fix:* block may legitimately quote a
    string in any language, and rejecting the rewrite for that would lose the
    finding's code.
    """
    prose = FENCE_RE.sub(" ", text or "")
    return any(_is_foreign_letter(ch) for ch in prose)


def unusable_language(text: str) -> bool:
    """True when a reply must be thrown back at the model."""
    return looks_too_russian(text) or has_foreign_script(text)


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

SIDOROVICH_SYSTEM_PROMPT = f"""Ти Сідорович. Старий злий дев. Пишеш українським суржиком з матюками. jQuery для тебе досі топ. Реліз їде повз твої руки, і ти це зневажаєш. Своїми словами — ця фраза не для переказу.

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
3) Одне закриття. Тип закриття тобі дадуть — тримайся його.
   Не починай останнє речення зі слова "Якщо". Не пиши "пішли всі нахуй".

Максимум 100–150 слів. Не канцелярит.
Зачин тобі дадуть готовий — почни рівно з нього, слово в слово, і далі вже сам.
ЗАБОРОНЕНІ ТІКИ (вони вже всім набридли): "без нормального рев'ю", "недолугий
висер", "закинули в репозиторій", "пішли всі нахуй", "шукайте собі нову роботу".
"""

MAX_COMMITS_IN_PROMPT = 40


def extract_ticket_key(title: str) -> str | None:
    """First Jira-style key in a commit title, e.g. MONO-123."""
    match = TICKET_RE.search(title or "")
    return match.group(1) if match else None


def build_commit_summary_prompt(
    commits: Sequence[dict], opener: str = "", closer: str = "",
) -> str:
    """Formats MR commits for the Sidorovich summary prompt.

    `opener` is the exact first phrase the roast must start with; `closer` is
    the kind of closing move to end on. Both exist because the model collapsed
    onto one template: every roast opened on the same swear and all ten of a
    sampled ten closed on "Якщо це розвалить — пішли всі нахуй".
    """
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
    head = "Коміти. Ключ задачі (MONO-123) лишай на початку кожного булета.\n"
    if opener:
        head = (
            f"Твій зачин на цей раз: «{opener}»\n"
            "Почни відповідь рівно з цієї фрази, слово в слово, і далі вже "
            "своїми. Не повторюй її вдруге.\n\n" + head
        )
    if closer:
        head = f"Тип закриття на цей раз: {closer}\n" + head
    return head + "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Conservative char-to-token estimate for code-heavy prompts."""
    return max(1, len(text) // 3)


def input_token_budget() -> int:
    """Tokens available for the user prompt after system prompt and output reserve.

    Sized from the review backend, not from Ollama: review runs on OpenRouter
    and OLLAMA_NUM_CTX now governs only the Sidorovich voice fallback.
    """
    budget = (
        config.REVIEW_CONTEXT_TOKENS
        - config.REVIEW_MAX_OUTPUT_TOKENS
        - estimate_tokens(SYSTEM_PROMPT)
        - config.PROMPT_TOKEN_BUFFER
    )
    if budget < 256:
        logging.warning(
            "REVIEW_CONTEXT_TOKENS=%s leaves only ~%s input tokens; raise it or "
            "lower REVIEW_MAX_OUTPUT_TOKENS",
            config.REVIEW_CONTEXT_TOKENS, budget,
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
