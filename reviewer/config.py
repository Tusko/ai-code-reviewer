import logging
import os

def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.warning("Invalid int for %s=%r, using default %s", name, raw, default)
        return default

def env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items or list(default)

def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

# GitLab
GITLAB_URL = os.environ.get("GITLAB_URL", "https://gitlab.com")
GITLAB_TOKEN = os.environ.get("GITLAB_TOKEN")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")
# The account GITLAB_TOKEN belongs to. Only needed when the bot cannot ask
# GitLab who it is (see gitlab_client.bot_username); it exists so that a
# locked-down token cannot disable the /review command outright.
SIDOROVICH_BOT_USERNAME = os.environ.get("SIDOROVICH_BOT_USERNAME", "")

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_NUM_CTX = env_int("OLLAMA_NUM_CTX", 8192)
OLLAMA_NUM_PREDICT = env_int("OLLAMA_NUM_PREDICT", 320)
# Metal default is 512. The previous hardcoded 128 slowed prompt eval 2-4x.
OLLAMA_NUM_BATCH = env_int("OLLAMA_NUM_BATCH", 512)

# Prompt budget
PROMPT_TOKEN_BUFFER = env_int("PROMPT_TOKEN_BUFFER", 128)
# Off, and a 15-line window, were sized for an 8192-token Ollama context. The
# review backend now budgets a quarter of a million tokens per call, and with
# context off the model saw the hunk and nothing else — 83 tokens for a change
# in a 722-line file. 80 lines costs about 2.3k tokens, roughly 1% of budget.
INCLUDE_FILE_CONTEXT = env_bool("INCLUDE_FILE_CONTEXT", True)
CONTEXT_WINDOW = env_int("CONTEXT_WINDOW", 80)

def paid_model(name: str) -> str:
    """Strips a `:free` suffix. On OpenRouter the paid slug is the same one.

    Free variants share one saturated upstream pool and 429 constantly, which
    trips the voice circuit breaker and leaves half an MR dry. Rewriting is
    loud rather than silent: an operator who typed `:free` on purpose has to
    see that it was overruled.
    """
    stripped = (name or "").strip()
    if stripped.endswith(":free"):
        stripped = stripped[: -len(":free")]
        logging.warning(
            "Ignoring the :free variant of %s; using the paid slug %s",
            name.strip(), stripped,
        )
    return stripped


def env_models(name: str, default: list[str]) -> list[str]:
    """env_list with every `:free` variant rewritten and duplicates dropped."""
    seen, models = set(), []
    for raw in env_list(name, default):
        model = paid_model(raw)
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return models


# OpenRouter — used only for release/hotfix Sidorovich summaries.
# Empty key keeps those summaries on local Ollama.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or None
# The voice runs once per finding, so it is the highest-volume call here — but
# a fully budgeted MR costs under a cent whichever of these you pick, so the
# choice is latency and instruction-following, not price. Flash-Lite is the
# low-latency tier (VOICE_DEADLINE_S is 20s) and has to reproduce every *Fix:*
# code fence byte-identically or preserves_findings throws the rewrite away.
OPENROUTER_MODEL = paid_model(
    os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite"),
)
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1",
)
# Extra models, tried in order after OPENROUTER_MODEL within a single request.
# `:free` variants share one saturated upstream pool and 429 constantly; naming
# a paid model here lets OpenRouter reroute instead of failing the summary.
# A different family on purpose: a Gemini-side outage should not take the
# voice down, and OpenRouter walks this list inside one request.
OPENROUTER_FALLBACK_MODELS = [
    m for m in env_models("OPENROUTER_FALLBACK_MODELS", ["google/gemma-4-26b-a4b-it"])
    if m != OPENROUTER_MODEL
]
# The commit roast is 100-150 words, but Cyrillic costs roughly twice the
# tokens per character that English does on these tokenizers, so 512 clipped
# the closing line off longer release lists.
OPENROUTER_MAX_TOKENS = env_int("OPENROUTER_MAX_TOKENS", 1024)
# qwen2.5-coder and friends cannot write Ukrainian surzhyk; letting them try
# produces gibberish in Sidorovich's name. Off means: no OpenRouter, no roast —
# release/hotfix MRs get a plain commit digest instead.
SIDOROVICH_OLLAMA_FALLBACK = env_bool("SIDOROVICH_OLLAMA_FALLBACK", False)

# Review backend. Ollama no longer serves the review path; it remains only as
# the optional Sidorovich voice fallback above.
OPENROUTER_REVIEW_MODEL = paid_model(
    os.environ.get("OPENROUTER_REVIEW_MODEL", "poolside/laguna-s-2.1"),
)
# Extra models tried in order after OPENROUTER_REVIEW_MODEL within one request.
# Empty by default: a paid model already reroutes across providers on its own.
OPENROUTER_REVIEW_FALLBACK_MODELS = [
    m for m in env_models("OPENROUTER_REVIEW_FALLBACK_MODELS", [])
    if m != OPENROUTER_REVIEW_MODEL
]
# Deliberately below the model's 1_048_576 window. Prompt tokens are billed and
# estimate_tokens is a pessimistic len//3; one file whose diff exceeds this is
# not something a single review call should attempt. Raise it to use the rest.
REVIEW_CONTEXT_TOKENS = env_int("REVIEW_CONTEXT_TOKENS", 262144)
# Deliberately below the model's 131_072 ceiling. A single-file review needing
# more than this is producing a wall of comments, which is what we prevent.
REVIEW_MAX_OUTPUT_TOKENS = env_int("REVIEW_MAX_OUTPUT_TOKENS", 4096)

# The voice rewrite has to re-emit the whole finding, *Fix:* code fences
# included, and preserves_findings compares those fences byte for byte. A
# rewrite that runs out of room fails that check and the comment silently ships
# dry, so this is DERIVED from the review ceiling rather than guessed: whatever
# a review is allowed to produce, the voice must be able to reproduce, plus
# roughly half again for Cyrillic, which costs about twice the tokens per
# character that English does.
OPENROUTER_VOICE_MAX_TOKENS = env_int(
    "OPENROUTER_VOICE_MAX_TOKENS", REVIEW_MAX_OUTPUT_TOKENS * 3 // 2,
)

# Tone. Off keeps summaries and inline comments dry. On adds a meme to the
# summary and, when OpenRouter is keyed, rewrites inline findings as
# Sidorovich. The bugs themselves still come from the local coder model.
SNARK = env_bool("SNARK", True)

# Deadlines and limits
PER_FILE_TIMEOUT_S = env_int("PER_FILE_TIMEOUT_S", 90)
MR_TIMEOUT_S = env_int("MR_TIMEOUT_S", 480)
MAX_FILES = env_int("MAX_FILES", 40)
# Reviewable files above which per-file review is skipped entirely and the MR
# gets one note instead. A 300-file MR produced 592 comments before this.
MAX_MR_FILES = env_int("MAX_MR_FILES", 60)
# Comments the bot may post in one MR across its whole lifetime, counting inline
# discussions, summaries and the oversized note. The state note is excluded: it
# is created once and edited thereafter.
MR_COMMENT_BUDGET = env_int("MR_COMMENT_BUDGET", 30)
# Reviewed hunk keys retained per MR before the oldest are dropped. 2000 keys is
# roughly 30 KB of marker against a 1 MB GitLab note limit.
LEDGER_MAX_HUNKS = env_int("LEDGER_MAX_HUNKS", 2000)
# Global kill switch. False makes the webhook a no-op without touching GitLab.
SIDOROVICH_ENABLED = env_bool("SIDOROVICH_ENABLED", True)
QUEUE_MAXSIZE = env_int("QUEUE_MAXSIZE", 32)
DEDUPE_CACHE_SIZE = env_int("DEDUPE_CACHE_SIZE", 256)

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.environ.get("LOG_FILE")

def configure_logging() -> None:
    handlers = [logging.StreamHandler()]
    if LOG_FILE:
        handlers.append(logging.FileHandler(LOG_FILE))
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )
