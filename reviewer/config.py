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

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_NUM_CTX = env_int("OLLAMA_NUM_CTX", 8192)
OLLAMA_NUM_PREDICT = env_int("OLLAMA_NUM_PREDICT", 320)
# Metal default is 512. The previous hardcoded 128 slowed prompt eval 2-4x.
OLLAMA_NUM_BATCH = env_int("OLLAMA_NUM_BATCH", 512)

# Prompt budget
PROMPT_TOKEN_BUFFER = env_int("PROMPT_TOKEN_BUFFER", 128)
INCLUDE_FILE_CONTEXT = env_bool("INCLUDE_FILE_CONTEXT", False)
CONTEXT_WINDOW = env_int("CONTEXT_WINDOW", 15)

# OpenRouter — used only for release/hotfix Sidorovich summaries.
# Empty key keeps those summaries on local Ollama.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or None
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free",
)
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1",
)
# Extra models, tried in order after OPENROUTER_MODEL within a single request.
# `:free` variants share one saturated upstream pool and 429 constantly; naming
# a paid model here lets OpenRouter reroute instead of failing the summary.
OPENROUTER_FALLBACK_MODELS = env_list("OPENROUTER_FALLBACK_MODELS", [])
OPENROUTER_MAX_TOKENS = env_int("OPENROUTER_MAX_TOKENS", 512)
# The voice rewrite has to re-emit the whole finding, *Fix:* code fences
# included, so it needs far more room than the commit-list roast. Too low and
# every multi-finding review stops at finish_reason=length and silently posts
# dry.
OPENROUTER_VOICE_MAX_TOKENS = env_int("OPENROUTER_VOICE_MAX_TOKENS", 2048)
# qwen2.5-coder and friends cannot write Ukrainian surzhyk; letting them try
# produces gibberish in Sidorovich's name. Off means: no OpenRouter, no roast —
# release/hotfix MRs get a plain commit digest instead.
SIDOROVICH_OLLAMA_FALLBACK = env_bool("SIDOROVICH_OLLAMA_FALLBACK", False)

# Review backend. Ollama no longer serves the review path; it remains only as
# the optional Sidorovich voice fallback above.
OPENROUTER_REVIEW_MODEL = os.environ.get(
    "OPENROUTER_REVIEW_MODEL", "poolside/laguna-s-2.1",
)
# Extra models tried in order after OPENROUTER_REVIEW_MODEL within one request.
# Empty by default: a paid model already reroutes across providers on its own.
OPENROUTER_REVIEW_FALLBACK_MODELS = env_list("OPENROUTER_REVIEW_FALLBACK_MODELS", [])
# Deliberately below the model's 1_048_576 window. Prompt tokens are billed and
# estimate_tokens is a pessimistic len//3; one file whose diff exceeds this is
# not something a single review call should attempt. Raise it to use the rest.
REVIEW_CONTEXT_TOKENS = env_int("REVIEW_CONTEXT_TOKENS", 262144)
# Deliberately below the model's 131_072 ceiling. A single-file review needing
# more than this is producing a wall of comments, which is what we prevent.
REVIEW_MAX_OUTPUT_TOKENS = env_int("REVIEW_MAX_OUTPUT_TOKENS", 4096)

# Tone. Off keeps summaries and inline comments dry. On adds a meme to the
# summary and, when OpenRouter is keyed, rewrites inline findings as
# Sidorovich. The bugs themselves still come from the local coder model.
SNARK = env_bool("SNARK", True)

# Deadlines and limits
PER_FILE_TIMEOUT_S = env_int("PER_FILE_TIMEOUT_S", 90)
MR_TIMEOUT_S = env_int("MR_TIMEOUT_S", 480)
MAX_FILES = env_int("MAX_FILES", 40)
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
