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
