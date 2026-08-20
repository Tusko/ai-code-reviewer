# AI Code Reviewer Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut per-MR review latency from minutes to seconds and eliminate hangs by splitting the monolithic Flask app into tested modules, reviewing one file at a time against a token budget, and fixing seven confirmed defects.

**Architecture:** `review_server.py` becomes a thin Flask entry point that validates webhooks and enqueues jobs. A single background worker drains an in-process coalescing queue, parses each MR's diffs into typed `FileDiff`/`Hunk` objects, filters unreviewable files, and issues one streaming Ollama request per file under a wall-clock deadline. Results post as inline GitLab discussions anchored to real line numbers, followed by a summary note.

**Tech Stack:** Python 3.11, Flask 3.0, python-gitlab 4.4, requests 2.31, gunicorn 21.2, pytest. Ollama HTTP API on the host. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-08-13-reviewer-speedup-design.md`

## Global Constraints

- Python 3.11 (matches `Dockerfile` base image `python:3.11-slim`).
- No new runtime dependencies. `pytest` is a dev dependency only, added to a new `requirements-dev.txt`.
- Remove the unused `ollama` package from `requirements.txt`. The code calls the HTTP API through `requests`.
- `OLLAMA_NUM_PARALLEL=1` on the host stays. Never issue concurrent Ollama requests.
- Gunicorn must run `--workers 1`. The queue and dedupe cache are in-process; multiple workers break serialization (defect B3).
- Removed (`-`) diff lines are never sent to the model.
- Every skipped file must appear in the summary note with a reason. No silent drops.
- All modules under `reviewer/` are importable without network access. Tests never contact GitLab or Ollama.
- Existing `SYSTEM_PROMPT` ban rules are preserved verbatim; only the scoping sentence changes from whole-MR to single-file.
- `meme_phrases` easter egg is preserved unchanged, relocated to `reviewer/memes.py`.

---

## File Structure

| File | Responsibility |
|---|---|
| `reviewer/config.py` | All environment parsing. Single source of truth for tunables. |
| `reviewer/memes.py` | `meme_phrases` list, moved verbatim. |
| `reviewer/diff_parser.py` | Unified diff text → `FileDiff` / `Hunk` dataclasses with correct line numbers. |
| `reviewer/filters.py` | `is_reviewable(FileDiff) -> (bool, reason)`. |
| `reviewer/prompt.py` | System prompt, hunk rendering, token budget, per-file prompt assembly. |
| `reviewer/ollama_client.py` | Streaming chat with wall-clock abort, returns text + `done_reason` + stats. |
| `reviewer/gitlab_client.py` | Fetch MR changes and file content, post inline discussions and notes. |
| `reviewer/queue.py` | Coalescing FIFO + LRU dedupe cache + worker thread. |
| `reviewer/pipeline.py` | Per-MR orchestration, degradation ladder, summary assembly. |
| `review_server.py` | Flask routes only. Webhook validation and enqueue. |
| `tests/` | pytest suite, no network. |

---

## Task 1: Test scaffolding and configuration module

**Files:**
- Create: `reviewer/__init__.py`
- Create: `reviewer/config.py`
- Create: `reviewer/memes.py`
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: module `reviewer.config` exposing these names — `GITLAB_URL: str`, `GITLAB_TOKEN: str | None`, `WEBHOOK_SECRET: str | None`, `OLLAMA_HOST: str`, `OLLAMA_MODEL: str`, `OLLAMA_NUM_CTX: int`, `OLLAMA_NUM_PREDICT: int`, `OLLAMA_NUM_BATCH: int`, `INCLUDE_FILE_CONTEXT: bool`, `CONTEXT_WINDOW: int`, `PER_FILE_TIMEOUT_S: int`, `MR_TIMEOUT_S: int`, `MAX_FILES: int`, `QUEUE_MAXSIZE: int`, `DEDUPE_CACHE_SIZE: int`, `PROMPT_TOKEN_BUFFER: int`, `LOG_LEVEL: str`, `LOG_FILE: str | None`. Also `env_int(name: str, default: int) -> int` and `env_bool(name: str, default: bool) -> bool`. Module `reviewer.memes` exposing `meme_phrases: list[str]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import importlib


def _reload(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import reviewer.config
    return importlib.reload(reviewer.config)


def test_defaults_match_16gb_tuning(monkeypatch):
    for key in ("OLLAMA_NUM_CTX", "OLLAMA_NUM_PREDICT", "OLLAMA_NUM_BATCH",
                "INCLUDE_FILE_CONTEXT", "CONTEXT_WINDOW", "MAX_FILES"):
        monkeypatch.delenv(key, raising=False)
    cfg = _reload(monkeypatch)
    assert cfg.OLLAMA_NUM_CTX == 8192
    assert cfg.OLLAMA_NUM_PREDICT == 320
    assert cfg.OLLAMA_NUM_BATCH == 512
    assert cfg.INCLUDE_FILE_CONTEXT is False
    assert cfg.CONTEXT_WINDOW == 15
    assert cfg.MAX_FILES == 40


def test_env_overrides_are_applied(monkeypatch):
    cfg = _reload(monkeypatch, OLLAMA_NUM_CTX="4096", INCLUDE_FILE_CONTEXT="true")
    assert cfg.OLLAMA_NUM_CTX == 4096
    assert cfg.INCLUDE_FILE_CONTEXT is True


def test_env_bool_accepts_common_truthy_spellings(monkeypatch):
    import reviewer.config as cfg
    monkeypatch.setenv("SOME_FLAG", "YES")
    assert cfg.env_bool("SOME_FLAG", False) is True
    monkeypatch.setenv("SOME_FLAG", "0")
    assert cfg.env_bool("SOME_FLAG", True) is False


def test_env_int_falls_back_on_garbage(monkeypatch):
    import reviewer.config as cfg
    monkeypatch.setenv("SOME_INT", "not-a-number")
    assert cfg.env_int("SOME_INT", 7) == 7


def test_memes_preserved():
    from reviewer.memes import meme_phrases
    assert len(meme_phrases) == 41
    assert "Nihuyasobi na oborot." in meme_phrases
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reviewer'`

- [ ] **Step 3: Create the package scaffolding**

Create empty `reviewer/__init__.py` and empty `tests/__init__.py`.

Create `requirements-dev.txt`:

```
-r requirements.txt
pytest==8.2.0
```

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
pythonpath = .
```

- [ ] **Step 4: Write the configuration module**

Create `reviewer/config.py`:

```python
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
```

- [ ] **Step 5: Move the memes**

Create `reviewer/memes.py` containing exactly the `meme_phrases` list from `review_server.py:13-55`, copied verbatim with no edits to any string:

```python
meme_phrases = [
    "Ше й в'єбав",
    # ... copy all 41 entries from review_server.py:13-55 unchanged ...
    "Та йди ти нахуй зі своїми порадами"
]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add reviewer/__init__.py reviewer/config.py reviewer/memes.py \
        requirements-dev.txt pytest.ini tests/__init__.py tests/test_config.py
git commit -m "refactor: extract config and memes into reviewer package"
```

---

## Task 2: Diff parser

**Files:**
- Create: `reviewer/diff_parser.py`
- Test: `tests/test_diff_parser.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Hunk` frozen dataclass with fields `old_start: int`, `new_start: int`, `lines: tuple[str, ...]`, and methods `added_lines() -> list[tuple[int, str]]` and `first_added_line() -> int | None`.
  - `FileDiff` frozen dataclass with fields `old_path: str`, `new_path: str`, `is_new: bool`, `is_deleted: bool`, `is_renamed: bool`, `is_binary: bool`, `hunks: tuple[Hunk, ...]`, and property `total_lines: int`.
  - `parse_hunks(diff_text: str) -> tuple[Hunk, ...]`
  - `file_diff_from_change(change: dict) -> FileDiff` — `change` is one entry from python-gitlab's `mr.changes()['changes']`.

This replaces defect B4: the old regex `r'@@ -\d+,\d+ \+(\d+)(?:,\d+)? @@'` required a comma on the old side and silently failed on single-line hunks.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diff_parser.py`:

```python
from reviewer.diff_parser import parse_hunks, file_diff_from_change

MULTI_HUNK = """@@ -1,4 +1,5 @@
 def a():
-    return 1
+    return 2
+    # new
 
@@ -20,3 +21,4 @@ def b():
     x = 1
+    y = 2
     return x
"""

SINGLE_LINE_HUNK = """@@ -12 +12,3 @@
-old
+new_one
+new_two
+new_three
"""

WITH_FILE_HEADERS = """--- a/src/app.py
+++ b/src/app.py
@@ -5,2 +5,3 @@
 keep
+added
"""

NO_NEWLINE_AT_EOF = """@@ -1,2 +1,2 @@
 keep
-old
\\ No newline at end of file
+new
\\ No newline at end of file
"""

AT_IN_CONTEXT = """@@ -3,2 +3,3 @@ class Foo:  # @@ tricky @@
 keep
+added
"""


def test_parses_multiple_hunks():
    hunks = parse_hunks(MULTI_HUNK)
    assert len(hunks) == 2
    assert hunks[0].old_start == 1
    assert hunks[0].new_start == 1
    assert hunks[1].new_start == 21


def test_single_line_old_range_parses():
    # Regression for B4: git emits "@@ -12 +12,3 @@" with no comma on the old side.
    hunks = parse_hunks(SINGLE_LINE_HUNK)
    assert len(hunks) == 1
    assert hunks[0].old_start == 12
    assert hunks[0].new_start == 12


def test_added_lines_carry_correct_new_line_numbers():
    hunks = parse_hunks(MULTI_HUNK)
    assert hunks[0].added_lines() == [(2, "    return 2"), (3, "    # new")]
    assert hunks[1].added_lines() == [(22, "    y = 2")]


def test_removed_lines_do_not_advance_new_line_counter():
    hunks = parse_hunks(SINGLE_LINE_HUNK)
    assert hunks[0].added_lines() == [
        (12, "new_one"), (13, "new_two"), (14, "new_three"),
    ]


def test_file_headers_are_ignored():
    hunks = parse_hunks(WITH_FILE_HEADERS)
    assert len(hunks) == 1
    assert hunks[0].added_lines() == [(6, "added")]


def test_no_newline_marker_is_dropped():
    hunks = parse_hunks(NO_NEWLINE_AT_EOF)
    assert all(not line.startswith("\\") for line in hunks[0].lines)
    assert hunks[0].added_lines() == [(2, "new")]


def test_at_symbols_in_trailing_context_do_not_break_parsing():
    hunks = parse_hunks(AT_IN_CONTEXT)
    assert len(hunks) == 1
    assert hunks[0].new_start == 3
    assert hunks[0].added_lines() == [(4, "added")]


def test_first_added_line():
    hunks = parse_hunks(MULTI_HUNK)
    assert hunks[0].first_added_line() == 2


def test_first_added_line_is_none_when_only_deletions():
    hunks = parse_hunks("@@ -1,2 +1,1 @@\n keep\n-gone\n")
    assert hunks[0].first_added_line() is None


def test_file_diff_from_change_reads_gitlab_flags():
    fd = file_diff_from_change({
        "old_path": "src/app.py",
        "new_path": "src/app.py",
        "new_file": False,
        "deleted_file": False,
        "renamed_file": False,
        "diff": MULTI_HUNK,
    })
    assert fd.new_path == "src/app.py"
    assert fd.is_binary is False
    assert len(fd.hunks) == 2
    assert fd.total_lines == 3


def test_binary_diff_is_detected():
    fd = file_diff_from_change({
        "old_path": "logo.png",
        "new_path": "logo.png",
        "diff": "Binary files a/logo.png and b/logo.png differ\n",
    })
    assert fd.is_binary is True
    assert fd.hunks == ()


def test_empty_diff_yields_no_hunks():
    fd = file_diff_from_change({"old_path": "a", "new_path": "a", "diff": ""})
    assert fd.hunks == ()
    assert fd.total_lines == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diff_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reviewer.diff_parser'`

- [ ] **Step 3: Write the parser**

Create `reviewer/diff_parser.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_diff_parser.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add reviewer/diff_parser.py tests/test_diff_parser.py
git commit -m "feat: add unified diff parser with correct line numbers

Fixes B4: previous hunk regex required a comma on the old side and
silently dropped context for single-line hunks like '@@ -12 +12,3 @@'."
```

---

## Task 3: Reviewable-file filter

**Files:**
- Create: `reviewer/filters.py`
- Test: `tests/test_filters.py`

**Interfaces:**
- Consumes: `reviewer.diff_parser.FileDiff` from Task 2.
- Produces: `is_reviewable(file_diff: FileDiff) -> tuple[bool, str]` — returns `(True, "")` when reviewable, otherwise `(False, reason)` where reason is a short human-readable string used verbatim in the summary note.

- [ ] **Step 1: Write the failing test**

Create `tests/test_filters.py`:

```python
import pytest

from reviewer.diff_parser import FileDiff, Hunk
from reviewer.filters import is_reviewable

HUNK = Hunk(old_start=1, new_start=1, lines=(" keep", "+added"))


def make(path, **kwargs):
    return FileDiff(
        old_path=kwargs.pop("old_path", path),
        new_path=path,
        is_new=kwargs.pop("is_new", False),
        is_deleted=kwargs.pop("is_deleted", False),
        is_renamed=kwargs.pop("is_renamed", False),
        is_binary=kwargs.pop("is_binary", False),
        hunks=kwargs.pop("hunks", (HUNK,)),
    )


def test_normal_source_file_is_reviewable():
    assert is_reviewable(make("src/app.py")) == (True, "")


def test_deleted_file_is_skipped():
    ok, reason = is_reviewable(make("src/app.py", is_deleted=True))
    assert ok is False
    assert reason == "deleted"


def test_binary_file_is_skipped():
    ok, reason = is_reviewable(make("logo.png", is_binary=True, hunks=()))
    assert ok is False
    assert reason == "binary"


def test_rename_without_content_change_is_skipped():
    ok, reason = is_reviewable(make("b.py", old_path="a.py", is_renamed=True, hunks=()))
    assert ok is False
    assert reason == "no content change"


@pytest.mark.parametrize("path", [
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Cargo.lock",
    "go.sum",
    "Gemfile.lock",
    "composer.lock",
])
def test_lockfiles_are_skipped(path):
    ok, reason = is_reviewable(make(path))
    assert ok is False
    assert reason == "lockfile"


@pytest.mark.parametrize("path", [
    "static/app.min.js",
    "static/app.min.css",
    "api/service.pb.go",
    "api/service_pb2.py",
    "src/schema.generated.ts",
])
def test_generated_files_are_skipped(path):
    ok, reason = is_reviewable(make(path))
    assert ok is False
    assert reason == "generated"


@pytest.mark.parametrize("path", [
    "vendor/lib/x.go",
    "dist/bundle.js",
    "build/out.js",
    "node_modules/pkg/index.js",
    "tests/__snapshots__/App.test.js.snap",
    ".venv/lib/site.py",
])
def test_vendored_and_build_output_is_skipped(path):
    ok, reason = is_reviewable(make(path))
    assert ok is False
    assert reason == "vendored or build output"


@pytest.mark.parametrize("path", [
    "assets/logo.svg",
    "assets/photo.jpg",
    "assets/icon.ico",
    "fonts/x.woff2",
    "docs/manual.pdf",
    "static/app.js.map",
])
def test_asset_files_are_skipped(path):
    ok, reason = is_reviewable(make(path))
    assert ok is False
    assert reason == "asset"


@pytest.mark.parametrize("path", [
    "db/migrations/001_add_users.sql",
    "src/vendored_client.py",
    "src/distance.py",
    "src/buildings.py",
])
def test_lookalike_paths_are_not_skipped(path):
    # 'migrations' stays reviewable: raw SQL is exactly what we want reviewed.
    # Substring matches like 'distance' must not trip the 'dist/' rule.
    assert is_reviewable(make(path)) == (True, "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_filters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reviewer.filters'`

- [ ] **Step 3: Write the filter**

Create `reviewer/filters.py`:

```python
import re

from reviewer.diff_parser import FileDiff

LOCKFILE_NAMES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Cargo.lock", "composer.lock", "go.sum", "Gemfile.lock",
    "Pipfile.lock", "mix.lock",
})

ASSET_SUFFIXES = (
    ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf", ".pdf", ".map",
    ".zip", ".gz", ".tar", ".jar", ".so", ".dylib", ".dll",
)

GENERATED_PATTERNS = (
    re.compile(r"\.min\.(js|css)$"),
    re.compile(r"\.pb\.go$"),
    re.compile(r"_pb2(_grpc)?\.py$"),
    re.compile(r"\.generated\.[^/]+$"),
    re.compile(r"\.g\.dart$"),
)

# Matched as whole path segments so 'src/distance.py' is not caught by 'dist'.
VENDOR_DIR_SEGMENTS = frozenset({
    "vendor", "dist", "build", "node_modules", "__snapshots__",
    ".venv", "venv", "site-packages", "third_party",
})


def is_reviewable(file_diff: FileDiff) -> tuple[bool, str]:
    """Returns (True, '') or (False, reason). Reason is shown in the summary note."""
    if file_diff.is_deleted:
        return False, "deleted"
    if file_diff.is_binary:
        return False, "binary"
    if not file_diff.hunks:
        return False, "no content change"

    path = file_diff.new_path
    segments = path.split("/")
    name = segments[-1]

    if name in LOCKFILE_NAMES or name.endswith(".lock"):
        return False, "lockfile"
    for pattern in GENERATED_PATTERNS:
        if pattern.search(path):
            return False, "generated"
    if any(segment in VENDOR_DIR_SEGMENTS for segment in segments[:-1]):
        return False, "vendored or build output"
    if path.endswith(ASSET_SUFFIXES):
        return False, "asset"

    return True, ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_filters.py -v`
Expected: 27 passed

- [ ] **Step 5: Commit**

```bash
git add reviewer/filters.py tests/test_filters.py
git commit -m "feat: skip lockfiles, generated code, vendored dirs and assets"
```

---

## Task 4: Prompt builder and token budget

**Files:**
- Create: `reviewer/prompt.py`
- Test: `tests/test_prompt.py`

**Interfaces:**
- Consumes: `reviewer.diff_parser.Hunk`, `reviewer.diff_parser.FileDiff` (Task 2); `reviewer.config` (Task 1).
- Produces:
  - `SYSTEM_PROMPT: str`
  - `estimate_tokens(text: str) -> int`
  - `input_token_budget() -> int`
  - `render_hunk(hunk: Hunk) -> str`
  - `build_file_prompt(path: str, hunks: Sequence[Hunk], context: str = "") -> str`
  - `fits(prompt: str) -> bool`

Removed lines are never rendered. This replaces the dead `annotate_diff_for_ai` (defect B5) with a cheaper mechanism: the model cannot hallucinate about deleted code it never sees, and input drops roughly 30% on edit-heavy diffs.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompt.py`:

```python
from reviewer.diff_parser import Hunk
from reviewer.prompt import (
    SYSTEM_PROMPT, build_file_prompt, estimate_tokens, fits,
    input_token_budget, render_hunk,
)

HUNK = Hunk(
    old_start=10,
    new_start=10,
    lines=(" def handler(req):", "-    q = req.q", "+    q = sanitize(req.q)", "+    return run(q)"),
)


def test_render_hunk_omits_removed_lines():
    rendered = render_hunk(HUNK)
    assert "q = req.q" not in rendered
    assert "sanitize(req.q)" in rendered


def test_render_hunk_numbers_lines_from_new_start():
    rendered = render_hunk(HUNK)
    lines = rendered.splitlines()
    assert lines[0].strip().startswith("10")
    assert "11 +" in lines[1]
    assert "12 +" in lines[2]


def test_render_hunk_marks_added_lines():
    rendered = render_hunk(HUNK)
    added = [line for line in rendered.splitlines() if " + " in line]
    assert len(added) == 2


def test_build_file_prompt_includes_path_and_hunk():
    prompt = build_file_prompt("src/api.py", [HUNK])
    assert "src/api.py" in prompt
    assert "sanitize(req.q)" in prompt


def test_build_file_prompt_omits_context_block_when_empty():
    prompt = build_file_prompt("src/api.py", [HUNK])
    assert "FILE CONTEXT" not in prompt


def test_build_file_prompt_includes_context_block_when_given():
    prompt = build_file_prompt("src/api.py", [HUNK], context="def run(q): ...")
    assert "FILE CONTEXT" in prompt
    assert "def run(q): ..." in prompt


def test_system_prompt_keeps_ban_rules():
    assert "NEVER complain about" in SYSTEM_PROMPT
    assert "[LGTM]" in SYSTEM_PROMPT
    assert "[BLOCKER]" in SYSTEM_PROMPT


def test_estimate_tokens_is_never_zero():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcdef") == 2


def test_input_token_budget_leaves_room_for_output(monkeypatch):
    budget = input_token_budget()
    assert budget > 0
    assert budget < 8192


def test_fits_rejects_oversized_prompt():
    assert fits("x") is True
    assert fits("x" * 10_000_000) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reviewer.prompt'`

- [ ] **Step 3: Write the prompt module**

Create `reviewer/prompt.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prompt.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add reviewer/prompt.py tests/test_prompt.py
git commit -m "feat: per-file prompt builder that never sends removed lines

Replaces dead annotate_diff_for_ai (B5). Omitting '-' lines is cheaper
than annotating them and removes the hallucination risk entirely."
```

---

## Task 5: Streaming Ollama client with wall-clock abort

**Files:**
- Create: `reviewer/ollama_client.py`
- Test: `tests/test_ollama_client.py`

**Interfaces:**
- Consumes: `reviewer.config` (Task 1), `reviewer.memes.meme_phrases` (Task 1).
- Produces:
  - `ChatResult` frozen dataclass with fields `text: str`, `done_reason: str`, `prompt_eval_count: int`, `eval_count: int`, `elapsed_s: float`, and property `truncated: bool`.
  - `chat(system: str, user: str, deadline_s: int) -> ChatResult`
  - `clean_response(text: str) -> str`

`truncated` is `done_reason in ("length", "timeout")`. This replaces defect B2: the old `eval_count < 30` heuristic classified every short `[LGTM]` as truncated and triggered a second full inference.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ollama_client.py`:

```python
import json

import pytest

from reviewer.ollama_client import ChatResult, chat, clean_response


class FakeResponse:
    def __init__(self, payloads, status=200):
        self._payloads = payloads
        self.status_code = status
        self.closed = False

    @property
    def ok(self):
        return self.status_code == 200

    def iter_lines(self):
        for payload in self._payloads:
            yield json.dumps(payload).encode()

    def close(self):
        self.closed = True

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")

    @property
    def text(self):
        return "error body"


def _stream(chunks, done_reason="stop", eval_count=120):
    payloads = [{"message": {"content": c}, "done": False} for c in chunks]
    payloads.append({
        "message": {"content": ""},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 900,
        "eval_count": eval_count,
    })
    return payloads


def test_streamed_chunks_are_concatenated(monkeypatch):
    monkeypatch.setattr(
        "reviewer.ollama_client.requests.post",
        lambda *a, **kw: FakeResponse(_stream(["Hello ", "world"])),
    )
    result = chat("sys", "user", deadline_s=30)
    assert result.text == "Hello world"
    assert result.eval_count == 120
    assert result.prompt_eval_count == 900


def test_short_lgtm_is_not_treated_as_truncated(monkeypatch):
    # Regression for B2: eval_count of 8 must not trigger a retry.
    monkeypatch.setattr(
        "reviewer.ollama_client.requests.post",
        lambda *a, **kw: FakeResponse(_stream(["[LGTM]"], eval_count=8)),
    )
    result = chat("sys", "user", deadline_s=30)
    assert result.truncated is False


def test_length_done_reason_is_truncated(monkeypatch):
    monkeypatch.setattr(
        "reviewer.ollama_client.requests.post",
        lambda *a, **kw: FakeResponse(_stream(["partial"], done_reason="length")),
    )
    result = chat("sys", "user", deadline_s=30)
    assert result.truncated is True


def test_deadline_aborts_and_closes_connection(monkeypatch):
    clock = {"t": 0.0}

    def fake_monotonic():
        clock["t"] += 50.0
        return clock["t"]

    response = FakeResponse(_stream(["a", "b", "c", "d"]))
    monkeypatch.setattr("reviewer.ollama_client.requests.post", lambda *a, **kw: response)
    monkeypatch.setattr("reviewer.ollama_client.time.monotonic", fake_monotonic)

    result = chat("sys", "user", deadline_s=60)
    assert result.done_reason == "timeout"
    assert result.truncated is True
    assert response.closed is True


def test_num_batch_512_is_sent(monkeypatch):
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured.update(json)
        return FakeResponse(_stream(["ok"]))

    monkeypatch.setattr("reviewer.ollama_client.requests.post", fake_post)
    chat("sys", "user", deadline_s=30)
    # Regression for B1: the old hardcoded 128 slowed prompt eval 2-4x on Metal.
    assert captured["options"]["num_batch"] == 512
    assert captured["stream"] is True


def test_transport_error_returns_error_result(monkeypatch):
    def boom(*a, **kw):
        raise ConnectionError("refused")

    monkeypatch.setattr("reviewer.ollama_client.requests.post", boom)
    result = chat("sys", "user", deadline_s=30)
    assert result.done_reason == "error"
    assert "refused" in result.text


def test_clean_response_strips_harmony_tokens():
    assert clean_response("<|channel|>final<|message|>real text") == "final real text"


def test_clean_response_normalises_bare_lgtm():
    out = clean_response("[LGTM]")
    assert out == "LGTM. The changes are clean and follow best practices."


def test_clean_response_keeps_findings_alongside_lgtm():
    text = "**🔴 [BLOCKER]**\nbad\n[LGTM]"
    assert "[BLOCKER]" in clean_response(text)


def test_clean_response_swaps_the_for_meme():
    from reviewer.memes import meme_phrases
    assert clean_response("The") in meme_phrases
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ollama_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reviewer.ollama_client'`

- [ ] **Step 3: Write the client**

Create `reviewer/ollama_client.py`:

```python
import json
import logging
import random
import re
import time
from dataclasses import dataclass

import requests

from reviewer import config
from reviewer.memes import meme_phrases

HARMONY_TOKEN_RE = re.compile(r"<\|[^>]*\|?>")
FINDING_TAGS = ("[BLOCKER]", "[SUGGESTION]", "[NIT]")


@dataclass(frozen=True)
class ChatResult:
    text: str
    done_reason: str
    prompt_eval_count: int
    eval_count: int
    elapsed_s: float

    @property
    def truncated(self) -> bool:
        return self.done_reason in ("length", "timeout")

    @property
    def failed(self) -> bool:
        return self.done_reason == "error"


def clean_response(text: str) -> str:
    text = HARMONY_TOKEN_RE.sub(" ", text).strip()
    text = re.sub(r"[ \t]{2,}", " ", text)
    if text == "The":
        return random.choice(meme_phrases)
    if "[LGTM]" in text and not any(tag in text for tag in FINDING_TAGS):
        return "LGTM. The changes are clean and follow best practices."
    return text


def chat(system: str, user: str, deadline_s: int) -> ChatResult:
    """Streams a chat completion, aborting the connection at deadline_s."""
    started = time.monotonic()
    body = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "think": False,
        "stream": True,
        "keep_alive": "24h",
        "options": {
            "num_ctx": config.OLLAMA_NUM_CTX,
            "num_predict": config.OLLAMA_NUM_PREDICT,
            "num_batch": config.OLLAMA_NUM_BATCH,
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "seed": 42,
        },
    }

    chunks: list[str] = []
    done_reason = "incomplete"
    prompt_eval = 0
    eval_count = 0
    response = None

    try:
        response = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json=body,
            stream=True,
            timeout=(10, deadline_s),
        )
        if not response.ok:
            logging.error("Ollama returned %s: %s", response.status_code, response.text[:500])
        response.raise_for_status()

        for raw in response.iter_lines():
            if time.monotonic() - started > deadline_s:
                logging.warning("Aborting Ollama stream after %ss deadline", deadline_s)
                done_reason = "timeout"
                break
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            chunks.append(obj.get("message", {}).get("content", "") or "")
            if obj.get("done"):
                done_reason = obj.get("done_reason") or "stop"
                prompt_eval = obj.get("prompt_eval_count") or 0
                eval_count = obj.get("eval_count") or 0
                break

    except Exception as exc:
        logging.error("Error communicating with Ollama: %s", exc)
        return ChatResult(
            text=f"Error communicating with AI Reviewer: {exc}",
            done_reason="error",
            prompt_eval_count=0,
            eval_count=0,
            elapsed_s=time.monotonic() - started,
        )
    finally:
        if response is not None:
            response.close()

    elapsed = time.monotonic() - started
    logging.info(
        "Ollama done in %.1fs (reason=%s prompt_eval=%s eval=%s)",
        elapsed, done_reason, prompt_eval, eval_count,
    )
    return ChatResult(
        text=clean_response("".join(chunks)),
        done_reason=done_reason,
        prompt_eval_count=prompt_eval,
        eval_count=eval_count,
        elapsed_s=elapsed,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ollama_client.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add reviewer/ollama_client.py tests/test_ollama_client.py
git commit -m "feat: streaming Ollama client with deadline abort

Fixes B1 (num_batch 128 -> 512) and B2 (eval_count heuristic replaced
by Ollama's own done_reason, so short LGTM no longer re-runs inference)."
```

---

## Task 6: GitLab client

**Files:**
- Create: `reviewer/gitlab_client.py`
- Test: `tests/test_gitlab_client.py`

**Interfaces:**
- Consumes: `reviewer.config` (Task 1), `reviewer.diff_parser.file_diff_from_change` (Task 2).
- Produces:
  - `get_client() -> gitlab.Gitlab | None`
  - `fetch_mr(project_id: int, mr_iid: int) -> tuple[object, object]` returning `(project, mr)`
  - `fetch_file_diffs(mr) -> list[FileDiff]`
  - `fetch_file_content(project, path: str, ref: str) -> str` returning `""` on failure
  - `surgical_context(full_text: str, hunks, window: int) -> str`
  - `post_inline(mr, path: str, new_line: int, body: str) -> bool` returning `False` when GitLab rejects the position
  - `post_note(mr, body: str) -> None`
  - `diff_fingerprint(file_diffs) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gitlab_client.py`:

```python
import pytest

from reviewer.diff_parser import FileDiff, Hunk
from reviewer.gitlab_client import (
    diff_fingerprint, fetch_file_content, post_inline, post_note, surgical_context,
)

HUNK = Hunk(old_start=1, new_start=5, lines=(" a", "+b"))
FD = FileDiff("x.py", "x.py", False, False, False, False, (HUNK,))


class FakeDiscussions:
    def __init__(self, fail=False):
        self.created = []
        self.fail = fail

    def create(self, payload):
        if self.fail:
            raise RuntimeError("400 Bad Request")
        self.created.append(payload)


class FakeNotes:
    def __init__(self):
        self.created = []

    def create(self, payload):
        self.created.append(payload)


class FakeMR:
    def __init__(self, fail_inline=False):
        self.discussions = FakeDiscussions(fail=fail_inline)
        self.notes = FakeNotes()
        self.diff_refs = {"base_sha": "b1", "start_sha": "s1", "head_sha": "h1"}


def test_post_inline_builds_position():
    mr = FakeMR()
    assert post_inline(mr, "src/a.py", 42, "body") is True
    payload = mr.discussions.created[0]
    assert payload["body"] == "body"
    assert payload["position"]["new_line"] == 42
    assert payload["position"]["new_path"] == "src/a.py"
    assert payload["position"]["position_type"] == "text"
    assert payload["position"]["head_sha"] == "h1"


def test_post_inline_returns_false_when_gitlab_rejects():
    mr = FakeMR(fail_inline=True)
    assert post_inline(mr, "src/a.py", 42, "body") is False


def test_post_inline_returns_false_without_diff_refs():
    mr = FakeMR()
    mr.diff_refs = None
    assert post_inline(mr, "src/a.py", 42, "body") is False


def test_post_note_creates_note():
    mr = FakeMR()
    post_note(mr, "summary")
    assert mr.notes.created == [{"body": "summary"}]


def test_surgical_context_extracts_window_around_hunk():
    text = "\n".join(f"line{i}" for i in range(1, 21))
    out = surgical_context(text, [HUNK], window=2)
    assert "line5" in out
    assert "line20" not in out


def test_surgical_context_handles_hunk_past_end_of_file():
    out = surgical_context("only one line", [Hunk(1, 999, (" a",))], window=3)
    assert isinstance(out, str)


def test_surgical_context_merges_overlapping_windows():
    text = "\n".join(f"line{i}" for i in range(1, 21))
    hunks = [Hunk(1, 5, (" a",)), Hunk(1, 7, (" b",))]
    out = surgical_context(text, hunks, window=5)
    assert out.count("line5") == 1


def test_fetch_file_content_returns_empty_on_error():
    class Boom:
        class files:
            @staticmethod
            def get(**kwargs):
                raise RuntimeError("404")

    assert fetch_file_content(Boom, "a.py", "main") == ""


def test_diff_fingerprint_is_stable_and_content_sensitive():
    a = diff_fingerprint([FD])
    b = diff_fingerprint([FD])
    other = FileDiff("x.py", "x.py", False, False, False, False,
                     (Hunk(1, 5, (" a", "+c")),))
    assert a == b
    assert diff_fingerprint([other]) != a
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gitlab_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reviewer.gitlab_client'`

- [ ] **Step 3: Write the client**

Create `reviewer/gitlab_client.py`:

```python
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
    try:
        mr.discussions.create({
            "body": body,
            "position": {
                "base_sha": refs["base_sha"],
                "start_sha": refs["start_sha"],
                "head_sha": refs["head_sha"],
                "position_type": "text",
                "new_path": path,
                "old_path": path,
                "new_line": new_line,
            },
        })
        return True
    except Exception as exc:
        logging.info("Inline discussion rejected for %s:%s (%s)", path, new_line, exc)
        return False


def post_note(mr, body: str) -> None:
    mr.notes.create({"body": body})


def diff_fingerprint(file_diffs: Sequence[FileDiff]) -> str:
    """Stable hash of all diff content, used for dedupe."""
    digest = hashlib.sha256()
    for fd in file_diffs:
        digest.update(fd.new_path.encode())
        for hunk in fd.hunks:
            digest.update(str(hunk.new_start).encode())
            for line in hunk.lines:
                digest.update(line.encode())
    return digest.hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gitlab_client.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add reviewer/gitlab_client.py tests/test_gitlab_client.py
git commit -m "feat: GitLab client with inline discussions and diff fingerprint"
```

---

## Task 7: Review pipeline with degradation ladder

**Files:**
- Create: `reviewer/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces:
  - `FileOutcome` frozen dataclass with fields `path: str`, `status: str`, `detail: str`. `status` is one of `"reviewed"`, `"clean"`, `"skipped"`, `"error"`.
  - `select_files(file_diffs) -> tuple[list[FileDiff], list[FileOutcome]]` — applies the filter and `MAX_FILES` cap, returning kept files ordered smallest-first plus outcomes for rejects.
  - `build_prompt_ladder(path, hunks, context) -> list[tuple[str, str]]` — returns `[(level_name, prompt), ...]` in the order they should be attempted.
  - `review_file(mr, file_diff, context) -> FileOutcome`
  - `render_summary(outcomes) -> str`
  - `review_merge_request(project_id, mr_iid) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
import pytest

from reviewer.diff_parser import FileDiff, Hunk
from reviewer.pipeline import (
    FileOutcome, build_prompt_ladder, render_summary, select_files,
)


def fd(path, added=1, **kwargs):
    lines = [" ctx"] + [f"+line{i}" for i in range(added)]
    return FileDiff(
        old_path=path, new_path=path,
        is_new=False,
        is_deleted=kwargs.get("is_deleted", False),
        is_renamed=False,
        is_binary=kwargs.get("is_binary", False),
        hunks=kwargs.get("hunks", (Hunk(1, 1, tuple(lines)),)),
    )


def test_select_files_orders_smallest_first():
    kept, _ = select_files([fd("big.py", added=50), fd("small.py", added=2)])
    assert [f.new_path for f in kept] == ["small.py", "big.py"]


def test_select_files_reports_filtered_files():
    kept, outcomes = select_files([fd("src/a.py"), fd("package-lock.json")])
    assert [f.new_path for f in kept] == ["src/a.py"]
    assert outcomes[0].path == "package-lock.json"
    assert outcomes[0].status == "skipped"
    assert outcomes[0].detail == "lockfile"


def test_select_files_caps_at_max_files(monkeypatch):
    monkeypatch.setattr("reviewer.config.MAX_FILES", 2)
    kept, outcomes = select_files([fd(f"f{i}.py", added=i + 1) for i in range(5)])
    assert len(kept) == 2
    over = [o for o in outcomes if o.detail == "over MAX_FILES limit"]
    assert len(over) == 3


def test_ladder_starts_at_l1_when_context_disabled(monkeypatch):
    monkeypatch.setattr("reviewer.config.INCLUDE_FILE_CONTEXT", False)
    ladder = build_prompt_ladder("a.py", [Hunk(1, 1, (" a", "+b"))], context="ctx text")
    assert ladder[0][0] == "L1"
    assert "ctx text" not in ladder[0][1]


def test_ladder_starts_at_l0_when_context_enabled(monkeypatch):
    monkeypatch.setattr("reviewer.config.INCLUDE_FILE_CONTEXT", True)
    ladder = build_prompt_ladder("a.py", [Hunk(1, 1, (" a", "+b"))], context="ctx text")
    assert ladder[0][0] == "L0"
    assert "ctx text" in ladder[0][1]


def test_ladder_adds_per_hunk_level_for_multi_hunk_files(monkeypatch):
    monkeypatch.setattr("reviewer.config.INCLUDE_FILE_CONTEXT", False)
    hunks = [Hunk(1, 1, (" a", "+b")), Hunk(1, 40, (" c", "+d"))]
    ladder = build_prompt_ladder("a.py", hunks, context="")
    assert [level for level, _ in ladder] == ["L1", "L2", "L2"]


def test_ladder_has_no_l2_for_single_hunk_files(monkeypatch):
    monkeypatch.setattr("reviewer.config.INCLUDE_FILE_CONTEXT", False)
    ladder = build_prompt_ladder("a.py", [Hunk(1, 1, (" a", "+b"))], context="")
    assert [level for level, _ in ladder] == ["L1"]


def test_render_summary_lists_every_category():
    summary = render_summary([
        FileOutcome("a.py", "reviewed", "2 findings"),
        FileOutcome("b.py", "clean", ""),
        FileOutcome("huge.py", "skipped", "single hunk exceeds context budget"),
        FileOutcome("c.py", "error", "timeout"),
    ])
    assert "a.py" in summary
    assert "huge.py" in summary
    assert "single hunk exceeds context budget" in summary
    assert "c.py" in summary
    assert "timeout" in summary


def test_render_summary_of_all_clean_says_lgtm():
    summary = render_summary([FileOutcome("a.py", "clean", "")])
    assert "LGTM" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reviewer.pipeline'`

- [ ] **Step 3: Write the pipeline**

Create `reviewer/pipeline.py`:

```python
import logging
import time
from dataclasses import dataclass
from typing import Sequence

from reviewer import config, gitlab_client, prompt as prompt_mod
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.filters import is_reviewable
from reviewer.ollama_client import chat


@dataclass(frozen=True)
class FileOutcome:
    path: str
    status: str   # "reviewed" | "clean" | "skipped" | "error"
    detail: str


def select_files(file_diffs: Sequence[FileDiff]) -> tuple[list[FileDiff], list[FileOutcome]]:
    kept: list[FileDiff] = []
    outcomes: list[FileOutcome] = []

    for fd in file_diffs:
        ok, reason = is_reviewable(fd)
        if ok:
            kept.append(fd)
        else:
            outcomes.append(FileOutcome(fd.new_path, "skipped", reason))

    kept.sort(key=lambda f: f.total_lines)
    if len(kept) > config.MAX_FILES:
        for fd in kept[config.MAX_FILES:]:
            outcomes.append(FileOutcome(fd.new_path, "skipped", "over MAX_FILES limit"))
        kept = kept[: config.MAX_FILES]

    return kept, outcomes


def build_prompt_ladder(path: str, hunks: Sequence[Hunk], context: str) -> list[tuple[str, str]]:
    """Ordered attempts, cheapest-viable first. L3 is the absence of any fitting level."""
    ladder: list[tuple[str, str]] = []
    if config.INCLUDE_FILE_CONTEXT and context:
        ladder.append(("L0", prompt_mod.build_file_prompt(path, hunks, context)))
    ladder.append(("L1", prompt_mod.build_file_prompt(path, hunks)))
    if len(hunks) > 1:
        for hunk in hunks:
            ladder.append(("L2", prompt_mod.build_file_prompt(path, [hunk])))
    return ladder


def review_file(mr, file_diff: FileDiff, context: str) -> FileOutcome:
    path = file_diff.new_path
    ladder = build_prompt_ladder(path, file_diff.hunks, context)
    attempts = [(level, text) for level, text in ladder if prompt_mod.fits(text)]

    if not attempts:
        return FileOutcome(path, "skipped", "single hunk exceeds context budget")

    # Take the first fitting level; if it is L2, take every L2 entry that fits.
    chosen_level = attempts[0][0]
    prompts = [text for level, text in attempts if level == chosen_level]

    bodies: list[str] = []
    for text in prompts:
        result = chat(prompt_mod.SYSTEM_PROMPT, text, deadline_s=config.PER_FILE_TIMEOUT_S)
        if result.failed:
            return FileOutcome(path, "error", result.done_reason)
        if result.done_reason == "timeout":
            return FileOutcome(path, "error", f"timeout after {config.PER_FILE_TIMEOUT_S}s")
        if result.text and not result.text.startswith("LGTM."):
            bodies.append(result.text)

    if not bodies:
        return FileOutcome(path, "clean", "")

    body = f"### 📄 `{path}`\n\n" + "\n\n".join(bodies)
    anchor = file_diff.hunks[0].first_added_line()
    posted = False
    if anchor is not None:
        posted = gitlab_client.post_inline(mr, path, anchor, body)
    if not posted:
        gitlab_client.post_note(mr, body)

    return FileOutcome(path, "reviewed", f"{len(bodies)} response(s)")


def render_summary(outcomes: Sequence[FileOutcome]) -> str:
    reviewed = [o for o in outcomes if o.status == "reviewed"]
    clean = [o for o in outcomes if o.status == "clean"]
    skipped = [o for o in outcomes if o.status == "skipped"]
    errored = [o for o in outcomes if o.status == "error"]

    lines = []

    if reviewed:
        lines.append(f"\n**Findings on {len(reviewed)} file(s):** "
                     + ", ".join(f"`{o.path}`" for o in reviewed))
    if clean and not reviewed and not errored:
        lines.append("\nLGTM. No logic or security issues found in the changed lines.")
    elif clean:
        lines.append(f"\n**Clean:** {len(clean)} file(s)")
    if skipped:
        lines.append("\n**Skipped:**")
        lines.extend(f"- `{o.path}` — {o.detail}" for o in skipped)
    if errored:
        lines.append("\n**Errors:**")
        lines.extend(f"- `{o.path}` — {o.detail}" for o in errored)

    return "\n".join(lines)


def review_merge_request(project_id: int, mr_iid: int) -> None:
    started = time.monotonic()
    try:
        project, mr = gitlab_client.fetch_mr(project_id, mr_iid)
        file_diffs = gitlab_client.fetch_file_diffs(mr)
        kept, outcomes = select_files(file_diffs)

        if not kept:
            logging.info("MR !%s: nothing reviewable", mr_iid)
            gitlab_client.post_note(mr, render_summary(outcomes))
            return

        for file_diff in kept:
            if time.monotonic() - started > config.MR_TIMEOUT_S:
                outcomes.append(FileOutcome(
                    file_diff.new_path, "skipped",
                    f"MR deadline of {config.MR_TIMEOUT_S}s reached",
                ))
                continue

            context = ""
            if config.INCLUDE_FILE_CONTEXT:
                content = gitlab_client.fetch_file_content(
                    project, file_diff.new_path, mr.source_branch,
                )
                if content:
                    context = gitlab_client.surgical_context(
                        content, file_diff.hunks, config.CONTEXT_WINDOW,
                    )

            try:
                outcomes.append(review_file(mr, file_diff, context))
            except Exception as exc:
                logging.error("Review failed for %s: %s", file_diff.new_path, exc)
                outcomes.append(FileOutcome(file_diff.new_path, "error", str(exc)[:120]))

        gitlab_client.post_note(mr, render_summary(outcomes))
        logging.info("MR !%s reviewed in %.1fs", mr_iid, time.monotonic() - started)

    except Exception as exc:
        logging.error("Critical error reviewing MR !%s: %s", mr_iid, exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add reviewer/pipeline.py tests/test_pipeline.py
git commit -m "feat: per-file review pipeline with degradation ladder

Every skipped file is now named in the summary note instead of
disappearing into a log line."
```

---

## Task 8: Coalescing queue with dedupe and worker

**Files:**
- Create: `reviewer/queue.py`
- Test: `tests/test_queue.py`

**Interfaces:**
- Consumes: `reviewer.config` (Task 1).
- Produces:
  - `ReviewQueue` class with methods `submit(key: str, job: tuple) -> bool`, `take() -> tuple`, `size() -> int`.
  - `DedupeCache` class with methods `seen(fingerprint: str) -> bool` and `remember(fingerprint: str) -> None`.
  - `start_worker(handler: Callable[[tuple], None]) -> ReviewQueue` — starts one daemon thread and returns the queue it drains.

`ReviewQueue` replaces `review_lock`. Coalescing keeps the oldest queue position but the newest payload, so repeated pushes to the same MR do not stack up.

- [ ] **Step 1: Write the failing test**

Create `tests/test_queue.py`:

```python
import threading
import time

from reviewer.queue import DedupeCache, ReviewQueue, start_worker


def test_take_returns_jobs_in_fifo_order():
    q = ReviewQueue(maxsize=4)
    q.submit("a", (1, 1))
    q.submit("b", (2, 2))
    assert q.take() == (1, 1)
    assert q.take() == (2, 2)


def test_submit_coalesces_same_key_keeping_newest_payload():
    q = ReviewQueue(maxsize=4)
    q.submit("mr-1", (1, "old"))
    q.submit("mr-2", (2, "other"))
    q.submit("mr-1", (1, "new"))
    assert q.size() == 2
    assert q.take() == (1, "new")   # keeps original position
    assert q.take() == (2, "other")


def test_submit_rejects_when_full():
    q = ReviewQueue(maxsize=2)
    assert q.submit("a", (1,)) is True
    assert q.submit("b", (2,)) is True
    assert q.submit("c", (3,)) is False


def test_dedupe_cache_remembers_and_evicts():
    cache = DedupeCache(maxsize=2)
    assert cache.seen("x") is False
    cache.remember("x")
    assert cache.seen("x") is True
    cache.remember("y")
    cache.remember("z")
    assert cache.seen("x") is False   # evicted
    assert cache.seen("z") is True


def test_worker_drains_queue():
    handled = []
    done = threading.Event()

    def handler(job):
        handled.append(job)
        done.set()

    q = start_worker(handler)
    q.submit("a", (7, 8))
    assert done.wait(timeout=5) is True
    assert handled == [(7, 8)]


def test_worker_survives_handler_exception():
    calls = []
    second = threading.Event()

    def handler(job):
        calls.append(job)
        if len(calls) == 1:
            raise RuntimeError("boom")
        second.set()

    q = start_worker(handler)
    q.submit("a", (1,))
    q.submit("b", (2,))
    assert second.wait(timeout=5) is True
    assert calls == [(1,), (2,)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_queue.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reviewer.queue'`

- [ ] **Step 3: Write the queue**

Create `reviewer/queue.py`:

```python
import logging
import threading
from collections import OrderedDict
from typing import Callable

from reviewer import config


class DedupeCache:
    """LRU set of diff fingerprints already reviewed."""

    def __init__(self, maxsize: int = None):
        self._maxsize = maxsize or config.DEDUPE_CACHE_SIZE
        self._entries: OrderedDict[str, bool] = OrderedDict()
        self._lock = threading.Lock()

    def seen(self, fingerprint: str) -> bool:
        with self._lock:
            if fingerprint in self._entries:
                self._entries.move_to_end(fingerprint)
                return True
            return False

    def remember(self, fingerprint: str) -> None:
        with self._lock:
            self._entries[fingerprint] = True
            self._entries.move_to_end(fingerprint)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)


class ReviewQueue:
    """FIFO queue that coalesces repeat submissions for the same key.

    Replaces the old threading.Lock, which could not serialize across
    gunicorn worker processes (defect B3).
    """

    def __init__(self, maxsize: int = None):
        self._maxsize = maxsize or config.QUEUE_MAXSIZE
        self._items: OrderedDict[str, tuple] = OrderedDict()
        self._cv = threading.Condition()

    def submit(self, key: str, job: tuple) -> bool:
        with self._cv:
            if key in self._items:
                # Keep queue position, replace payload with the newer one.
                self._items[key] = job
                logging.info("Coalesced queued job %s", key)
                return True
            if len(self._items) >= self._maxsize:
                logging.warning("Review queue full (%s); rejecting %s", self._maxsize, key)
                return False
            self._items[key] = job
            self._cv.notify()
            return True

    def take(self) -> tuple:
        with self._cv:
            while not self._items:
                self._cv.wait()
            _, job = self._items.popitem(last=False)
            return job

    def size(self) -> int:
        with self._cv:
            return len(self._items)


def start_worker(handler: Callable[[tuple], None]) -> ReviewQueue:
    """Starts one daemon consumer. Only ever one, to keep Ollama serialized."""
    queue = ReviewQueue()

    def loop() -> None:
        while True:
            job = queue.take()
            try:
                handler(job)
            except Exception as exc:
                logging.error("Worker handler failed for %r: %s", job, exc)

    threading.Thread(target=loop, name="review-worker", daemon=True).start()
    return queue
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_queue.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add reviewer/queue.py tests/test_queue.py
git commit -m "feat: coalescing review queue with LRU dedupe cache"
```

---

## Task 9: Rewire Flask entry point and fix gunicorn workers

**Files:**
- Modify: `review_server.py` (full rewrite, 510 lines → ~90)
- Modify: `Dockerfile:11-14`
- Modify: `requirements.txt`
- Test: `tests/test_webhook.py`

**Interfaces:**
- Consumes: `reviewer.pipeline.review_merge_request` (Task 7), `reviewer.queue.start_worker` / `DedupeCache` (Task 8), `reviewer.config` (Task 1).
- Produces: Flask `app` with routes `POST /webhook` and `GET /health`, plus `should_review(event_type: str, data: dict) -> tuple[int, int] | None`.

`should_review` fixes defect B7: a `Merge Request Hook` with `action == "update"` is only accepted when `object_attributes.oldrev` is present, which GitLab sets only when new commits arrived. Title, description, and label edits are ignored.

- [ ] **Step 1: Write the failing test**

Create `tests/test_webhook.py`:

```python
import pytest

import review_server
from review_server import app, should_review


@pytest.fixture
def client(monkeypatch):
    # review_server reads config.WEBHOOK_SECRET at request time, so patching
    # the config module attribute is sufficient.
    monkeypatch.setattr("reviewer.config.WEBHOOK_SECRET", "s3cret")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _mr_hook(action, oldrev=None):
    attrs = {"action": action, "iid": 7}
    if oldrev:
        attrs["oldrev"] = oldrev
    return {"project": {"id": 3}, "object_attributes": attrs}


def test_open_action_is_reviewed():
    assert should_review("Merge Request Hook", _mr_hook("open")) == (3, 7)


def test_reopen_action_is_reviewed():
    assert should_review("Merge Request Hook", _mr_hook("reopen")) == (3, 7)


def test_update_with_new_commits_is_reviewed():
    assert should_review("Merge Request Hook", _mr_hook("update", oldrev="abc123")) == (3, 7)


def test_update_without_oldrev_is_ignored():
    # Regression for B7: title/description/label edits carry no oldrev.
    assert should_review("Merge Request Hook", _mr_hook("update")) is None


def test_merge_action_is_ignored():
    assert should_review("Merge Request Hook", _mr_hook("merge")) is None


def test_review_comment_triggers_review():
    data = {
        "project": {"id": 3},
        "merge_request": {"iid": 7},
        "object_attributes": {"noteable_type": "MergeRequest", "note": "please /review this"},
    }
    assert should_review("Note Hook", data) == (3, 7)


def test_unrelated_comment_is_ignored():
    data = {
        "project": {"id": 3},
        "merge_request": {"iid": 7},
        "object_attributes": {"noteable_type": "MergeRequest", "note": "nice work"},
    }
    assert should_review("Note Hook", data) is None


def test_note_on_issue_is_ignored():
    data = {
        "project": {"id": 3},
        "object_attributes": {"noteable_type": "Issue", "note": "/review"},
    }
    assert should_review("Note Hook", data) is None


def test_webhook_rejects_bad_token(client):
    resp = client.post("/webhook", json=_mr_hook("open"),
                       headers={"X-Gitlab-Token": "wrong", "X-Gitlab-Event": "Merge Request Hook"})
    assert resp.status_code == 403


def test_webhook_enqueues_and_returns_202(client, monkeypatch):
    submitted = []
    monkeypatch.setattr(review_server.review_queue, "submit",
                        lambda key, job: submitted.append((key, job)) or True)
    resp = client.post("/webhook", json=_mr_hook("open"),
                       headers={"X-Gitlab-Token": "s3cret", "X-Gitlab-Event": "Merge Request Hook"})
    assert resp.status_code == 202
    assert submitted[0][1] == (3, 7)


def test_webhook_returns_503_when_queue_full(client, monkeypatch):
    monkeypatch.setattr(review_server.review_queue, "submit", lambda key, job: False)
    resp = client.post("/webhook", json=_mr_hook("open"),
                       headers={"X-Gitlab-Token": "s3cret", "X-Gitlab-Event": "Merge Request Hook"})
    assert resp.status_code == 503


def test_health_endpoint(client):
    assert client.get("/health").status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_webhook.py -v`
Expected: FAIL with `ImportError: cannot import name 'should_review' from 'review_server'`

- [ ] **Step 3: Rewrite the entry point**

Replace the entire contents of `review_server.py`:

```python
import logging

from flask import Flask, jsonify, request

from reviewer import config
from reviewer.pipeline import review_merge_request
from reviewer.queue import start_worker

config.configure_logging()

app = Flask(__name__)

logging.info(
    "Ollama config: host=%s model=%s num_ctx=%s num_predict=%s num_batch=%s "
    "include_context=%s per_file_timeout=%ss mr_timeout=%ss max_files=%s",
    config.OLLAMA_HOST, config.OLLAMA_MODEL, config.OLLAMA_NUM_CTX,
    config.OLLAMA_NUM_PREDICT, config.OLLAMA_NUM_BATCH, config.INCLUDE_FILE_CONTEXT,
    config.PER_FILE_TIMEOUT_S, config.MR_TIMEOUT_S, config.MAX_FILES,
)


def _handle(job: tuple) -> None:
    project_id, mr_iid = job
    review_merge_request(project_id, mr_iid)


review_queue = start_worker(_handle)


def should_review(event_type: str, data: dict) -> tuple[int, int] | None:
    """Returns (project_id, mr_iid) if this event warrants a review, else None."""
    attrs = data.get("object_attributes", {})

    if event_type == "Note Hook":
        if attrs.get("noteable_type") != "MergeRequest":
            return None
        if "/review" not in (attrs.get("note") or "").lower():
            return None
        return data["project"]["id"], data["merge_request"]["iid"]

    if event_type == "Merge Request Hook":
        action = attrs.get("action")
        if action in ("open", "reopen"):
            return data["project"]["id"], attrs["iid"]
        # 'update' fires on title, description and label edits too. GitLab sets
        # oldrev only when new commits arrived, so it is our new-commits signal.
        if action == "update" and attrs.get("oldrev"):
            return data["project"]["id"], attrs["iid"]
        return None

    return None


@app.route("/webhook", methods=["POST"])
def webhook():
    token = request.headers.get("X-Gitlab-Token")
    if config.WEBHOOK_SECRET and token != config.WEBHOOK_SECRET:
        return jsonify({"error": "Invalid token"}), 403

    target = should_review(request.headers.get("X-Gitlab-Event"), request.json or {})
    if not target:
        return jsonify({"message": "Ignored event"}), 200

    project_id, mr_iid = target
    if review_queue.submit(f"{project_id}:{mr_iid}", (project_id, mr_iid)):
        return jsonify({"message": "Review queued", "depth": review_queue.size()}), 202
    return jsonify({"error": "Review queue full"}), 503


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "queue_depth": review_queue.size()}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

Note: the release-branch check from the old code is intentionally dropped here and re-added in Task 10, where it can run inside the worker without an extra blocking GitLab call on the webhook path.

- [ ] **Step 4: Fix the gunicorn worker count**

Replace `Dockerfile:11-14` with:

```dockerfile
# One worker only. The review queue and dedupe cache live in process memory;
# forking a second worker would allow two concurrent Ollama inferences and
# exhaust unified memory on a 16 GB host (defect B3).
# Threads handle webhook concurrency; actual reviews are serialized by the queue.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", \
     "--workers", "1", "--threads", "4", "review_server:app"]
```

- [ ] **Step 5: Drop the unused dependency**

Edit `requirements.txt` — remove the trailing `ollama` line, leaving:

```
flask==3.0.0
python-gitlab==4.4.0
requests==2.31.0
gunicorn==21.2.0
```

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -v`
Expected: all tests pass, 12 of them from `tests/test_webhook.py`

- [ ] **Step 7: Commit**

```bash
git add review_server.py Dockerfile requirements.txt tests/test_webhook.py
git commit -m "refactor: thin Flask entry point, queue-backed reviews

Fixes B3 (gunicorn --workers 2 defeated the in-process review lock) and
B7 ('update' events on title/label edits triggered full re-reviews)."
```

---

## Task 10: Dedupe wiring, release-branch skip, and operator docs

**Files:**
- Modify: `reviewer/pipeline.py`
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `README.md`
- Test: `tests/test_pipeline_dedupe.py`

**Interfaces:**
- Consumes: `reviewer.gitlab_client.diff_fingerprint` (Task 6), `reviewer.queue.DedupeCache` (Task 8).
- Produces: `reviewer.pipeline.should_skip_branch(branch: str) -> bool` and a module-level `dedupe = DedupeCache()` consulted inside `review_merge_request`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_dedupe.py`:

```python
import reviewer.pipeline as pipeline
from reviewer.diff_parser import FileDiff, Hunk

FD = FileDiff("a.py", "a.py", False, False, False, False, (Hunk(1, 1, (" x", "+y")),))


class FakeMR:
    def __init__(self, branch="feature/x"):
        self.source_branch = branch
        self.diff_refs = {"base_sha": "b", "start_sha": "s", "head_sha": "h"}
        self.notes_posted = []

    def changes(self):
        return {"changes": []}


def test_release_branches_are_skipped():
    assert pipeline.should_skip_branch("release/2026.08") is True
    assert pipeline.should_skip_branch("feature/x") is False
    assert pipeline.should_skip_branch("") is False


def test_identical_diff_is_reviewed_once(monkeypatch):
    calls = []
    mr = FakeMR()

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "fetch_file_diffs", lambda m: [FD])
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: calls.append(body))
    monkeypatch.setattr(pipeline, "review_file",
                        lambda mr_, fd, ctx: pipeline.FileOutcome(fd.new_path, "clean", ""))
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    assert len(calls) == 1

    pipeline.review_merge_request(1, 1)
    assert len(calls) == 1   # second identical diff costs nothing


def test_changed_diff_is_reviewed_again(monkeypatch):
    calls = []
    mr = FakeMR()
    diffs = [FD]

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "fetch_file_diffs", lambda m: diffs)
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: calls.append(body))
    monkeypatch.setattr(pipeline, "review_file",
                        lambda mr_, fd, ctx: pipeline.FileOutcome(fd.new_path, "clean", ""))
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    diffs[0] = FileDiff("a.py", "a.py", False, False, False, False,
                        (Hunk(1, 1, (" x", "+z")),))
    pipeline.review_merge_request(1, 1)
    assert len(calls) == 2


def test_release_branch_mr_posts_nothing(monkeypatch):
    calls = []
    mr = FakeMR(branch="release/2026.08")

    monkeypatch.setattr(pipeline.gitlab_client, "fetch_mr", lambda p, i: (object(), mr))
    monkeypatch.setattr(pipeline.gitlab_client, "fetch_file_diffs", lambda m: [FD])
    monkeypatch.setattr(pipeline.gitlab_client, "post_note",
                        lambda m, body: calls.append(body))
    monkeypatch.setattr(pipeline, "dedupe", pipeline.DedupeCache(maxsize=8))

    pipeline.review_merge_request(1, 1)
    assert calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_dedupe.py -v`
Expected: FAIL with `AttributeError: module 'reviewer.pipeline' has no attribute 'should_skip_branch'`

- [ ] **Step 3: Add dedupe and branch skip to the pipeline**

In `reviewer/pipeline.py`, add to the imports:

```python
from reviewer.queue import DedupeCache
```

Add below the `FileOutcome` dataclass:

```python
dedupe = DedupeCache()

SKIP_BRANCH_PREFIXES = ("release/",)


def should_skip_branch(branch: str) -> bool:
    return any(prefix in (branch or "") for prefix in SKIP_BRANCH_PREFIXES)
```

Replace the opening of `review_merge_request` — everything from `project, mr = gitlab_client.fetch_mr(...)` down to and including the `if not kept:` block — with:

```python
        project, mr = gitlab_client.fetch_mr(project_id, mr_iid)

        if should_skip_branch(getattr(mr, "source_branch", "")):
            logging.info("MR !%s targets a release branch; skipping", mr_iid)
            return

        file_diffs = gitlab_client.fetch_file_diffs(mr)

        fingerprint = gitlab_client.diff_fingerprint(file_diffs)
        if dedupe.seen(fingerprint):
            logging.info("MR !%s diff unchanged since last review; skipping", mr_iid)
            return
        dedupe.remember(fingerprint)

        kept, outcomes = select_files(file_diffs)

        if not kept:
            logging.info("MR !%s: nothing reviewable", mr_iid)
            gitlab_client.post_note(mr, render_summary(outcomes))
            return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline_dedupe.py -v`
Expected: 4 passed

- [ ] **Step 5: Update `.env.example`**

Replace the Ollama and context sections of `.env.example` with:

```dotenv
# The Ollama model to use.
# 16 GB RAM: prefer qwen2.5-coder:7b (~5 GB) or a 12B QAT build (~7 GB).
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_HOST=http://host.docker.internal:11434

# Context and generation limits. Keep num_ctx low on 16 GB to avoid swap thrash.
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=320
# Metal default. Do not lower this — 128 slows prompt eval 2-4x.
OLLAMA_NUM_BATCH=512

# Surrounding-file context. Off by default: it roughly doubles input tokens
# and adds one GitLab file fetch per file, for marginal review quality gain.
INCLUDE_FILE_CONTEXT=false
CONTEXT_WINDOW=15

# Deadlines and limits.
PER_FILE_TIMEOUT_S=90
MR_TIMEOUT_S=480
MAX_FILES=40
QUEUE_MAXSIZE=32
```

- [ ] **Step 6: Update `docker-compose.yml`**

Replace the `environment:` block of the `app` service with:

```yaml
    environment:
      - GITLAB_URL=${GITLAB_URL:-https://gitlab.com}
      - GITLAB_TOKEN=${GITLAB_TOKEN}
      - WEBHOOK_SECRET=${WEBHOOK_SECRET}
      - OLLAMA_HOST=http://host.docker.internal:11434
      - OLLAMA_MODEL=${OLLAMA_MODEL:-qwen2.5-coder:7b}
      - OLLAMA_NUM_CTX=${OLLAMA_NUM_CTX:-8192}
      - OLLAMA_NUM_PREDICT=${OLLAMA_NUM_PREDICT:-320}
      - OLLAMA_NUM_BATCH=${OLLAMA_NUM_BATCH:-512}
      - INCLUDE_FILE_CONTEXT=${INCLUDE_FILE_CONTEXT:-false}
      - CONTEXT_WINDOW=${CONTEXT_WINDOW:-15}
      - PER_FILE_TIMEOUT_S=${PER_FILE_TIMEOUT_S:-90}
      - MR_TIMEOUT_S=${MR_TIMEOUT_S:-480}
      - MAX_FILES=${MAX_FILES:-40}
      - QUEUE_MAXSIZE=${QUEUE_MAXSIZE:-32}
```

- [ ] **Step 7: Update `README.md`**

In the "Tuning for Mac Mini M4 16 GB" section, replace the `.env` snippet with the values from Step 5. Add this subsection immediately after it:

```markdown
### How reviews are scheduled

Webhooks return immediately after enqueuing. One background worker drains the
queue, so only one Ollama request is ever in flight. Repeat webhooks for the
same MR coalesce into the single queued job, and an MR whose diff has not
changed since its last review is skipped entirely.

`GET /health` reports `queue_depth`, which is the fastest way to tell whether
the bot is busy or stuck.

Reviews run one file at a time. Each file gets its own request with a
`PER_FILE_TIMEOUT_S` deadline, and the whole MR is bounded by `MR_TIMEOUT_S`.
Files that cannot fit the context budget are named in the summary note rather
than dropped silently.
```

Replace the "Review stuck / never finishes" troubleshooting bullet with:

```markdown
*   **Review stuck / never finishes:** check `curl localhost:5000/health` for
    `queue_depth`. A depth above zero with no log progress means Ollama is
    wedged — run `ollama ps` and confirm `PROCESSOR=100% GPU`. Individual files
    now abort after `PER_FILE_TIMEOUT_S` instead of hanging.
```

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest -v`
Expected: all tests pass across all 8 test files

- [ ] **Step 9: Commit**

```bash
git add reviewer/pipeline.py .env.example docker-compose.yml README.md \
        tests/test_pipeline_dedupe.py
git commit -m "feat: skip unchanged diffs and release branches, update docs"
```

---

## Verification

After Task 10, confirm all seven defects are closed:

| Defect | Verified by |
|---|---|
| B1 `num_batch: 128` | `tests/test_ollama_client.py::test_num_batch_512_is_sent` |
| B2 double inference on LGTM | `tests/test_ollama_client.py::test_short_lgtm_is_not_treated_as_truncated` |
| B3 `--workers 2` breaks lock | `Dockerfile` sets `--workers 1`; queue is in-process |
| B4 single-line hunk regex | `tests/test_diff_parser.py::test_single_line_old_range_parses` |
| B5 dead `annotate_diff_for_ai` | Function removed; `tests/test_prompt.py::test_render_hunk_omits_removed_lines` |
| B6 dead `MAX_PROMPT_CHARS` | Variable removed from `config.py`, `.env.example`, `docker-compose.yml` |
| B7 `update` re-reviews | `tests/test_webhook.py::test_update_without_oldrev_is_ignored` |

Run once more: `python -m pytest -v`

Then confirm nothing references the old module layout:

```bash
grep -rn "MAX_PROMPT_CHARS\|MIN_OUTPUT_TOKENS\|annotate_diff_for_ai\|review_lock" \
  --include="*.py" --include="*.yml" --include="*.md" .
```

Expected: matches only inside `docs/superpowers/`.

---

## Operator Runbook — Mac Mini M4 16 GB

### 1. Edit `.env` on the Mac Mini

Open `~/Dev/ai-code-reviewer/.env` (or wherever the repo lives on the Mini) and set:

```dotenv
GITLAB_URL=https://gitlab.com
GITLAB_TOKEN=<unchanged>
WEBHOOK_SECRET=<unchanged>
TUNNEL_TOKEN=<unchanged>

OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=320
OLLAMA_NUM_BATCH=512

INCLUDE_FILE_CONTEXT=false
CONTEXT_WINDOW=15

PER_FILE_TIMEOUT_S=90
MR_TIMEOUT_S=480
MAX_FILES=40
QUEUE_MAXSIZE=32
```

**Remove these lines if present** — they no longer do anything:

```dotenv
MAX_PROMPT_CHARS=120000
PROMPT_TOKEN_BUFFER=128
MIN_OUTPUT_TOKENS=30
```

Changes from the previous values: `OLLAMA_NUM_PREDICT` drops from 1024 to 320,
`CONTEXT_WINDOW` drops from 25 to 15, and `OLLAMA_NUM_BATCH`,
`INCLUDE_FILE_CONTEXT`, `PER_FILE_TIMEOUT_S`, `MR_TIMEOUT_S`, `MAX_FILES`, and
`QUEUE_MAXSIZE` are new.

### 2. Verify host-level Ollama settings

These are set by `./scripts/setup-ollama-host.sh` and are still required:

```bash
launchctl getenv OLLAMA_FLASH_ATTENTION   # 1
launchctl getenv OLLAMA_KV_CACHE_TYPE     # q8_0
launchctl getenv OLLAMA_KEEP_ALIVE        # 24h
launchctl getenv OLLAMA_MAX_LOADED_MODELS # 1
launchctl getenv OLLAMA_NUM_PARALLEL      # 1
```

If any are blank, re-run `./scripts/setup-ollama-host.sh`, then fully quit and
relaunch the Ollama app so it re-reads the environment.

### 3. Swap the model

```bash
ollama stop gemma4:12b-it-qat        # or whatever is currently loaded
ollama pull qwen2.5-coder:7b
```

Stopping matters: Ollama keeps the old KV cache allocation until the model is
unloaded, so a changed `num_ctx` will not take effect otherwise.

### 4. Rebuild and restart the container

The `Dockerfile` changed in Task 9, so a plain restart is not enough:

```bash
cd ~/Dev/ai-code-reviewer
docker compose up -d --build --force-recreate app
```

`docker compose restart` does **not** re-read `.env` and does **not** rebuild.
Always use the command above after editing `.env` or any source file.

### 5. Verify

```bash
# Container is healthy and the queue is idle
curl -s localhost:5000/health
# expect: {"queue_depth":0,"status":"ok"}

# Startup log shows the new tunables
docker compose logs app | grep "Ollama config"
# expect: num_batch=512 num_predict=320 include_context=False

# Pre-warm the model and confirm it is fully on GPU
ollama run qwen2.5-coder:7b "ok" </dev/null
ollama ps
# expect: PROCESSOR=100% GPU, CONTEXT=8192
```

If `PROCESSOR` shows any CPU percentage, the machine is out of memory. Drop
`OLLAMA_NUM_CTX` to 4096, run `ollama stop qwen2.5-coder:7b`, and repeat step 4.

### 6. Smoke test

Post a comment containing `/review` on any open merge request. Expect:

- HTTP 202 in `docker compose logs -f app` within milliseconds
- First inline discussion on the MR within roughly 15 seconds
- A summary note when all files are done

Posting `/review` a second time with no new commits should log
`diff unchanged since last review; skipping` and cost nothing.

### 7. Rollback

```bash
cd ~/Dev/ai-code-reviewer
git checkout <previous-commit-sha>
docker compose up -d --build --force-recreate app
```
