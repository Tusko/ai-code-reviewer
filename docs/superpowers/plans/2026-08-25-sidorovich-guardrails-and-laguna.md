# Sidorovich Guardrails and Laguna Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the bot from posting hundreds of duplicate comments on large merge requests, and move code review from local Ollama to OpenRouter `poolside/laguna-s-2.1`.

**Architecture:** A per-MR ledger of reviewed hunk content-hashes lives in a single GitLab note that the bot creates once and edits in place. Each run drops hunks it has already reviewed, refuses MRs above a file-count threshold with one note, and spends against a lifetime comment budget. Review requests go to OpenRouter; Ollama remains only as the optional Sidorovich voice fallback.

**Tech Stack:** Python 3.11, Flask, python-gitlab 4.4.0, requests, pytest 8.2.0. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-25-sidorovich-comment-guardrails-design.md`

## Global Constraints

- No new runtime dependencies. `requirements.txt` stays at flask, python-gitlab, requests, gunicorn.
- All new settings read through the existing `env_int` / `env_bool` / `env_list` helpers in `reviewer/config.py`, and are mirrored into `docker-compose.yml` and `.env.example`.
- Tests run with `python3 -m pytest` from the repo root; `pytest.ini` sets `testpaths = tests` and `pythonpath = .`.
- Code, comments, commit messages: English. User-facing GitLab comment copy: Ukrainian, matching the existing Sidorovich voice.
- Never write Russian-language strings. Ukrainian surzhyk in Sidorovich copy is deliberate and stays.
- Exact model id: `poolside/laguna-s-2.1`. Not the `:free` variant.
- `MAX_FILES` (40) per-run cap is unchanged by this plan.
- The single-worker queue and `--workers 1` in the Dockerfile are unchanged by this plan.
- No `git push`. Commit locally only.

## File Structure

**Created:**
- `reviewer/chat_types.py` — `ChatResult` and `clean_response`, shared by both model clients. Removes the backwards dependency where `openrouter_client` imports from `ollama_client`.
- `reviewer/voice.py` — Sidorovich voice rewrite: `VoiceState`, `flavor_review`, `prefer_ukrainian`, `preserves_findings`. Lifted out of `pipeline.py` so pipeline stays orchestration.
- `reviewer/ledger.py` — `Ledger` value type, `hunk_key`, marker serialisation, `LedgerStore` (owns the GitLab note handle).
- `tests/test_chat_types.py`, `tests/test_voice.py`, `tests/test_ledger.py`, `tests/test_review_backend.py`, `tests/test_pipeline_ledger.py`.

**Modified:**
- `reviewer/config.py` — eight new settings.
- `reviewer/openrouter_client.py` — `chat()` gains model/fallback parameters; new `review_chat()`.
- `reviewer/prompt.py:input_token_budget` — reads review-backend settings instead of Ollama's.
- `reviewer/gitlab_client.py` — `find_note_with`, `create_note`, `update_note`.
- `reviewer/pipeline.py` — `partition_reviewable`, `select_files` signature, `drop_known_hunks`, `ReviewState`, rewritten `review_merge_request`, `mute_merge_request`.
- `reviewer/ollama_client.py` — re-imports moved symbols from `chat_types`.
- `review_server.py` — `ReviewJob`, kill switch, `/sidorovich stop`.
- `tests/test_pipeline.py`, `tests/test_openrouter_client.py`, `tests/test_webhook.py` — updated imports and signatures.
- `docker-compose.yml`, `.env.example`, `README.md`.

**Task order rationale:** Tasks 1–2 are pure moves that unblock everything else without behaviour change. Tasks 3–5 migrate the backend and can ship on their own. Tasks 6–12 build the guardrails on top. Task 13 wires configuration and docs.

---

### Task 1: Extract shared chat types

`ChatResult` and `clean_response` live in `ollama_client.py`, and `openrouter_client.py` imports them from there. Once review leaves Ollama that direction is actively misleading. This task is a pure move: no behaviour changes.

**Files:**
- Create: `reviewer/chat_types.py`
- Modify: `reviewer/ollama_client.py:1-58`, `reviewer/openrouter_client.py:8`, `reviewer/pipeline.py:11`
- Test: `tests/test_chat_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `reviewer.chat_types` exporting `ChatResult` (frozen dataclass with fields `text: str`, `done_reason: str`, `prompt_eval_count: int`, `eval_count: int`, `elapsed_s: float`, and properties `truncated: bool`, `failed: bool`), `clean_response(text: str) -> str`, `LGTM_TEXT: str`, `FINDING_TAGS: tuple[str, ...]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_chat_types.py`:

```python
from reviewer.chat_types import (
    FINDING_TAGS, LGTM_TEXT, ChatResult, clean_response,
)


def test_chat_result_failed_covers_ratelimit():
    result = ChatResult("", "ratelimit", 0, 0, 0.0)
    assert result.failed is True


def test_chat_result_truncated_covers_length():
    result = ChatResult("x", "length", 0, 0, 0.0)
    assert result.truncated is True
    assert result.failed is False


def test_clean_response_normalises_bare_lgtm():
    assert clean_response("**LGTM.**") == LGTM_TEXT


def test_clean_response_drops_degenerate_reply():
    assert clean_response("the") == ""


def test_clean_response_preserves_fenced_indentation():
    text = "**🔴 [BLOCKER]** broken\n\n*Fix:*\n```py\nif x:\n    pass\n```"
    assert "\n    pass\n" in clean_response(text)


def test_finding_tags_exported():
    assert FINDING_TAGS == ("[BLOCKER]", "[SUGGESTION]", "[NIT]")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_chat_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reviewer.chat_types'`

- [ ] **Step 3: Create the new module**

Create `reviewer/chat_types.py` by moving these symbols out of `reviewer/ollama_client.py` verbatim: `HARMONY_TOKEN_RE`, `FENCE_SPLIT_RE`, `FINDING_TAGS`, `LGTM_TEXT`, `BARE_LGTM_RE`, `DEGENERATE_REPLIES`, `ChatResult`, `clean_response`.

```python
import re
from dataclasses import dataclass

HARMONY_TOKEN_RE = re.compile(r"<\|[^>]*\|?>")
FENCE_SPLIT_RE = re.compile(r"(```[\s\S]*?```)")
FINDING_TAGS = ("[BLOCKER]", "[SUGGESTION]", "[NIT]")
LGTM_TEXT = "LGTM. The changes are clean and follow best practices."
# `LGTM`, `LGTM.`, `[LGTM]`, `**LGTM.**` — any bare clean verdict.
BARE_LGTM_RE = re.compile(r"^[*\[\s]*LGTM[\]\s*.!]*$", re.IGNORECASE)
# Single-token replies the model emits when it gives up mid-generation.
DEGENERATE_REPLIES = frozenset({"the", "ok", "okay", "none", "n/a", "-", "..."})


@dataclass(frozen=True)
class ChatResult:
    text: str
    done_reason: str
    prompt_eval_count: int
    eval_count: int
    elapsed_s: float

    @property
    def truncated(self) -> bool:
        return self.done_reason in ("length", "timeout", "incomplete")

    @property
    def failed(self) -> bool:
        return self.done_reason in ("error", "ratelimit")


def clean_response(text: str) -> str:
    text = HARMONY_TOKEN_RE.sub(" ", text).strip()
    pieces = []
    for i, part in enumerate(FENCE_SPLIT_RE.split(text)):
        if i % 2 == 0:
            pieces.append(re.sub(r"[ \t]{2,}", " ", part))
        else:
            pieces.append(part)
    text = "".join(pieces)
    has_finding = any(tag in text for tag in FINDING_TAGS)
    # The model bailed and emitted a stray token. Returning anything readable
    # here would be posted verbatim as a review finding, so return nothing and
    # let the caller report an error.
    if not has_finding and text.strip().lower() in DEGENERATE_REPLIES:
        return ""
    if not has_finding and BARE_LGTM_RE.match(text.strip()):
        return LGTM_TEXT
    if "[LGTM]" in text and not has_finding:
        return LGTM_TEXT
    return text
```

- [ ] **Step 4: Update the importers**

In `reviewer/ollama_client.py`, delete the moved symbols and replace them with an import placed after `from reviewer import config`:

```python
from reviewer.chat_types import (
    DEGENERATE_REPLIES, FINDING_TAGS, LGTM_TEXT, ChatResult, clean_response,
)
```

Keep any symbol `ollama_client` still uses in its own streaming code. Do not delete `chat()`.

In `reviewer/openrouter_client.py:8`, replace:

```python
from reviewer.ollama_client import ChatResult, clean_response
```

with:

```python
from reviewer.chat_types import ChatResult, clean_response
```

In `reviewer/pipeline.py:11`, replace:

```python
from reviewer.ollama_client import LGTM_TEXT, ChatResult, chat
```

with:

```python
from reviewer.chat_types import FINDING_TAGS, LGTM_TEXT, ChatResult
from reviewer.ollama_client import chat
```

Then delete the duplicate `FINDING_TAGS = ("[BLOCKER]", "[SUGGESTION]", "[NIT]")` line from `reviewer/pipeline.py:18` — it is now imported.

- [ ] **Step 5: Run the full suite to verify nothing broke**

Run: `python3 -m pytest -v`
Expected: PASS. If a test imports `ChatResult` from `reviewer.ollama_client`, that still works via the re-export; leave those imports alone in this task.

- [ ] **Step 6: Commit**

```bash
git add reviewer/chat_types.py reviewer/ollama_client.py reviewer/openrouter_client.py reviewer/pipeline.py tests/test_chat_types.py
git commit -m "refactor: move ChatResult and clean_response to chat_types"
```

---

### Task 2: Extract the Sidorovich voice into its own module

`pipeline.py` is 431 lines and later tasks add ledger handling and budget accounting to it. The voice rewrite is a self-contained concern with one entry point.

**Files:**
- Create: `reviewer/voice.py`
- Modify: `reviewer/pipeline.py` (remove lines 14-20 constants and the voice functions), `tests/test_pipeline.py:1-11`
- Test: `tests/test_voice.py`

**Interfaces:**
- Consumes: `reviewer.chat_types.ChatResult`, `reviewer.chat_types.FINDING_TAGS` from Task 1.
- Produces: `reviewer.voice` exporting `VoiceState` (class with `.enabled: bool`, `.record_failure()`, `.record_success()`), `flavor_review(text: str, voice: VoiceState | None = None) -> str`, `prefer_ukrainian(result: ChatResult, retry) -> ChatResult`, `preserves_findings(original: str, flavored: str) -> bool`, `VOICE_DEADLINE_S: int`, `VOICE_FAILURE_LIMIT: int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_voice.py`:

```python
from reviewer import voice
from reviewer.chat_types import ChatResult
from reviewer.voice import VoiceState, preserves_findings


def _result(text, done_reason="stop"):
    return ChatResult(text, done_reason, 0, 0, 0.0)


def test_voice_state_opens_closed_after_limit():
    state = VoiceState()
    assert state.enabled is True
    for _ in range(voice.VOICE_FAILURE_LIMIT):
        state.record_failure()
    assert state.enabled is False


def test_voice_state_success_resets_failures():
    state = VoiceState()
    state.record_failure()
    state.record_success()
    assert state.enabled is True


def test_preserves_findings_rejects_dropped_tag():
    original = "**🔴 [BLOCKER]** boom"
    assert preserves_findings(original, "нема нічого") is False


def test_preserves_findings_rejects_invented_fence():
    original = "**🔵 [NIT]** cast it"
    assert preserves_findings(original, "**🔵 [NIT]** хуйня\n```py\nx = 1\n```") is False


def test_preserves_findings_accepts_identical_fences():
    original = "**🟡 [SUGGESTION]** fix\n```py\nx = 1\n```"
    flavored = "**🟡 [SUGGESTION]** блять, полагодь\n```py\nx = 1\n```"
    assert preserves_findings(original, flavored) is True


def test_flavor_review_returns_text_unchanged_without_key(monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    assert voice.flavor_review("**🔴 [BLOCKER]** boom") == "**🔴 [BLOCKER]** boom"


def test_flavor_review_keeps_dry_when_voice_disabled(monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr("reviewer.config.SNARK", True)
    state = VoiceState()
    for _ in range(voice.VOICE_FAILURE_LIMIT):
        state.record_failure()
    assert voice.flavor_review("**🔴 [BLOCKER]** boom", state) == "**🔴 [BLOCKER]** boom"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_voice.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reviewer.voice'`

- [ ] **Step 3: Create `reviewer/voice.py`**

Move these out of `reviewer/pipeline.py` verbatim: `VOICE_DEADLINE_S`, `VOICE_FAILURE_LIMIT`, `FENCE_BODY_RE`, `VoiceState`, `_fence_bodies`, `preserves_findings`, `_voice_chat`, `flavor_review`, `prefer_ukrainian`. The module header:

```python
import logging
import re
from typing import Sequence

from reviewer import config, openrouter_client, prompt as prompt_mod
from reviewer.chat_types import FINDING_TAGS, ChatResult

VOICE_DEADLINE_S = 20
# Consecutive failed voice calls before the rest of the MR stays dry. Free-tier
# OpenRouter models rate-limit mid-review; retrying every file just makes half
# the comments Sidorovich and half of them dry.
VOICE_FAILURE_LIMIT = 2
FENCE_BODY_RE = re.compile(r"```(?:\w*)\n?(.*?)```", re.DOTALL)
```

The function bodies move unchanged. `prefer_ukrainian` moves too: it is used by `pipeline.summarize_release_mr` as well, which will import it from `reviewer.voice`.

- [ ] **Step 4: Update `pipeline.py`**

Delete the moved symbols. Add to the imports:

```python
from reviewer.voice import VoiceState, flavor_review, prefer_ukrainian
```

`review_file` keeps calling `flavor_review(finding, voice)` unchanged. `summarize_release_mr` keeps calling `prefer_ukrainian(...)` unchanged.

- [ ] **Step 5: Update `tests/test_pipeline.py` imports**

Replace the import block at `tests/test_pipeline.py:1-11` with:

```python
import pytest

from reviewer import pipeline
from reviewer.chat_types import ChatResult
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.pipeline import (
    FileOutcome, build_prompt_ladder, render_summary, review_file,
    review_merge_request, select_files,
)
from reviewer.voice import flavor_review, prefer_ukrainian, preserves_findings
```

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest -v`
Expected: PASS, same test count as before plus the new `tests/test_voice.py` cases.

- [ ] **Step 7: Commit**

```bash
git add reviewer/voice.py reviewer/pipeline.py tests/test_voice.py tests/test_pipeline.py
git commit -m "refactor: move Sidorovich voice rewrite into reviewer/voice.py"
```

---

### Task 3: Add review-backend settings and `review_chat`

**Files:**
- Modify: `reviewer/config.py:36-46` (new block after the OpenRouter section), `reviewer/openrouter_client.py:19-63`
- Test: `tests/test_review_backend.py`

**Interfaces:**
- Consumes: `reviewer.chat_types.ChatResult` from Task 1.
- Produces:
  - `config.OPENROUTER_REVIEW_MODEL: str`, `config.OPENROUTER_REVIEW_FALLBACK_MODELS: list[str]`, `config.REVIEW_CONTEXT_TOKENS: int`, `config.REVIEW_MAX_OUTPUT_TOKENS: int`
  - `openrouter_client.chat(system, user, deadline_s, *, temperature=1.0, max_tokens=None, history=(), model=None, fallback_models=None) -> ChatResult`
  - `openrouter_client.review_chat(system: str, user: str, deadline_s: int) -> ChatResult`

- [ ] **Step 1: Write the failing test**

Create `tests/test_review_backend.py`:

```python
import pytest

from reviewer import openrouter_client


class FakeResponse:
    ok = True
    status_code = 200
    headers: dict = {}

    def __init__(self, content="**🔴 [BLOCKER]** boom"):
        self._content = content

    def json(self):
        return {
            "choices": [
                {"message": {"content": self._content}, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "poolside/laguna-s-2.1",
        }


@pytest.fixture
def captured(monkeypatch):
    """Captures the request body openrouter_client would have sent."""
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent["url"] = url
        sent["body"] = json
        return FakeResponse()

    monkeypatch.setattr("reviewer.openrouter_client.requests.post", fake_post)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", "sk-test")
    return sent


def test_review_chat_uses_review_model(captured, monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_REVIEW_MODEL", "poolside/laguna-s-2.1")
    monkeypatch.setattr("reviewer.config.OPENROUTER_MODEL", "voice/model")
    openrouter_client.review_chat("sys", "user", 30)
    assert captured["body"]["model"] == "poolside/laguna-s-2.1"


def test_review_chat_uses_review_output_cap(captured, monkeypatch):
    monkeypatch.setattr("reviewer.config.REVIEW_MAX_OUTPUT_TOKENS", 4096)
    monkeypatch.setattr("reviewer.config.OPENROUTER_MAX_TOKENS", 512)
    openrouter_client.review_chat("sys", "user", 30)
    assert captured["body"]["max_tokens"] == 4096


def test_review_chat_omits_models_key_when_no_fallbacks(captured, monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_REVIEW_FALLBACK_MODELS", [])
    openrouter_client.review_chat("sys", "user", 30)
    assert "models" not in captured["body"]


def test_review_chat_sends_fallback_chain(captured, monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_REVIEW_MODEL", "poolside/laguna-s-2.1")
    monkeypatch.setattr(
        "reviewer.config.OPENROUTER_REVIEW_FALLBACK_MODELS", ["poolside/laguna-xs-2.1"],
    )
    openrouter_client.review_chat("sys", "user", 30)
    assert captured["body"]["models"] == [
        "poolside/laguna-s-2.1", "poolside/laguna-xs-2.1",
    ]


def test_review_chat_is_low_temperature(captured):
    openrouter_client.review_chat("sys", "user", 30)
    assert captured["body"]["temperature"] == 0.1


def test_voice_chat_still_uses_voice_model(captured, monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_MODEL", "voice/model")
    monkeypatch.setattr("reviewer.config.OPENROUTER_REVIEW_MODEL", "poolside/laguna-s-2.1")
    openrouter_client.chat("sys", "user", 30)
    assert captured["body"]["model"] == "voice/model"


def test_review_chat_without_key_reports_error(monkeypatch):
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    result = openrouter_client.review_chat("sys", "user", 30)
    assert result.failed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_review_backend.py -v`
Expected: FAIL with `AttributeError: module 'reviewer.openrouter_client' has no attribute 'review_chat'`

- [ ] **Step 3: Add the config settings**

In `reviewer/config.py`, insert after the `SIDOROVICH_OLLAMA_FALLBACK` line:

```python
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
```

- [ ] **Step 4: Parameterise `openrouter_client.chat`**

Change the signature at `reviewer/openrouter_client.py:19`:

```python
def chat(
    system: str,
    user: str,
    deadline_s: int,
    *,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    history: Sequence[dict] = (),
    model: str | None = None,
    fallback_models: Sequence[str] | None = None,
) -> ChatResult:
```

Replace the body construction so it uses the parameters, defaulting to the voice settings:

```python
    chosen = model or config.OPENROUTER_MODEL
    fallbacks = (
        list(fallback_models)
        if fallback_models is not None
        else list(config.OPENROUTER_FALLBACK_MODELS)
    )

    body = {
        "model": chosen,
        "messages": [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens or config.OPENROUTER_MAX_TOKENS,
    }
    if fallbacks:
        # OpenRouter walks this list itself when a provider 429s or errors, so
        # one request already survives a saturated pool.
        body["models"] = [chosen, *fallbacks]
```

Update the docstring's first line to `"""One-shot OpenRouter chat completion."""`.

- [ ] **Step 5: Add `review_chat`**

Append to `reviewer/openrouter_client.py`, directly after `chat`:

```python
def review_chat(system: str, user: str, deadline_s: int) -> ChatResult:
    """Code review completion. Low temperature: a review must not improvise.

    The Ollama path this replaces pinned seed=42 for reproducibility. OpenRouter
    exposes no seed, so identical diffs may now yield slightly different
    findings. The ledger dedupes on diff content, not on review text, so this
    does not cause repeat comments.
    """
    return chat(
        system,
        user,
        deadline_s,
        temperature=0.1,
        max_tokens=config.REVIEW_MAX_OUTPUT_TOKENS,
        model=config.OPENROUTER_REVIEW_MODEL,
        fallback_models=config.OPENROUTER_REVIEW_FALLBACK_MODELS,
    )
```

- [ ] **Step 6: Run the tests**

Run: `python3 -m pytest tests/test_review_backend.py tests/test_openrouter_client.py -v`
Expected: PASS. If an existing `test_openrouter_client.py` case asserts on `body["models"]` built from `OPENROUTER_FALLBACK_MODELS`, it still passes — the default path is unchanged.

- [ ] **Step 7: Commit**

```bash
git add reviewer/config.py reviewer/openrouter_client.py tests/test_review_backend.py
git commit -m "feat: add OpenRouter review model settings and review_chat"
```

---

### Task 4: Make the prompt budget describe the review backend

`input_token_budget()` computes about 7364 tokens from `OLLAMA_NUM_CTX`. Pointing review at a 262144-token model while still sending 7364-token prompts wastes the entire migration.

**Files:**
- Modify: `reviewer/prompt.py:154-170`
- Test: `tests/test_prompt.py`

**Interfaces:**
- Consumes: `config.REVIEW_CONTEXT_TOKENS`, `config.REVIEW_MAX_OUTPUT_TOKENS` from Task 3.
- Produces: `prompt.input_token_budget() -> int` now derived from review settings. `prompt.fits(text)` is unchanged in signature.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prompt.py`:

```python
from reviewer import prompt as prompt_mod


def test_input_token_budget_follows_review_context(monkeypatch):
    monkeypatch.setattr("reviewer.config.REVIEW_CONTEXT_TOKENS", 262144)
    monkeypatch.setattr("reviewer.config.REVIEW_MAX_OUTPUT_TOKENS", 4096)
    monkeypatch.setattr("reviewer.config.PROMPT_TOKEN_BUFFER", 128)
    budget = prompt_mod.input_token_budget()
    assert budget > 250_000


def test_input_token_budget_ignores_ollama_context(monkeypatch):
    monkeypatch.setattr("reviewer.config.REVIEW_CONTEXT_TOKENS", 262144)
    monkeypatch.setattr("reviewer.config.REVIEW_MAX_OUTPUT_TOKENS", 4096)
    before = prompt_mod.input_token_budget()
    monkeypatch.setattr("reviewer.config.OLLAMA_NUM_CTX", 512)
    assert prompt_mod.input_token_budget() == before


def test_input_token_budget_floors_at_256(monkeypatch):
    monkeypatch.setattr("reviewer.config.REVIEW_CONTEXT_TOKENS", 100)
    monkeypatch.setattr("reviewer.config.REVIEW_MAX_OUTPUT_TOKENS", 90)
    assert prompt_mod.input_token_budget() == 256
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prompt.py -k budget -v`
Expected: FAIL — `test_input_token_budget_follows_review_context` asserts `> 250000` but gets roughly 7364.

- [ ] **Step 3: Rewrite `input_token_budget`**

Replace `reviewer/prompt.py:154-170` with:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_prompt.py -v`
Expected: PASS.

- [ ] **Step 5: Note the ladder behaviour change**

Add this comment directly above `build_prompt_ladder` in `reviewer/pipeline.py`:

```python
# With a 262k-token review context essentially every file fits at L1, so the L2
# per-hunk rung no longer fires in practice. It is kept because it is what
# guarantees no file is silently dropped for being too large.
```

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest -v`
Expected: PASS. `tests/test_pipeline.py` cases that force L2 by shrinking the context must now shrink `REVIEW_CONTEXT_TOKENS` instead of `OLLAMA_NUM_CTX`; update any that fail.

- [ ] **Step 7: Commit**

```bash
git add reviewer/prompt.py reviewer/pipeline.py tests/test_prompt.py tests/test_pipeline.py
git commit -m "feat: size the prompt budget from the review backend"
```

---

### Task 5: Route review through OpenRouter with a rate-limit breaker

**Files:**
- Modify: `reviewer/pipeline.py` (`review_file`, new `ReviewState`)
- Test: `tests/test_review_backend.py`

**Interfaces:**
- Consumes: `openrouter_client.review_chat` from Task 3.
- Produces:
  - `pipeline.ReviewState` — class with `.open: bool`, `.record(done_reason: str) -> None`
  - `pipeline.REVIEW_FAILURE_LIMIT: int` = 2
  - `pipeline.review_file(mr, file_diff, context, voice=None, review_state=None) -> FileOutcome`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_review_backend.py`:

```python
from reviewer import pipeline
from reviewer.chat_types import ChatResult
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.pipeline import ReviewState


def _fd(path="a.py"):
    return FileDiff(
        old_path=path, new_path=path, is_new=False, is_deleted=False,
        is_renamed=False, is_binary=False,
        hunks=(Hunk(1, 1, (" ctx", "+added")),),
    )


class _MR:
    pass


def test_review_state_trips_after_two_ratelimits():
    state = ReviewState()
    assert state.open is True
    state.record("ratelimit")
    assert state.open is True
    state.record("ratelimit")
    assert state.open is False


def test_review_state_resets_on_success():
    state = ReviewState()
    state.record("ratelimit")
    state.record("stop")
    state.record("ratelimit")
    assert state.open is True


def test_review_file_calls_openrouter_not_ollama(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: (
            calls.append("openrouter"),
            ChatResult("[LGTM]", "stop", 0, 0, 0.0),
        )[1],
    )

    def _boom(*args, **kwargs):
        raise AssertionError("ollama_client.chat must not be called for review")

    monkeypatch.setattr("reviewer.ollama_client.chat", _boom)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    outcome = pipeline.review_file(_MR(), _fd(), "")
    assert calls == ["openrouter"]
    assert outcome.status == "clean"


def test_review_file_records_ratelimit_on_state(monkeypatch):
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: ChatResult("", "ratelimit", 0, 0, 0.0),
    )
    state = ReviewState()
    outcome = pipeline.review_file(_MR(), _fd(), "", None, state)
    assert outcome.status == "error"
    assert state.ratelimits == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_review_backend.py -k "review_state or review_file" -v`
Expected: FAIL with `AttributeError: module 'reviewer.pipeline' has no attribute 'ReviewState'`

- [ ] **Step 3: Add `ReviewState` to `pipeline.py`**

Insert after the `VoiceState` import block:

```python
# Consecutive rate-limited review calls before the run gives up. On the paid
# tier this should never fire; it exists because OPENROUTER_REVIEW_MODEL is a
# knob and someone will eventually point it at a `:free` model.
REVIEW_FAILURE_LIMIT = 2


class ReviewState:
    """Per-MR circuit breaker for rate-limited review calls."""

    def __init__(self) -> None:
        self.ratelimits = 0

    @property
    def open(self) -> bool:
        return self.ratelimits < REVIEW_FAILURE_LIMIT

    def record(self, done_reason: str) -> None:
        if done_reason == "ratelimit":
            self.ratelimits += 1
            if not self.open:
                logging.warning(
                    "Review stopped for this MR after %s consecutive rate limits",
                    self.ratelimits,
                )
        else:
            self.ratelimits = 0
```

- [ ] **Step 4: Point `review_file` at OpenRouter**

In `reviewer/pipeline.py`, change the import from

```python
from reviewer.ollama_client import chat
```

to

```python
from reviewer.openrouter_client import review_chat
```

Change the `review_file` signature to:

```python
def review_file(
    mr,
    file_diff: FileDiff,
    context: str,
    voice: "VoiceState | None" = None,
    review_state: "ReviewState | None" = None,
) -> FileOutcome:
```

Inside the prompt loop, replace

```python
        result = chat(prompt_mod.SYSTEM_PROMPT, text, deadline_s=config.PER_FILE_TIMEOUT_S)
        if result.failed:
            return FileOutcome(path, "error", result.done_reason)
```

with

```python
        result = review_chat(
            prompt_mod.SYSTEM_PROMPT, text, deadline_s=config.PER_FILE_TIMEOUT_S,
        )
        if review_state is not None:
            review_state.record(result.done_reason)
        if result.failed:
            return FileOutcome(path, "error", result.done_reason)
```

Leave every other branch of `review_file` exactly as it is.

- [ ] **Step 5: Check the Ollama summary path still compiles**

`summary_chat` in `pipeline.py` calls the bare `chat(...)` for the Ollama voice fallback. Since the module-level `chat` import is gone, change that call to a qualified one. Add `ollama_client` to the existing package import line and change the call inside `summary_chat` from `chat(` to `ollama_client.chat(`:

```python
from reviewer import (
    config, gitlab_client, ollama_client, openrouter_client, prompt as prompt_mod,
)
```

- [ ] **Step 6: Run the tests**

Run: `python3 -m pytest -v`
Expected: PASS. Existing `tests/test_pipeline.py` cases that patch `reviewer.pipeline.chat` must be repatched to `reviewer.pipeline.review_chat`; update every occurrence.

- [ ] **Step 7: Commit**

```bash
git add reviewer/pipeline.py tests/test_review_backend.py tests/test_pipeline.py
git commit -m "feat: run code review on OpenRouter with a rate-limit breaker"
```

---

### Task 6: The ledger value type

**Files:**
- Create: `reviewer/ledger.py`
- Modify: `reviewer/config.py` (guardrail settings)
- Test: `tests/test_ledger.py`

**Interfaces:**
- Consumes: `reviewer.diff_parser.Hunk`.
- Produces:
  - `config.MAX_MR_FILES: int`, `config.MR_COMMENT_BUDGET: int`, `config.LEDGER_MAX_HUNKS: int`, `config.SIDOROVICH_ENABLED: bool`
  - `ledger.MARKER_PREFIX: str` = `"sidorovich-state:v1"`
  - `ledger.LedgerUnavailable(Exception)`
  - `ledger.hunk_key(path: str, hunk: Hunk) -> str`
  - `ledger.Ledger` — frozen dataclass, fields `head: str = ""`, `posted: int = 0`, `muted: bool = False`, `oversized: bool = False`, `hunks: tuple[str, ...] = ()`; methods `remaining() -> int`, `record(keys) -> Ledger`, `spend(n=1) -> Ledger`, `mute() -> Ledger`, `unmute_and_reset() -> Ledger`, `mark_oversized() -> Ledger`, `at_head(head: str) -> Ledger`
  - `ledger.to_marker(value: Ledger) -> str`, `ledger.parse_marker(body: str) -> Ledger`, `ledger.render_note(value: Ledger) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ledger.py`:

```python
import pytest

from reviewer import ledger as ledger_mod
from reviewer.diff_parser import Hunk
from reviewer.ledger import Ledger, LedgerUnavailable, hunk_key


def test_hunk_key_ignores_line_number_shift():
    """A rebase moves hunks without changing them. The key must not move."""
    lines = (" ctx", "+added", " tail")
    assert hunk_key("a.py", Hunk(1, 1, lines)) == hunk_key("a.py", Hunk(90, 90, lines))


def test_hunk_key_changes_with_content():
    a = hunk_key("a.py", Hunk(1, 1, (" ctx", "+added")))
    b = hunk_key("a.py", Hunk(1, 1, (" ctx", "+altered")))
    assert a != b


def test_hunk_key_changes_with_path():
    lines = (" ctx", "+added")
    assert hunk_key("a.py", Hunk(1, 1, lines)) != hunk_key("b.py", Hunk(1, 1, lines))


def test_marker_round_trip():
    original = Ledger(
        head="abc1234", posted=7, muted=True, oversized=True, hunks=("aa", "bb"),
    )
    assert ledger_mod.parse_marker(ledger_mod.to_marker(original)) == original


def test_parse_marker_returns_fresh_ledger_when_absent():
    assert ledger_mod.parse_marker("just a normal comment") == Ledger()


def test_parse_marker_raises_on_broken_json():
    body = f"<!-- {ledger_mod.MARKER_PREFIX} {{not json}} -->"
    with pytest.raises(LedgerUnavailable):
        ledger_mod.parse_marker(body)


def test_record_is_idempotent():
    value = Ledger().record(["aa", "bb"]).record(["bb", "cc"])
    assert value.hunks == ("aa", "bb", "cc")


def test_record_evicts_oldest_past_cap(monkeypatch):
    monkeypatch.setattr("reviewer.config.LEDGER_MAX_HUNKS", 3)
    value = Ledger().record(["a", "b", "c", "d"])
    assert value.hunks == ("b", "c", "d")


def test_remaining_counts_down_from_budget(monkeypatch):
    monkeypatch.setattr("reviewer.config.MR_COMMENT_BUDGET", 30)
    assert Ledger(posted=28).remaining() == 2


def test_remaining_never_negative(monkeypatch):
    monkeypatch.setattr("reviewer.config.MR_COMMENT_BUDGET", 30)
    assert Ledger(posted=99).remaining() == 0


def test_unmute_and_reset_clears_both():
    value = Ledger(posted=30, muted=True).unmute_and_reset()
    assert value.posted == 0
    assert value.muted is False


def test_render_note_embeds_a_parseable_marker():
    value = Ledger(head="abc1234", posted=3, hunks=("aa",))
    assert ledger_mod.parse_marker(ledger_mod.render_note(value)) == value


def test_render_note_is_human_readable():
    body = ledger_mod.render_note(Ledger(head="abc1234", posted=3))
    assert "Сідорович" in body
    assert "abc1234" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ledger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reviewer.ledger'`

- [ ] **Step 3: Add the guardrail settings to `config.py`**

Insert into the "Deadlines and limits" block of `reviewer/config.py`, after `MAX_FILES`:

```python
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
```

- [ ] **Step 4: Create `reviewer/ledger.py`**

```python
import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Iterable

from reviewer import config
from reviewer.diff_parser import Hunk

MARKER_PREFIX = "sidorovich-state:v1"
MARKER_RE = re.compile(
    r"<!--\s*" + re.escape(MARKER_PREFIX) + r"\s*(\{.*?\})\s*-->", re.DOTALL,
)


class LedgerUnavailable(Exception):
    """The MR's review state could not be read. The run must not proceed.

    Treating an unreadable ledger as an empty one would re-review the whole MR,
    which is the exact flood this module exists to prevent.
    """


def hunk_key(path: str, hunk: Hunk) -> str:
    """Content hash of one hunk, stable across rebases.

    `new_start` and `old_start` are deliberately excluded: line numbers shift
    whenever an unrelated earlier hunk changes, but the content does not.
    """
    digest = hashlib.sha256()
    digest.update(path.encode())
    digest.update(b"\x00")
    for line in hunk.lines:
        digest.update(line.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:12]


@dataclass(frozen=True)
class Ledger:
    head: str = ""
    posted: int = 0
    muted: bool = False
    oversized: bool = False
    hunks: tuple[str, ...] = ()

    def remaining(self) -> int:
        return max(0, config.MR_COMMENT_BUDGET - self.posted)

    def record(self, keys: Iterable[str]) -> "Ledger":
        merged = list(self.hunks)
        known = set(self.hunks)
        for key in keys:
            if key not in known:
                merged.append(key)
                known.add(key)
        if len(merged) > config.LEDGER_MAX_HUNKS:
            merged = merged[len(merged) - config.LEDGER_MAX_HUNKS:]
        return replace(self, hunks=tuple(merged))

    def spend(self, n: int = 1) -> "Ledger":
        return replace(self, posted=self.posted + n)

    def mute(self) -> "Ledger":
        return replace(self, muted=True)

    def unmute_and_reset(self) -> "Ledger":
        return replace(self, muted=False, posted=0)

    def mark_oversized(self) -> "Ledger":
        return replace(self, oversized=True)

    def at_head(self, head: str) -> "Ledger":
        return replace(self, head=head)


def to_marker(value: Ledger) -> str:
    payload = {
        "head": value.head,
        "posted": value.posted,
        "muted": value.muted,
        "oversized": value.oversized,
        "hunks": list(value.hunks),
    }
    return f"<!-- {MARKER_PREFIX} {json.dumps(payload, separators=(',', ':'))} -->"


def parse_marker(body: str) -> Ledger:
    """Reads a ledger out of a note body. No marker means a fresh MR."""
    match = MARKER_RE.search(body or "")
    if not match:
        return Ledger()
    try:
        payload = json.loads(match.group(1))
    except ValueError as exc:
        raise LedgerUnavailable(f"state marker is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LedgerUnavailable("state marker is not a JSON object")
    try:
        return Ledger(
            head=str(payload.get("head") or ""),
            posted=int(payload.get("posted") or 0),
            muted=bool(payload.get("muted")),
            oversized=bool(payload.get("oversized")),
            hunks=tuple(str(key) for key in payload.get("hunks") or ()),
        )
    except (TypeError, ValueError) as exc:
        raise LedgerUnavailable(f"state marker has bad field types: {exc}") from exc


def render_note(value: Ledger) -> str:
    head = value.head or "—"
    status = "заглушений" if value.muted else "активний"
    return (
        "🔒 **Сідорович — стан рев'ю**\n\n"
        f"Коментарів: {value.posted}/{config.MR_COMMENT_BUDGET} · "
        f"останній head: `{head}` · {status}\n\n"
        "_Цю нотатку я редагую, а не пишу заново. Не чіпай._\n\n"
        + to_marker(value)
    )
```

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_ledger.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 6: Commit**

```bash
git add reviewer/ledger.py reviewer/config.py tests/test_ledger.py
git commit -m "feat: add per-MR review ledger with rebase-stable hunk keys"
```

---

### Task 7: Persist the ledger in a GitLab state note

**Files:**
- Modify: `reviewer/gitlab_client.py` (append three helpers), `reviewer/ledger.py` (append `LedgerStore`)
- Test: `tests/test_ledger.py`

**Interfaces:**
- Consumes: `ledger.Ledger`, `ledger.MARKER_PREFIX`, `ledger.parse_marker`, `ledger.render_note` from Task 6.
- Produces:
  - `gitlab_client.find_note_with(mr, marker: str)` — returns the note object or `None`; propagates API exceptions
  - `gitlab_client.create_note(mr, body: str)` — returns the created note object
  - `gitlab_client.update_note(note, body: str) -> None`
  - `ledger.LedgerStore` — `.ledger: Ledger` (assignable), `.load(mr) -> LedgerStore` classmethod raising `LedgerUnavailable`, `.save() -> None`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ledger.py`:

```python
from reviewer.ledger import LedgerStore


class FakeNote:
    def __init__(self, body=""):
        self.body = body
        self.saves = 0

    def save(self):
        self.saves += 1


class FakeNotes:
    def __init__(self, notes=(), raises=False):
        self._notes = list(notes)
        self._raises = raises
        self.created = []

    def list(self, iterator=False):
        if self._raises:
            raise RuntimeError("gitlab is down")
        return list(self._notes)

    def create(self, payload):
        note = FakeNote(payload["body"])
        self._notes.append(note)
        self.created.append(payload["body"])
        return note


class FakeMR:
    def __init__(self, notes=(), raises=False):
        self.notes = FakeNotes(notes, raises)


def test_store_load_fresh_when_no_marker():
    store = LedgerStore.load(FakeMR([FakeNote("unrelated chatter")]))
    assert store.ledger == Ledger()


def test_store_load_reads_existing_marker():
    body = ledger_mod.render_note(Ledger(head="abc1234", posted=4, hunks=("aa",)))
    store = LedgerStore.load(FakeMR([FakeNote("noise"), FakeNote(body)]))
    assert store.ledger.posted == 4
    assert store.ledger.hunks == ("aa",)


def test_store_load_raises_when_api_fails():
    with pytest.raises(LedgerUnavailable):
        LedgerStore.load(FakeMR(raises=True))


def test_store_load_raises_on_broken_marker():
    mr = FakeMR([FakeNote(f"<!-- {ledger_mod.MARKER_PREFIX} {{broken}} -->")])
    with pytest.raises(LedgerUnavailable):
        LedgerStore.load(mr)


def test_store_save_creates_note_once_then_edits():
    mr = FakeMR()
    store = LedgerStore.load(mr)
    store.ledger = store.ledger.spend(1)
    store.save()
    assert len(mr.notes.created) == 1

    store.ledger = store.ledger.spend(1)
    store.save()
    assert len(mr.notes.created) == 1, "second save must edit, not post again"
    assert ledger_mod.parse_marker(mr.notes._notes[0].body).posted == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ledger.py -k store -v`
Expected: FAIL with `ImportError: cannot import name 'LedgerStore'`

- [ ] **Step 3: Add the GitLab note helpers**

Append to `reviewer/gitlab_client.py`:

```python
def find_note_with(mr, marker: str):
    """First note on the MR whose body contains `marker`, or None.

    API failures propagate: the caller must be able to tell "no state yet" from
    "could not read state".
    """
    for note in mr.notes.list(iterator=True):
        if marker in (getattr(note, "body", None) or ""):
            return note
    return None


def create_note(mr, body: str):
    return mr.notes.create({"body": body})


def update_note(note, body: str) -> None:
    note.body = body
    note.save()
```

- [ ] **Step 4: Add `LedgerStore` to `reviewer/ledger.py`**

Change the existing `from reviewer import config` line to
`from reviewer import config, gitlab_client`, then append:

```python
class LedgerStore:
    """Owns the GitLab state note. `Ledger` itself stays a pure value.

    The note object is cached so a save is one API write, not a re-listing of
    every note on the merge request.
    """

    def __init__(self, mr, note, value: Ledger) -> None:
        self.mr = mr
        self._note = note
        self.ledger = value

    @classmethod
    def load(cls, mr) -> "LedgerStore":
        try:
            note = gitlab_client.find_note_with(mr, MARKER_PREFIX)
        except LedgerUnavailable:
            raise
        except Exception as exc:
            raise LedgerUnavailable(f"could not list MR notes: {exc}") from exc
        if note is None:
            return cls(mr, None, Ledger())
        return cls(mr, note, parse_marker(getattr(note, "body", None) or ""))

    def save(self) -> None:
        """Writes the current value. Failures propagate.

        A failed save means posted comments went unrecorded; continuing would
        post them again on the next push, so the run must stop instead.
        """
        body = render_note(self.ledger)
        if self._note is None:
            self._note = gitlab_client.create_note(self.mr, body)
        else:
            gitlab_client.update_note(self._note, body)
```

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_ledger.py -v`
Expected: PASS, 18 tests.

- [ ] **Step 6: Commit**

```bash
git add reviewer/ledger.py reviewer/gitlab_client.py tests/test_ledger.py
git commit -m "feat: persist the review ledger in an edited GitLab note"
```

---

### Task 8: Split reviewable filtering out of `select_files`

The oversized check needs the reviewable file count *before* file selection runs, so `is_reviewable` moves out of `select_files`. The skip reasons it produces must still reach the summary.

**Files:**
- Modify: `reviewer/pipeline.py:64-80`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `reviewer.filters.is_reviewable`.
- Produces:
  - `pipeline.partition_reviewable(file_diffs) -> tuple[list[FileDiff], list[FileOutcome]]`
  - `pipeline.select_files(reviewable: Sequence[FileDiff], skipped: Sequence[FileOutcome]) -> tuple[list[FileDiff], list[FileOutcome]]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
from reviewer.pipeline import partition_reviewable


def test_partition_reviewable_splits_and_names_reasons():
    good = fd("app.py")
    lock = fd("poetry.lock")
    keep, skipped = partition_reviewable([good, lock])
    assert [f.new_path for f in keep] == ["app.py"]
    assert [(o.path, o.detail) for o in skipped] == [("poetry.lock", "lockfile")]


def test_select_files_carries_skip_outcomes_through():
    keep, skipped = partition_reviewable([fd("app.py"), fd("yarn.lock")])
    kept, outcomes = select_files(keep, skipped)
    assert [f.new_path for f in kept] == ["app.py"]
    assert any(o.detail == "lockfile" for o in outcomes)


def test_select_files_caps_at_max_files(monkeypatch):
    monkeypatch.setattr("reviewer.config.MAX_FILES", 2)
    files = [fd(f"f{i}.py", added=i + 1) for i in range(5)]
    kept, outcomes = select_files(files, [])
    assert len(kept) == 2
    assert sum(1 for o in outcomes if o.detail == "over MAX_FILES limit") == 3


def test_select_files_keeps_smallest_first(monkeypatch):
    monkeypatch.setattr("reviewer.config.MAX_FILES", 2)
    files = [fd("big.py", added=9), fd("small.py", added=1), fd("mid.py", added=4)]
    kept, _ = select_files(files, [])
    assert [f.new_path for f in kept] == ["small.py", "mid.py"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_pipeline.py -k "partition or select_files" -v`
Expected: FAIL with `ImportError: cannot import name 'partition_reviewable'`

- [ ] **Step 3: Replace `select_files` in `pipeline.py`**

Replace `reviewer/pipeline.py:64-80` with:

```python
def partition_reviewable(
    file_diffs: Sequence[FileDiff],
) -> tuple[list[FileDiff], list[FileOutcome]]:
    """Splits diffs into reviewable files and named skip outcomes."""
    kept: list[FileDiff] = []
    outcomes: list[FileOutcome] = []
    for fd in file_diffs:
        ok, reason = is_reviewable(fd)
        if ok:
            kept.append(fd)
        else:
            outcomes.append(FileOutcome(fd.new_path, "skipped", reason))
    return kept, outcomes


def select_files(
    reviewable: Sequence[FileDiff], skipped: Sequence[FileOutcome] = (),
) -> tuple[list[FileDiff], list[FileOutcome]]:
    """Applies the per-run MAX_FILES cap, smallest files first."""
    outcomes = list(skipped)
    kept = sorted(reviewable, key=lambda f: f.total_lines)
    if len(kept) > config.MAX_FILES:
        for fd in kept[config.MAX_FILES:]:
            outcomes.append(FileOutcome(fd.new_path, "skipped", "over MAX_FILES limit"))
        kept = kept[: config.MAX_FILES]
    return kept, outcomes
```

- [ ] **Step 4: Update the one existing caller**

In `review_merge_request`, replace `kept, outcomes = select_files(file_diffs)` with:

```python
        reviewable, skipped = partition_reviewable(file_diffs)
        kept, outcomes = select_files(reviewable, skipped)
```

Task 10 rewrites this block again; this keeps the suite green in between.

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest -v`
Expected: PASS. Existing `select_files` tests that pass a mixed list of reviewable and non-reviewable diffs must be updated to call `partition_reviewable` first.

- [ ] **Step 6: Commit**

```bash
git add reviewer/pipeline.py tests/test_pipeline.py
git commit -m "refactor: split partition_reviewable out of select_files"
```

---

### Task 9: Refuse oversized merge requests with one note

**Files:**
- Modify: `reviewer/pipeline.py` (`render_oversized`, oversized branch in `review_merge_request`)
- Test: `tests/test_pipeline_ledger.py`

**Interfaces:**
- Consumes: `config.MAX_MR_FILES`, `ledger.LedgerStore` from Tasks 6-7.
- Produces: `pipeline.render_oversized(count: int) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_ledger.py`:

```python
import pytest

from reviewer import pipeline
from reviewer.diff_parser import FileDiff, Hunk
from reviewer.ledger import Ledger


def fd(path, added=1, hunks=None):
    lines = [" ctx"] + [f"+line{i}" for i in range(added)]
    return FileDiff(
        old_path=path, new_path=path, is_new=False, is_deleted=False,
        is_renamed=False, is_binary=False,
        hunks=hunks or (Hunk(1, 1, tuple(lines)),),
    )


class FakeMR:
    source_branch = "feature/x"
    diff_refs = {"base_sha": "b", "start_sha": "s", "head_sha": "abc1234"}


class Recorder:
    """Captures every comment the pipeline tries to post."""

    def __init__(self):
        self.notes = []
        self.inline = []


@pytest.fixture
def harness(monkeypatch):
    """Wires review_merge_request to fakes and returns (recorder, state)."""
    recorder = Recorder()
    state = {"ledger": Ledger(), "saves": 0}
    mr = FakeMR()

    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_mr", lambda pid, iid: (object(), mr),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.post_note",
        lambda mr, body: recorder.notes.append(body),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.post_inline",
        lambda mr, path, line, body: (recorder.inline.append((path, body)), True)[1],
    )

    class FakeStore:
        def __init__(self):
            self.mr = mr

        @property
        def ledger(self):
            return state["ledger"]

        @ledger.setter
        def ledger(self, value):
            state["ledger"] = value

        @classmethod
        def load(cls, mr):
            return cls()

        def save(self):
            state["saves"] += 1

    monkeypatch.setattr("reviewer.pipeline.LedgerStore", FakeStore)
    monkeypatch.setattr("reviewer.config.SNARK", False)
    return recorder, state


def test_oversized_mr_posts_one_note_and_no_inline(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 3)
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(10)],
    )
    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 1
    assert recorder.inline == []
    assert "10" in recorder.notes[0]
    assert state["ledger"].oversized is True
    assert state["ledger"].posted == 1


def test_oversized_mr_stays_silent_on_the_next_push(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 3)
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(10)],
    )
    pipeline.review_merge_request(1, 1)
    pipeline.review_merge_request(1, 1)
    assert len(recorder.notes) == 1, "the oversized note must not repeat"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_pipeline_ledger.py -v`
Expected: FAIL with `AttributeError: module 'reviewer.pipeline' has no attribute 'LedgerStore'`

- [ ] **Step 3: Add `render_oversized` to `pipeline.py`**

```python
def render_oversized(count: int) -> str:
    """The single comment an over-threshold MR receives."""
    lines = []
    if config.SNARK:
        lines.append(f"_{snark()}_\n")
    lines.append(
        f"**MR завеликий: {count} файлів до рев'ю, поріг — {config.MAX_MR_FILES}.**\n"
    )
    lines.append(
        "Пофайлове рев'ю пропущено. На такому обсязі воно дає сотні коментарів "
        "і нуль користі — розбий MR на менші або рев'юйте руками.\n"
    )
    lines.append("_Це єдиний коментар, який я лишу в цьому MR._")
    return "\n".join(lines)
```

- [ ] **Step 4: Wire the ledger and the oversized branch into `review_merge_request`**

Add to the imports in `pipeline.py`:

```python
from reviewer.ledger import LedgerStore, LedgerUnavailable, hunk_key
```

Replace the opening of the `try` block in `review_merge_request` with:

```python
        project, mr = gitlab_client.fetch_mr(project_id, mr_iid)

        try:
            store = LedgerStore.load(mr)
        except LedgerUnavailable as exc:
            # An unreadable ledger must never be treated as an empty one: that
            # re-reviews the whole MR, which is the flood this prevents.
            logging.error("MR !%s: review state unreadable (%s); skipping", mr_iid, exc)
            return

        if store.ledger.muted and not force:
            logging.info("MR !%s is muted; skipping", mr_iid)
            return
        if force:
            store.ledger = store.ledger.unmute_and_reset()

        if should_skip_branch(getattr(mr, "source_branch", "")):
            summarize_release_mr(project_id, mr_iid, mr, force)
            return

        file_diffs = gitlab_client.fetch_file_diffs(mr)
        head_sha = (getattr(mr, "diff_refs", None) or {}).get("head_sha", "")
        reviewable, skipped = partition_reviewable(file_diffs)

        if len(reviewable) > config.MAX_MR_FILES:
            if not store.ledger.oversized:
                gitlab_client.post_note(mr, render_oversized(len(reviewable)))
                store.ledger = store.ledger.spend(1).mark_oversized()
                logging.info(
                    "MR !%s: %s reviewable files over MAX_MR_FILES=%s; posted one note",
                    mr_iid, len(reviewable), config.MAX_MR_FILES,
                )
            store.ledger = store.ledger.at_head(head_sha)
            store.save()
            return
```

Leave the rest of the existing body below this point for now; Task 10 replaces it.

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_pipeline_ledger.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest -v`
Expected: PASS. Existing `review_merge_request` tests in `tests/test_pipeline.py` now need `reviewer.pipeline.LedgerStore` patched; add the same `FakeStore` fixture there or import it from a shared conftest.

- [ ] **Step 7: Commit**

```bash
git add reviewer/pipeline.py tests/test_pipeline_ledger.py tests/test_pipeline.py
git commit -m "feat: refuse oversized merge requests with a single note"
```

---

### Task 10: Review only unseen hunks, and stay silent when there are none

**Files:**
- Modify: `reviewer/pipeline.py` (`drop_known_hunks`, review loop in `review_merge_request`)
- Test: `tests/test_pipeline_ledger.py`

**Interfaces:**
- Consumes: `ledger.hunk_key`, `ledger.Ledger.record` from Task 6.
- Produces: `pipeline.drop_known_hunks(file_diffs: Sequence[FileDiff], value: Ledger) -> list[FileDiff]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_ledger.py`:

```python
from reviewer.ledger import hunk_key
from reviewer.pipeline import drop_known_hunks


def test_drop_known_hunks_removes_seen_hunks():
    first = Hunk(1, 1, (" ctx", "+one"))
    second = Hunk(9, 9, (" ctx", "+two"))
    diff = fd("a.py", hunks=(first, second))
    value = Ledger().record([hunk_key("a.py", first)])
    fresh = drop_known_hunks([diff], value)
    assert len(fresh) == 1
    assert fresh[0].hunks == (second,)


def test_drop_known_hunks_drops_fully_seen_files():
    only = Hunk(1, 1, (" ctx", "+one"))
    value = Ledger().record([hunk_key("a.py", only)])
    assert drop_known_hunks([fd("a.py", hunks=(only,))], value) == []


def test_drop_known_hunks_survives_a_rebase():
    """Same content at a new line number must still count as seen."""
    lines = (" ctx", "+one")
    value = Ledger().record([hunk_key("a.py", Hunk(1, 1, lines))])
    rebased = fd("a.py", hunks=(Hunk(400, 400, lines),))
    assert drop_known_hunks([rebased], value) == []


def test_unchanged_diff_posts_absolutely_nothing(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    diff = fd("a.py")
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: [diff])
    state["ledger"] = Ledger().record(
        [hunk_key("a.py", h) for h in diff.hunks]
    )
    pipeline.review_merge_request(1, 1)
    assert recorder.notes == []
    assert recorder.inline == []


def test_reviewed_hunks_are_recorded(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult(
            "**🔴 [BLOCKER]** boom", "stop", 0, 0, 0.0,
        ),
    )
    diff = fd("a.py")
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: [diff])
    pipeline.review_merge_request(1, 1)
    assert hunk_key("a.py", diff.hunks[0]) in state["ledger"].hunks
    assert state["saves"] >= 2, "the ledger must be saved incrementally"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_pipeline_ledger.py -k "drop_known or unchanged or recorded" -v`
Expected: FAIL with `ImportError: cannot import name 'drop_known_hunks'`

- [ ] **Step 3: Add `drop_known_hunks` to `pipeline.py`**

Add `from dataclasses import dataclass, replace` to the imports, then:

```python
def drop_known_hunks(
    file_diffs: Sequence[FileDiff], value: "Ledger",
) -> list[FileDiff]:
    """Returns the diffs with already-reviewed hunks removed.

    A file whose every hunk is known disappears from the result entirely.
    """
    known = set(value.hunks)
    fresh: list[FileDiff] = []
    for fd in file_diffs:
        hunks = tuple(h for h in fd.hunks if hunk_key(fd.new_path, h) not in known)
        if hunks:
            fresh.append(replace(fd, hunks=hunks))
    return fresh
```

Add `Ledger` to the `reviewer.ledger` import line.

- [ ] **Step 4: Replace the review loop in `review_merge_request`**

Replace everything from the old `fingerprint = gitlab_client.diff_fingerprint(...)` line down to the end of the `try` block with:

```python
        fresh = drop_known_hunks(reviewable, store.ledger)
        if not fresh:
            # Every hunk has been reviewed already. Say nothing at all: this is
            # what makes a container restart or a no-op push cost zero comments.
            logging.info("MR !%s: no unreviewed hunks; staying silent", mr_iid)
            store.ledger = store.ledger.at_head(head_sha)
            store.save()
            return

        kept, outcomes = select_files(fresh, skipped)
        voice = VoiceState()
        review_state = ReviewState()

        for file_diff in kept:
            if time.monotonic() - started > config.MR_TIMEOUT_S:
                outcomes.append(FileOutcome(
                    file_diff.new_path, "skipped",
                    f"MR deadline of {config.MR_TIMEOUT_S}s reached",
                ))
                continue
            if not review_state.open:
                outcomes.append(FileOutcome(
                    file_diff.new_path, "skipped", "review backend rate limited",
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
                outcome = review_file(mr, file_diff, context, voice, review_state)
            except Exception as exc:
                logging.error("Review failed for %s: %s", file_diff.new_path, exc)
                outcomes.append(FileOutcome(file_diff.new_path, "error", str(exc)[:120]))
                continue

            outcomes.append(outcome)
            if outcome.status in ("reviewed", "clean"):
                # Only settled files are recorded. An errored or rate-limited
                # file must be retried on the next push, so its hunks stay out.
                store.ledger = store.ledger.record(
                    hunk_key(file_diff.new_path, h) for h in file_diff.hunks
                )
            if outcome.status == "reviewed":
                store.ledger = store.ledger.spend(1)
            # Saved per file, not once at the end: a crash mid-run would
            # otherwise leave comments posted but unrecorded, and the next push
            # would post every one of them again.
            store.save()

        gitlab_client.post_note(mr, render_summary(outcomes))
        store.ledger = store.ledger.spend(1).at_head(head_sha)
        store.save()
        logging.info("MR !%s reviewed in %.1fs", mr_iid, time.monotonic() - started)
```

Also add `ChatResult` to `pipeline`'s exported names by keeping the existing `from reviewer.chat_types import FINDING_TAGS, LGTM_TEXT, ChatResult` import — the test above references `pipeline.ChatResult`.

- [ ] **Step 5: Remove the now-dead review-path dedupe**

`diff_fingerprint` and the `dedupe` cache are still used by `summarize_release_mr`; leave both in place. Only the review path stops calling them. Verify with:

Run: `grep -n "dedupe\|diff_fingerprint" reviewer/pipeline.py`
Expected: matches only inside `summarize_release_mr` and the module-level `dedupe = DedupeCache()`.

- [ ] **Step 6: Run the tests**

Run: `python3 -m pytest -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add reviewer/pipeline.py tests/test_pipeline_ledger.py
git commit -m "feat: review only unseen hunks and stay silent when there are none"
```

---

### Task 11: Spend against a lifetime comment budget

**Files:**
- Modify: `reviewer/pipeline.py` (`render_budget_exhausted`, budget checks in the review loop)
- Test: `tests/test_pipeline_ledger.py`

**Interfaces:**
- Consumes: `config.MR_COMMENT_BUDGET`, `Ledger.remaining`, `Ledger.mute` from Task 6.
- Produces: `pipeline.render_budget_exhausted() -> str`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_ledger.py`:

```python
def test_budget_exhaustion_posts_one_final_note_and_mutes(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.MR_COMMENT_BUDGET", 3)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult(
            "**🔴 [BLOCKER]** boom", "stop", 0, 0, 0.0,
        ),
    )
    monkeypatch.setattr(
        "reviewer.gitlab_client.fetch_file_diffs",
        lambda mr: [fd(f"f{i}.py") for i in range(10)],
    )
    pipeline.review_merge_request(1, 1)

    assert len(recorder.inline) == 2, "budget 3 leaves 2 inline plus a final note"
    assert len(recorder.notes) == 1
    assert "ліміт" in recorder.notes[0].lower()
    assert state["ledger"].muted is True


def test_muted_mr_ignores_a_plain_push(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    state["ledger"] = Ledger(muted=True)
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: [fd("a.py")])
    pipeline.review_merge_request(1, 1)
    assert recorder.notes == []
    assert recorder.inline == []


def test_force_review_clears_mute_and_resets_budget(harness, monkeypatch):
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.MR_COMMENT_BUDGET", 30)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult("[LGTM]", "stop", 0, 0, 0.0),
    )
    state["ledger"] = Ledger(muted=True, posted=30)
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: [fd("a.py")])
    pipeline.review_merge_request(1, 1, force=True)
    assert state["ledger"].muted is False
    assert len(recorder.notes) == 1
```

Also append the two end-to-end guards the spec's Error handling and
rate-limit sections call for:

```python
def test_unreadable_ledger_skips_the_run(harness, monkeypatch):
    """Fail closed. An empty-ledger fallback would re-review the whole MR."""
    recorder, state = harness
    from reviewer.ledger import LedgerUnavailable

    def _boom(mr):
        raise LedgerUnavailable("gitlab is down")

    monkeypatch.setattr("reviewer.pipeline.LedgerStore.load", staticmethod(_boom))
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: [fd("a.py")])
    pipeline.review_merge_request(1, 1)
    assert recorder.notes == []
    assert recorder.inline == []


def test_rate_limited_files_are_not_recorded(harness, monkeypatch):
    """A rate-limited file must be retried on the next push, so its hunks
    must not enter the ledger."""
    recorder, state = harness
    monkeypatch.setattr("reviewer.config.MAX_MR_FILES", 60)
    monkeypatch.setattr("reviewer.config.MR_COMMENT_BUDGET", 30)
    monkeypatch.setattr("reviewer.config.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "reviewer.pipeline.review_chat",
        lambda system, user, deadline_s: pipeline.ChatResult("", "ratelimit", 0, 0, 0.0),
    )
    diffs = [fd(f"f{i}.py") for i in range(5)]
    monkeypatch.setattr("reviewer.gitlab_client.fetch_file_diffs", lambda mr: diffs)
    pipeline.review_merge_request(1, 1)

    assert state["ledger"].hunks == (), "no hunk may be recorded on a rate limit"
    assert recorder.inline == []
    summary = recorder.notes[0]
    assert "rate limited" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_pipeline_ledger.py -k "budget or muted or force or unreadable or rate_limited" -v`
Expected: FAIL — `test_budget_exhaustion_posts_one_final_note_and_mutes` gets 10 inline comments, not 2.

- [ ] **Step 3: Add `render_budget_exhausted`**

```python
def render_budget_exhausted() -> str:
    """The closing comment when an MR has used its whole budget."""
    lines = []
    if config.SNARK:
        lines.append(f"_{snark()}_\n")
    lines.append(
        f"**Ліміт вичерпано: {config.MR_COMMENT_BUDGET} коментарів у цьому MR.**\n"
    )
    lines.append(
        "Далі мовчу до мержу. Розгреби те, що вже написав, а тоді кинь `/review` "
        "у коментар — лічильник обнулиться.\n"
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Reserve the last budget slot in the review loop**

In the `for file_diff in kept:` loop added in Task 10, insert this check directly after the `if not review_state.open:` block:

```python
            if store.ledger.remaining() <= 1:
                # The last slot is reserved for the closing note, so the run can
                # always tell the reader why it stopped.
                outcomes.append(FileOutcome(
                    file_diff.new_path, "skipped", "MR comment budget reached",
                ))
                continue
```

Then replace the closing `gitlab_client.post_note(mr, render_summary(outcomes))` block with:

```python
        if store.ledger.remaining() <= 1:
            gitlab_client.post_note(mr, render_budget_exhausted())
            store.ledger = store.ledger.spend(1).mute().at_head(head_sha)
            logging.warning(
                "MR !%s hit MR_COMMENT_BUDGET=%s; muted until /review",
                mr_iid, config.MR_COMMENT_BUDGET,
            )
        else:
            gitlab_client.post_note(mr, render_summary(outcomes))
            store.ledger = store.ledger.spend(1).at_head(head_sha)
        store.save()
```

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_pipeline_ledger.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add reviewer/pipeline.py tests/test_pipeline_ledger.py
git commit -m "feat: cap comments per merge request with a lifetime budget"
```

---

### Task 12: Kill switch and `/sidorovich stop`

**Files:**
- Modify: `review_server.py:39-80`, `reviewer/pipeline.py` (append `mute_merge_request`)
- Test: `tests/test_webhook.py`

**Interfaces:**
- Consumes: `config.SIDOROVICH_ENABLED` from Task 6, `LedgerStore` from Task 7.
- Produces:
  - `review_server.ReviewJob` — frozen dataclass with `project_id: int`, `mr_iid: int`, `force: bool`, `command: str`
  - `review_server.should_review(event_type: str, data: dict) -> ReviewJob | None`
  - `pipeline.mute_merge_request(project_id: int, mr_iid: int) -> None`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_webhook.py`:

```python
import review_server
from review_server import ReviewJob, should_review


def _note_event(text):
    return {
        "object_attributes": {"noteable_type": "MergeRequest", "note": text},
        "project": {"id": 7},
        "merge_request": {"iid": 42},
    }


def test_kill_switch_ignores_every_event(monkeypatch):
    monkeypatch.setattr("reviewer.config.SIDOROVICH_ENABLED", False)
    event = {"object_attributes": {"action": "open", "iid": 42}, "project": {"id": 7}}
    assert should_review("Merge Request Hook", event) is None


def test_stop_command_produces_a_mute_job(monkeypatch):
    monkeypatch.setattr("reviewer.config.SIDOROVICH_ENABLED", True)
    job = should_review("Note Hook", _note_event("/sidorovich stop"))
    assert job == ReviewJob(7, 42, False, "mute")


def test_stop_command_wins_over_review(monkeypatch):
    monkeypatch.setattr("reviewer.config.SIDOROVICH_ENABLED", True)
    job = should_review("Note Hook", _note_event("/review then /sidorovich stop"))
    assert job.command == "mute"


def test_review_command_produces_a_forced_review_job(monkeypatch):
    monkeypatch.setattr("reviewer.config.SIDOROVICH_ENABLED", True)
    job = should_review("Note Hook", _note_event("please /review this"))
    assert job == ReviewJob(7, 42, True, "review")


def test_push_produces_an_unforced_review_job(monkeypatch):
    monkeypatch.setattr("reviewer.config.SIDOROVICH_ENABLED", True)
    event = {
        "object_attributes": {"action": "update", "iid": 42, "oldrev": "abc"},
        "project": {"id": 7},
    }
    assert should_review("Merge Request Hook", event) == ReviewJob(7, 42, False, "review")


def test_mute_and_review_do_not_coalesce(monkeypatch):
    """A queued mute must not be replaced by a review for the same MR."""
    monkeypatch.setattr("reviewer.config.SIDOROVICH_ENABLED", True)
    mute = should_review("Note Hook", _note_event("/sidorovich stop"))
    review = should_review("Note Hook", _note_event("/review"))
    assert review_server.queue_key(mute) != review_server.queue_key(review)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_webhook.py -k "kill_switch or stop_command or coalesce" -v`
Expected: FAIL with `ImportError: cannot import name 'ReviewJob' from 'review_server'`

- [ ] **Step 3: Rewrite `should_review` in `review_server.py`**

Add `from dataclasses import dataclass` to the imports, then replace the whole `should_review` function and `_handle`:

```python
@dataclass(frozen=True)
class ReviewJob:
    project_id: int
    mr_iid: int
    force: bool
    command: str      # "review" | "mute"


def queue_key(job: ReviewJob) -> str:
    """Coalescing key. The command is part of it so a queued mute is never
    replaced by a review for the same merge request."""
    return f"{job.project_id}:{job.mr_iid}:{job.command}"


def _handle(job: ReviewJob) -> None:
    if job.command == "mute":
        mute_merge_request(job.project_id, job.mr_iid)
        return
    review_merge_request(job.project_id, job.mr_iid, force=job.force)


review_queue = start_worker(_handle)


def should_review(event_type: str, data: dict) -> ReviewJob | None:
    """Returns the job this event warrants, or None.

    force is True for a manual '/review' comment, which must re-run even though
    the diff itself is by definition unchanged. It is False for the automatic
    Merge Request Hook paths, which defer to the ledger.
    """
    if not config.SIDOROVICH_ENABLED:
        return None

    attrs = data.get("object_attributes", {})

    if event_type == "Note Hook":
        if attrs.get("noteable_type") != "MergeRequest":
            return None
        note = (attrs.get("note") or "").lower()
        project_id = data["project"]["id"]
        mr_iid = data["merge_request"]["iid"]
        # Checked before /review so a comment containing both silences the bot.
        if "/sidorovich stop" in note:
            return ReviewJob(project_id, mr_iid, False, "mute")
        if "/review" in note:
            return ReviewJob(project_id, mr_iid, True, "review")
        return None

    if event_type == "Merge Request Hook":
        action = attrs.get("action")
        if action in ("open", "reopen"):
            return ReviewJob(data["project"]["id"], attrs["iid"], False, "review")
        # 'update' fires on title, description and label edits too. GitLab sets
        # oldrev only when new commits arrived, so it is our new-commits signal.
        if action == "update" and attrs.get("oldrev"):
            return ReviewJob(data["project"]["id"], attrs["iid"], False, "review")
        return None

    return None
```

Update the import at the top from `from reviewer.pipeline import review_merge_request` to:

```python
from reviewer.pipeline import mute_merge_request, review_merge_request
```

- [ ] **Step 4: Update the webhook route**

Replace the tail of the `webhook()` function:

```python
    job = should_review(request.headers.get("X-Gitlab-Event"), request.json or {})
    if not job:
        return jsonify({"message": "Ignored event"}), 200

    if review_queue.submit(queue_key(job), job):
        return jsonify({"message": "Review queued", "depth": review_queue.size()}), 202
    return jsonify({"error": "Review queue full"}), 503
```

Note that `ReviewQueue.submit` is typed `job: tuple`; widen the annotation to `job: object` in `reviewer/queue.py` — it only ever stores and returns the value.

- [ ] **Step 5: Add `mute_merge_request` to `pipeline.py`**

```python
def mute_merge_request(project_id: int, mr_iid: int) -> None:
    """Silences the bot for one MR. Acknowledged by editing the state note.

    Deliberately posts no comment: a mute that costs a comment defeats itself.
    """
    try:
        _, mr = gitlab_client.fetch_mr(project_id, mr_iid)
        store = LedgerStore.load(mr)
        store.ledger = store.ledger.mute()
        store.save()
        logging.info("MR !%s muted by /sidorovich stop", mr_iid)
    except Exception as exc:
        logging.error("Could not mute MR !%s: %s", mr_iid, exc)
```

- [ ] **Step 6: Add the kill switch to the startup log**

In `review_server.py`, extend the existing startup `logging.info` call with the new settings so a misconfigured deployment is visible immediately. Add after the Sidorovich LLM line:

```python
logging.info(
    "Review LLM: openrouter:%s%s (context=%s out=%s)",
    config.OPENROUTER_REVIEW_MODEL,
    " -> " + " -> ".join(config.OPENROUTER_REVIEW_FALLBACK_MODELS)
    if config.OPENROUTER_REVIEW_FALLBACK_MODELS else "",
    config.REVIEW_CONTEXT_TOKENS, config.REVIEW_MAX_OUTPUT_TOKENS,
)
logging.info(
    "Guardrails: enabled=%s max_mr_files=%s comment_budget=%s ledger_max_hunks=%s",
    config.SIDOROVICH_ENABLED, config.MAX_MR_FILES,
    config.MR_COMMENT_BUDGET, config.LEDGER_MAX_HUNKS,
)
```

- [ ] **Step 7: Run the full suite**

Run: `python3 -m pytest -v`
Expected: PASS. Existing `tests/test_webhook.py` cases asserting a tuple return value must be updated to compare `ReviewJob` instances.

- [ ] **Step 8: Commit**

```bash
git add review_server.py reviewer/pipeline.py reviewer/queue.py tests/test_webhook.py
git commit -m "feat: add kill switch and /sidorovich stop"
```

---

### Task 13: Wire configuration and documentation

**Files:**
- Modify: `docker-compose.yml:6-33`, `.env.example`, `README.md`
- Test: manual verification only — this task ships no code paths.

**Interfaces:**
- Consumes: every setting added in Tasks 3 and 6.
- Produces: nothing importable.

- [ ] **Step 1: Add the settings to `docker-compose.yml`**

In the `app.environment` list, after the existing `SIDOROVICH_OLLAMA_FALLBACK` entry:

```yaml
      - OPENROUTER_REVIEW_MODEL=${OPENROUTER_REVIEW_MODEL:-poolside/laguna-s-2.1}
      - OPENROUTER_REVIEW_FALLBACK_MODELS=${OPENROUTER_REVIEW_FALLBACK_MODELS:-}
      - REVIEW_CONTEXT_TOKENS=${REVIEW_CONTEXT_TOKENS:-262144}
      - REVIEW_MAX_OUTPUT_TOKENS=${REVIEW_MAX_OUTPUT_TOKENS:-4096}
      - SIDOROVICH_ENABLED=${SIDOROVICH_ENABLED:-true}
      - MAX_MR_FILES=${MAX_MR_FILES:-60}
      - MR_COMMENT_BUDGET=${MR_COMMENT_BUDGET:-30}
      - LEDGER_MAX_HUNKS=${LEDGER_MAX_HUNKS:-2000}
```

- [ ] **Step 2: Add the settings to `.env.example`**

Append, matching the file's existing comment style:

```bash
# --- Review backend ---
# Code review runs on OpenRouter. OPENROUTER_API_KEY is now required for review,
# not just for the Sidorovich voice.
OPENROUTER_REVIEW_MODEL=poolside/laguna-s-2.1
# Comma-separated reroute targets tried in order. Empty is fine on a paid model.
OPENROUTER_REVIEW_FALLBACK_MODELS=
# The model allows 1048576; capped here because prompt tokens are billed.
REVIEW_CONTEXT_TOKENS=262144
REVIEW_MAX_OUTPUT_TOKENS=4096

# --- Comment guardrails ---
# false silences the bot everywhere without touching the GitLab webhook.
SIDOROVICH_ENABLED=true
# Reviewable files above which the MR gets one note instead of a per-file review.
MAX_MR_FILES=60
# Comments the bot may post in one MR, ever. /review resets the counter.
MR_COMMENT_BUDGET=30
LEDGER_MAX_HUNKS=2000
```

- [ ] **Step 3: Document the behaviour in `README.md`**

Append this section, and update any existing README passage that describes code
review as running on local Ollama — it no longer does.

```markdown
## Comment guardrails

The bot keeps a per-MR ledger of the diff hunks it has already reviewed, stored
in a single 🔒 state note it creates once and edits in place. **Do not edit or
delete that note by hand** — deleting it makes the bot forget the MR and review
it from scratch.

| Situation | What the bot does |
|---|---|
| More than `MAX_MR_FILES` (60) reviewable files | Posts exactly one note asking for a smaller MR. No per-file review, ever, for that MR. |
| A push whose hunks were all reviewed already | Posts nothing at all. |
| A push with new hunks | Reviews only the new hunks. |
| `MR_COMMENT_BUDGET` (30) comments reached | Posts one closing note and goes quiet until merge. |
| A rebase or force-push | Unchanged hunk content is not re-reviewed. Hunk keys hash content, not line numbers. |

### Commands

- `/review` in an MR comment — re-review now, clear any mute, reset the comment
  counter to zero. It does **not** override the `MAX_MR_FILES` threshold.
- `/sidorovich stop` in an MR comment — silence the bot for that MR only. It
  acknowledges by editing the state note, not by posting a comment. `/review`
  lifts it.

### Global off switch

`SIDOROVICH_ENABLED=false` makes every webhook a no-op. Use it instead of
disabling the GitLab webhook, which silences the bot for every project on the
instance.

## Review model

Code review runs on OpenRouter (`OPENROUTER_REVIEW_MODEL`, default
`poolside/laguna-s-2.1`). **`OPENROUTER_API_KEY` is now required for review**,
not only for the Sidorovich voice.

A forty-file MR costs roughly two cents. The `:free` variant exists but shares a
saturated pool across all OpenRouter users and rate-limits hard at forty calls
per MR; set `OPENROUTER_REVIEW_MODEL=poolside/laguna-s-2.1:free` only if you
would rather have 429s than a bill.

Ollama is no longer used for review. It remains available for the Sidorovich
voice behind `SIDOROVICH_OLLAMA_FALLBACK`, and `OLLAMA_NUM_CTX` /
`OLLAMA_NUM_PREDICT` govern only that path. The review prompt budget comes from
`REVIEW_CONTEXT_TOKENS` and `REVIEW_MAX_OUTPUT_TOKENS`.
```

- [ ] **Step 4: Verify the compose file parses**

Run: `docker compose config --quiet && echo OK`
Expected: `OK` with no output above it.

- [ ] **Step 5: Verify every setting is reachable**

Run:

```bash
python3 -c "
from reviewer import config
for name in ('OPENROUTER_REVIEW_MODEL', 'OPENROUTER_REVIEW_FALLBACK_MODELS',
             'REVIEW_CONTEXT_TOKENS', 'REVIEW_MAX_OUTPUT_TOKENS',
             'SIDOROVICH_ENABLED', 'MAX_MR_FILES', 'MR_COMMENT_BUDGET',
             'LEDGER_MAX_HUNKS'):
    print(name, '=', getattr(config, name))
"
```

Expected: eight lines, with `OPENROUTER_REVIEW_MODEL = poolside/laguna-s-2.1` and `MAX_MR_FILES = 60`.

- [ ] **Step 6: Run the full suite one last time**

Run: `python3 -m pytest -v`
Expected: PASS, no skips.

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml .env.example README.md
git commit -m "docs: document review backend and comment guardrails"
```

---

## Verification After Task 13

Confirm the two scenarios from the spec, using the fakes already built in `tests/test_pipeline_ledger.py`:

1. **The incident.** 300 reviewable files against `MAX_MR_FILES=60` produces exactly one note and zero inline comments, and a second run produces nothing. Covered by `test_oversized_mr_posts_one_note_and_no_inline` and `test_oversized_mr_stays_silent_on_the_next_push`.

2. **Repeated pushes.** A 30-file MR pushed repeatedly posts at most `MR_COMMENT_BUDGET` comments in total, and an unchanged diff posts none. Covered by `test_budget_exhaustion_posts_one_final_note_and_mutes` and `test_unchanged_diff_posts_absolutely_nothing`.

Before deploying, run the existing diagnostic against the new model to confirm the key can reach it:

```bash
OPENROUTER_MODEL=poolside/laguna-s-2.1 python3 scripts/openrouter_check.py --env .env
```

Expected: `status: 200` and a `served by: poolside/laguna-s-2.1` line.
