# AI Code Reviewer — Speedup Refactor Design

**Date:** 2026-08-13
**Status:** Approved for planning
**Target host:** Mac Mini M4, 16 GB unified memory, Ollama on host, Flask app in Docker

## Problem

Reviews on medium and large merge requests take minutes, back up behind each
other, and frequently time out or return truncated output. Large MRs silently
lose files.

The user asked whether a Go or Rust rewrite, or a third-party diff parser,
would help.

## Diagnosis

Language is not the bottleneck. On this hardware a medium MR breaks down as:

| Stage | Time | Cause |
|---|---|---|
| GitLab file fetches, 20 files, sequential | ~10 s | one blocking call per file |
| Prompt eval, ~7K tokens at ~150 tok/s | ~47 s | `num_batch: 128` |
| Generation, 1024 tokens at ~14 tok/s | ~73 s | 12B QAT on M4 GPU |
| Doubled by the truncation-retry bug | **~240 s** | see B2 below |

Python contributes roughly 50 ms. A Rust or Go rewrite would change total wall
clock by well under one percent. **Rejected.**

A diff parser is worth adopting, but for token reduction and correct line
numbers — not for parsing speed.

### Confirmed defects in the current code

**B1 — `num_batch: 128` hardcoded** (`review_server.py:333`)
Metal's default is 512. At 128, prompt evaluation runs 2–4x slower. Primary
cause of timeouts on `qwen2.5-coder:7b-instruct-q4_K_M`.

**B2 — every clean MR runs inference twice** (`review_server.py:388`)

```python
def _response_looks_truncated(review_text, stats):
    if stats.get("eval_count", 0) < MIN_OUTPUT_TOKENS:   # 30
        return True                                       # fires first
    ...
    if "[LGTM]" in review_text:
        return False                                      # unreachable
```

An `[LGTM]` response is ~8 tokens, so it is always classified as truncated and
triggers a second full inference.

**B3 — `--workers 2` defeats the review lock** (`Dockerfile:14`)
Gunicorn forks two processes. `review_lock = threading.Lock()` is process-local
and does not span forks. Two webhooks arriving on different workers run two
concurrent Ollama inferences on a 16 GB machine. The lock has never provided
the protection its comment claims.

**B4 — hunk-header regex misses single-line hunks** (`review_server.py:141`)

```python
r'@@ -\d+,\d+ \+(\d+)(?:,\d+)? @@'
```

The old-side `,\d+` is mandatory, but git emits `@@ -12 +12,3 @@` for
single-line ranges. No match means file context is silently dropped.

**B5 — `annotate_diff_for_ai` is dead code** (`review_server.py:111`)
Defined in commit a95b296, never called. The annotations never reached the model.

**B6 — `MAX_PROMPT_CHARS` has no effect**
The effective cap is derived from `num_ctx`:
`(8192 − 500 − system − 128) × 3 ≈ 21,000` chars. Raising the variable to
120,000 in commit f408860 changed nothing.

**B7 — `update` events re-review unchanged diffs**
`obj.get('action') in ['open', 'reopen', 'update']` fires on title edits,
description edits, and label changes, each costing a full review.

## Goals

1. Cut per-MR latency, especially time to first comment.
2. Stop reviews hanging or timing out.
3. Eliminate queue backup from redundant and concurrent work.
4. Make large MRs complete rather than silently dropping files.

## Non-goals

- Rewriting in another language.
- Running concurrent inference. `OLLAMA_NUM_PARALLEL=1` stays. On 16 GB,
  concurrency causes swap thrash; throughput comes from doing less work.
- Changing the deployment story. Docker Compose and Cloudflare Tunnel setup
  are unchanged.

## Architecture

Split the 510-line single file into bounded, independently testable modules.

```
review_server.py          # Flask routes only, ~60 lines
reviewer/
  config.py               # all env parsing, single source of truth
  diff_parser.py          # unified diff -> FileDiff / Hunk
  filters.py              # is_reviewable(path) predicate
  gitlab_client.py        # MR changes, file content, posting
  prompt.py               # system prompt, per-file builder, token budget
  ollama_client.py        # streaming chat, wall-clock abort, stats
  pipeline.py             # per-file orchestration, degradation ladder
  queue.py                # single-consumer FIFO, dedupe, coalescing
  memes.py                # meme_phrases easter egg, relocated unchanged
```

`diff_parser`, `filters`, and `prompt` are pure functions over strings and are
tested without a running Ollama or GitLab.

### Data model

```python
@dataclass(frozen=True)
class Hunk:
    old_start: int
    new_start: int
    lines: list[str]                                    # +/-/space preserved
    def added_lines(self) -> list[tuple[int, str]]: ...  # (new_lineno, text)

@dataclass(frozen=True)
class FileDiff:
    old_path: str
    new_path: str
    is_new: bool
    is_deleted: bool
    is_renamed: bool
    is_binary: bool
    hunks: list[Hunk]
```

Correct line numbers are the reason for a real parser. They enable inline
discussions anchored to the changed line, which in turn lets each file post its
findings as soon as that file is reviewed.

## Review pipeline

```
webhook -> validate -> enqueue(project_id, mr_iid)
worker  -> fetch changes -> parse -> filter -> order -> per-file review
        -> post inline -> summary note
```

### Filtering

Dropped before any token is spent: deleted files, binary files, renames with no
content change, and the skiplist — `*.lock`, `package-lock.json`, `*.min.*`,
`__snapshots__/`, `vendor/`, `dist/`, `*.pb.go`, `*.generated.*`, `*.svg`, and
image formats.

### Ordering

Smallest files first, so the first inline comment appears in roughly 10–15
seconds rather than after the entire MR completes.

### Degradation ladder

Applied per file. No level is silent; every skip is named in the summary note.

| Level | Action | Trigger |
|---|---|---|
| L0 | hunks plus `CONTEXT_WINDOW` lines of surrounding file context | fits budget |
| L1 | hunks only, no file context | L0 over budget |
| L2 | one request per hunk | file still over budget at L1 |
| L3 | skip and record reason | single hunk over budget |

`INCLUDE_FILE_CONTEXT` defaults to `false`, so the ladder normally **starts at
L1** and L0 is never attempted. Setting it to `true` enables L0 as the entry
point, at the cost of roughly doubled input tokens and one extra GitLab file
fetch per file. L0 also requires fetching full file content, so with the
default off, the GitLab file-content fetches disappear entirely.

Budget per file: the entire remaining context window after the system prompt
and output reservation — `input_token_budget()` returns about 7119 input
tokens with the shipped defaults (`OLLAMA_NUM_CTX=8192`,
`OLLAMA_NUM_PREDICT=320`), not the ~1500 tokens originally estimated here.
Consequence: L2 and L3 are reached only by genuinely enormous single files or
hunks, and one large file can consume a near-full-context prompt evaluation
by itself, at the expense of files queued behind it.

### Prompt construction

Removed lines are **not sent to the model**. `annotate_diff_for_ai` (B5) was
intended to stop the model complaining about deleted code by appending a marker
to each `-` line, at roughly 8 tokens per line. Omitting removed lines entirely
is cheaper and stricter: the model cannot hallucinate about code it never sees.
Input drops around 30 percent on edit-heavy diffs.

The existing system prompt is retained. Its bans on unused-variable,
duplicate-block, and style complaints are well tuned and address real failure
modes of small local models. It is rescoped from whole-MR to single-file.

## Scheduling

- `queue.Queue(maxsize=32)` with one daemon consumer thread replaces
  `review_lock`. Same serialization, with backpressure and observability.
- **Dedupe:** key is `sha256(project_id, mr_iid, concatenated diff bodies)`,
  held in a 256-entry LRU. An unchanged diff costs nothing.
- **Coalescing:** if the same `(project_id, mr_iid)` is queued and not yet
  started, replace it with the newer job. Stale pushes are not reviewed.
- **Webhook tightening:** for `Merge Request Hook`, act only when
  `object_attributes.oldrev` is present, indicating real new commits. Title,
  description, and label edits carry no `oldrev` and are ignored. This
  addresses B7.
- **Gunicorn:** `--workers 1 --threads 4`. Required for correctness — the queue
  and dedupe cache are in-process, and B3 shows multiple workers break
  serialization.

## Timeouts

- **Per file:** 90 s. `stream=True` lets the client abort by closing the
  connection instead of blocking for the full request timeout.
- **Per MR:** 480 s. Remaining files are marked skipped in the summary; the
  review still posts.

## Posting

Findings post as inline discussions via `mr.discussions.create`, with
`position` built from `mr.diff_refs` and the hunk's new-side line number. On a
400 response, fall back to a plain MR note.

A final summary note reports: files reviewed, files skipped with reasons, and
files that errored.

## Error handling

Per-file isolation. A failure on one file never aborts the MR review. Failures
are counted and reported in the summary.

## Configuration changes

| Key | Current | New | Reason |
|---|---|---|---|
| `num_batch` | 128, hardcoded | **512**, env-tunable | B1 |
| `OLLAMA_NUM_PREDICT` | 1024 | **320** | per-file output is shorter |
| `INCLUDE_FILE_CONTEXT` | implicit `true` | **`false`** | doubled input for marginal gain |
| `CONTEXT_WINDOW` | 25 | **15** | only used when L0 is enabled |
| `MAX_PROMPT_CHARS` | 120000 | **removed** | B6 — dead |
| `MIN_OUTPUT_TOKENS` | 30 | **removed** | B2 — replaced by finish-reason check |
| `PER_FILE_TIMEOUT_S` | — | 90 | new |
| `MR_TIMEOUT_S` | — | 480 | new |
| `MAX_FILES` | — | 40 | new; remainder listed in summary |
| gunicorn workers | 2 | **1**, `--threads 4` | B3 |

Truncation detection replaces the `eval_count` heuristic (B2) with Ollama's own
`done_reason` field: `length` means truncated, `stop` means complete.

The unused `ollama` package is removed from `requirements.txt`; the code calls
the HTTP API through `requests` directly.

## Model

No model change is mandated. Reported instability on
`qwen2.5-coder:7b-instruct-q4_K_M` is attributed to B1 and B3 rather than to
the model. After this refactor, both `qwen2.5-coder:7b` and the current
12B QAT model should be re-benchmarked on the same MR; 7B is expected to be
roughly twice as fast at generation.

## Testing

pytest, no network access required.

- **Parser golden fixtures:** single-line hunks (`@@ -12 +12,3 @@`), renames,
  new files, deleted files, binary markers, no-newline-at-EOF, multi-hunk
  files, and hunk headers containing `@@` inside the trailing context text.
- **Filter tests:** each skiplist pattern, plus paths that must not be skipped.
- **Budget ladder tests:** each of L0 through L3 is reachable and reports
  correctly.
- **Truncation detection:** `done_reason` of `stop` and `length`.
- **Dedupe and coalescing:** identical diff dropped; newer job replaces older.

## Expected outcome

Medium MR, 20 files:

| Metric | Current | After |
|---|---|---|
| Time to first comment | ~240 s, or timeout | ~15 s |
| Full review | timeout, or silent skips | ~90 s, complete |
| Repeat `update` event | full re-review | 0 s |
| Concurrent inference risk | present (B3) | eliminated |

## Rollout

`review_server.py` remains the entry point. `Dockerfile` changes only the
gunicorn worker flags. `docker-compose.yml` and the Cloudflare Tunnel
configuration are unchanged. Deployment remains
`docker compose up -d --force-recreate app`.
