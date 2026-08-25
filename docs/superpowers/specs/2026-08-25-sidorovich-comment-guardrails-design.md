# Sidorovich comment guardrails and review backend migration — design

Date: 2026-08-25
Status: approved, awaiting implementation plan

## Problem

A merge request touching 300+ files drew 592 bot comments. The flood stopped
only because the GitLab webhook was disabled by hand — a change that silences
the bot for every project on the instance, not just the offending MR.

### Why 592 and not 41

A single review run is already capped: `MAX_FILES` (40) inline comments plus
one summary note. 592 comments means roughly fifteen runs against the same MR.

Five separate gaps combine to produce that:

1. **Every push starts a full run.** `should_review` queues a review for any
   `Merge Request Hook` with `action == "update"` and an `oldrev`
   (`review_server.py:57`). A large MR under active development pushes often.
2. **No idempotency.** `review_file` always creates *new* comments
   (`pipeline.py:135`). Nothing reads what the bot already said, updates it, or
   resolves it.
3. **Dedupe only catches byte-identical diffs.** `diff_fingerprint` hashes the
   whole MR including hunk line numbers (`gitlab_client.py:118`). One new commit
   changes the digest, so the entire MR is reviewed and posted again.
4. **Dedupe state is in-process.** `DedupeCache` lives in memory. With
   `restart: always` in `docker-compose.yml`, a container restart erases it and
   an unchanged diff is reviewed again from scratch.
5. **File selection is stable.** `select_files` sorts by `total_lines` and keeps
   the smallest 40 (`pipeline.py:75`). Every run picks the *same* 40 files, so
   fifteen runs produce fifteen copies of the same comments.

There is a per-run cap. There is no cap across the lifetime of an MR, and no
size threshold above which per-file review stops making sense at all.

## Goals

- A 300-file MR produces one comment, not 592.
- A 30-file MR pushed fifteen times does not produce 450 comments.
- A container restart never causes a re-review of an unchanged diff.
- Stopping the bot never requires touching the instance-wide webhook.
- Code review runs on OpenRouter `poolside/laguna-s-2.1` instead of local
  Ollama, and the prompt budget reflects the model actually serving the request.

## Non-goals

- Reviewing more than `MAX_FILES` files per run. The per-run cap stays as is.
- Editing or resolving comments the bot posted in earlier runs. Comments are
  append-only; the fix is to stop producing duplicates, not to clean them up.
- Persisting state outside GitLab. No database, no volume.
- Removing Ollama from the codebase. It stays as the optional Sidorovich voice
  fallback behind `SIDOROVICH_OLLAMA_FALLBACK`; only the *review* path leaves it.
- Parallelising review across files. The single-worker queue stays as it is.

## Approach: content-hash ledger

The bot records a hash of every diff hunk it has reviewed. On each run it
fetches the full MR diff as it does today, drops hunks whose hash it already
knows, and reviews only what is left.

### Why not SHA-based incremental diff

The obvious alternative is to store `last_reviewed_sha` and ask GitLab for
`last_sha..head_sha` via the compare API. It was rejected: after a force-push
or rebase, `last_sha` is unreachable, the compare fails, and the fallback is a
full-MR review — the exact flood this design exists to prevent. Rebases are
most common on precisely the large, long-lived MRs that caused the incident.

Content hashes are rebase-proof. A squash, cherry-pick, or force-push that
preserves hunk content preserves the hash, so the hunk is not reviewed twice.
The approach also needs no extra API call: the diff is already fetched
(`gitlab_client.py:26`).

The known cost: a reformat that genuinely changes a hunk — even by whitespace —
reads as new content and is reviewed again. That is accepted; the content did
change.

## Component 1: the ledger

New module `reviewer/ledger.py`. Owns the per-MR review state: reading it,
writing it, hashing hunks, and filtering already-seen hunks out of a diff.

### Hunk hashing

```python
def hunk_key(path: str, hunk: Hunk) -> str:
    digest = hashlib.sha256()
    digest.update(path.encode())
    digest.update(b"\x00")
    for line in hunk.lines:
        digest.update(line.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:12]
```

`hunk.new_start` and `hunk.old_start` are deliberately excluded. Line numbers
shift whenever an unrelated earlier hunk changes; the content does not. This is
what makes the ledger survive a rebase, and it is the one meaningful difference
from the existing `diff_fingerprint`, which does include `new_start`
(`gitlab_client.py:124`).

Accepted consequence: two byte-identical hunks in the same file collide, so the
second one draws no comment. Losing one duplicated comment is far cheaper than
a flood.

### State record

```python
@dataclass(frozen=True)
class Ledger:
    head: str                 # last reviewed head SHA, for display only
    posted: int               # comments this MR has drawn so far
    muted: bool               # bot silenced for this MR
    oversized: bool           # the "MR too large" note has been posted
    hunks: tuple[str, ...]    # reviewed hunk keys, oldest first
```

`hunks` is capped at `LEDGER_MAX_HUNKS` (2000) entries; the oldest are dropped
on overflow. At 15 bytes per entry that bounds the marker at roughly 30 KB
against a GitLab note limit of 1 MB.

## Component 2: the state note

The ledger is stored in GitLab itself — a single note per MR, created once and
**edited in place** on every subsequent write via `note.save()`. It is never
re-posted, so it does not consume the comment budget.

```markdown
🔒 **Сідорович — стан рев'ю**

Коментарів: 27/30 · останній head: `abc1234`

<!-- sidorovich-state:v1 {"posted":27,"muted":false,"head":"abc1234","oversized":false,"hunks":["3f2a1b9c8d7e"]} -->
```

The human-readable lines exist so a reviewer who stumbles on the note
understands it. The machine state is JSON inside an HTML comment: invisible in
the GitLab UI, and robust to parse. The `v1` tag allows a future format change
to be detected rather than misread.

Lookup is `mr.notes.list(iterator=True)` filtered for bodies containing
`sidorovich-state:v1`; the first match wins. Writes go through `note.save()`.

Concurrency is not a concern: `start_worker` runs exactly one consumer thread
(`queue.py:78`) and the Dockerfile pins `--workers 1`.

`gitlab_client.py` gains the note find/create/update helpers. The parsing and
state shape stay in `ledger.py`.

## Component 3: the review flow

```
review_merge_request(project_id, mr_iid, force):
    mr = fetch_mr()
    ledger = ledger.load(mr)                    # fail closed — see Error handling
    if ledger.muted and not force: return
    if release/hotfix branch: summarize_release_mr(); return   # unchanged

    file_diffs = fetch_file_diffs(mr)
    reviewable, skipped = partition_reviewable(file_diffs)

    if len(reviewable) > MAX_MR_FILES:
        if not ledger.oversized:
            post_note(oversized_message)        # costs 1
            ledger = ledger.spend(1).mark_oversized()
        ledger.save()
        return

    fresh = drop_known_hunks(reviewable, ledger)
    if not fresh:
        ledger.save(head=head_sha)              # silent: no note posted
        return

    kept, outcomes = select_files(fresh, skipped)   # MAX_FILES cap, unchanged
    for file_diff in kept:
        if ledger.remaining() <= 1:             # reserve the last slot
            outcomes.append(skipped_for_budget(file_diff))
            continue
        outcome = review_file(...)
        ledger = ledger.record(file_diff.hunks)
        if outcome.status == "reviewed":
            ledger = ledger.spend(1)
        ledger.save()                           # incremental — see Error handling

    if ledger.remaining() <= 1:
        post_note(budget_exhausted_message)     # the reserved slot
        ledger = ledger.spend(1).mute()
    else:
        post_note(render_summary(outcomes))
        ledger = ledger.spend(1)
    ledger.save()
```

The silent `if not fresh: return` is the fix for gap 4. An unchanged diff after
a container restart produces zero comments and zero noise, where today it
produces a full re-review.

`DedupeCache` is retained only for the release/hotfix summary path, which the
ledger does not cover. For the review path the ledger supersedes it.

`select_files` changes signature. The `is_reviewable` pass moves out into a new
`partition_reviewable`, because the oversized check needs the reviewable count
*before* file selection runs. `select_files` then takes the reviewable diffs
plus the skip outcomes `partition_reviewable` produced, so the summary note
still names lockfiles, assets and generated files with their reasons as it does
today.

## Component 4: guardrails

### Size threshold

`MAX_MR_FILES` (default 60). Above it, per-file review is skipped entirely and
the MR draws a single note: a meme, the file count, the threshold, and advice
to split the MR. The `oversized` flag prevents the note repeating on every
push, so an oversized MR costs exactly one comment for its whole life.

A manual `/review` does **not** bypass the threshold. An override that
re-enables a 300-file review is the same foot-gun in a different shape.

### Comment budget

`MR_COMMENT_BUDGET` (default 30) caps everything the bot creates in one MR:
inline discussions, summary notes, and the oversized note. The state note is
excluded — it is created once and thereafter edited.

The last slot of the budget is reserved so the run always has room for a
closing note. While `remaining() > 1` files are reviewed normally and the run
ends with the usual summary. Once `remaining()` reaches 1, remaining files are
recorded as skipped for budget, and the reserved slot carries a final note
saying the limit is reached and that `/review` resets it. `muted` is then set to
true, so later pushes cost nothing at all.

A manual `/review` clears `muted` and resets `posted` to zero.

### Kill switch

Two levels, because disabling the webhook is instance-wide and too blunt:

- `SIDOROVICH_ENABLED` (default true). When false, `should_review` returns
  `None` immediately — no fetch, no queue, no work.
- `/sidorovich stop` in an MR comment sets `muted = true` for that MR alone.
  Acknowledgement is an edit to the state note, not a new comment; the mute
  path must not itself cost a comment. `/review` lifts it.

`should_review` returns a dataclass instead of a tuple, so a second command can
be expressed:

```python
@dataclass(frozen=True)
class ReviewJob:
    project_id: int
    mr_iid: int
    force: bool
    command: str      # "review" | "mute"
```

## Component 5: review backend migration

Review moves from local Ollama to OpenRouter `poolside/laguna-s-2.1`, the paid
tier. Verified against the OpenRouter models API on 2026-08-25: 1 048 576 token
context, 131 072 max completion tokens, $0.09/M prompt and $0.18/M completion.
The `:free` sibling exists at 262 144 context and zero cost, and is explicitly
not the default — see below.

Ollama keeps exactly one job: the optional Sidorovich voice fallback behind
`SIDOROVICH_OLLAMA_FALLBACK`. It is no longer reachable from `review_file`.

### Two model roles

`OPENROUTER_MODEL` today means "the model that writes Sidorovich's voice". The
review path needs its own, because the two roles want different models,
different temperatures and different token ceilings. Two new settings:

- `OPENROUTER_REVIEW_MODEL`, default `poolside/laguna-s-2.1`
- `OPENROUTER_REVIEW_FALLBACK_MODELS`, default empty

The fallback list feeds the same `body["models"]` mechanism the voice client
already uses (`openrouter_client.py:57`), letting OpenRouter reroute internally
instead of failing the request.

### Why the paid tier and not `:free`

Review is roughly forty model calls per MR where the Sidorovich voice was one.
The note already in `config.py:44` records what happens to `:free` pools under
that load — they 429 constantly, because the free allowance is shared across
every OpenRouter user pointed at the same upstream.

The cost of not being rate limited: a forty-file MR sends on the order of
160 000 prompt tokens and receives perhaps 32 000, which is $0.014 in and
$0.006 out — about two cents per merge request. Free-tier reliability is not
worth two cents.

The fallback list defaults to empty because a paid model already reroutes
across providers on its own. `poolside/laguna-xs-2.1` is the obvious entry if a
cheaper degraded path is ever wanted, and `poolside/laguna-s-2.1:free` is the
entry for anyone who wants to spend nothing at the cost of reliability.

### Context budget becomes backend-aware

`prompt.input_token_budget()` currently computes
`OLLAMA_NUM_CTX - OLLAMA_NUM_PREDICT - estimate_tokens(SYSTEM_PROMPT) - PROMPT_TOKEN_BUFFER`,
which yields about 7 364 input tokens. Pointing review at a 262 144 token model
while still sending 7 364 token prompts would waste the entire reason for the
migration.

The function takes its context and output reserve from two new settings that
describe the review backend rather than Ollama:

- `REVIEW_CONTEXT_TOKENS`, default 262144
- `REVIEW_MAX_OUTPUT_TOKENS`, default 4096

Both sit deliberately below what the model allows.

`REVIEW_CONTEXT_TOKENS` is capped at 262 144 rather than the model's full
1 048 576. Prompt tokens are billed, `estimate_tokens` is a deliberately
conservative `len // 3`, and one file whose diff genuinely exceeds 262 144
tokens is not something a single review call should be attempting. Raising the
variable unlocks the rest of the window without a code change.

`REVIEW_MAX_OUTPUT_TOKENS` is capped at 4 096 against a 131 072 ceiling: a
single-file review needing more than 4 096 output tokens is producing a wall of
comments, which is the failure this document exists to prevent. It is also the
`max_tokens` sent on each review request.

The `OLLAMA_NUM_CTX` and `OLLAMA_NUM_PREDICT` settings survive untouched and
keep governing the Ollama voice fallback.

### Consequences for the prompt ladder

`build_prompt_ladder` builds L0/L1/L2 attempts and `review_file` takes the
cheapest that fits (`pipeline.py:100`). The L2 rung splits a file into
per-hunk prompts purely to squeeze past an 8 192 token context. At 262 144
tokens essentially every file fits at L1, so L2 stops firing.

The ladder is kept rather than deleted: it is the mechanism that guarantees no
file is silently dropped for being too large, and a pathological file still
needs it. This is a documented behaviour change, not dead code to remove.

### Rate-limit circuit breaker

`review_file` returns a `FileOutcome` with status `error` when a chat call
fails. Under a saturated free pool every one of forty files can fail that way,
producing a run that burns its deadline and reports forty errors.

A `ReviewState` breaker mirrors the existing `VoiceState` (`pipeline.py:36`):
after `REVIEW_FAILURE_LIMIT` (2) consecutive failures whose `done_reason` is
`ratelimit`, the run stops early. On the paid tier this should never fire; it
exists because `OPENROUTER_REVIEW_MODEL` is a knob and someone will eventually
point it at a `:free` model. Remaining files are recorded as skipped with
a rate-limit reason and the summary says so once. Hunks from unreviewed files
are *not* recorded in the ledger, so the next push retries them.

### Shared chat types

`ChatResult` and `clean_response` live in `ollama_client.py`, and
`openrouter_client.py` imports them from there (`openrouter_client.py:8`). With
review no longer touching Ollama that dependency direction is backwards. Both
move to a new `reviewer/chat_types.py`; `ollama_client` and `openrouter_client`
import from it. This is a pure move — no behaviour change — and it is in scope
only because the migration makes the current direction actively misleading.

## Configuration

| Name | Default | Meaning |
|---|---|---|
| `MAX_MR_FILES` | 60 | Reviewable files above which inline review is skipped |
| `MR_COMMENT_BUDGET` | 30 | Comments the bot may post per MR, lifetime |
| `SIDOROVICH_ENABLED` | true | Global kill switch |
| `LEDGER_MAX_HUNKS` | 2000 | Hunk keys retained per MR before oldest are dropped |
| `OPENROUTER_REVIEW_MODEL` | `poolside/laguna-s-2.1` | Model that performs code review |
| `OPENROUTER_REVIEW_FALLBACK_MODELS` | *(empty)* | Extra reroute targets, tried in order |
| `REVIEW_CONTEXT_TOKENS` | 262144 | Context window of the review backend |
| `REVIEW_MAX_OUTPUT_TOKENS` | 4096 | Output reserve and per-request `max_tokens` for review |

All eight go in `reviewer/config.py` alongside the existing limits, and are
mirrored into `docker-compose.yml` and `.env.example`.

## Error handling

**Ledger loading fails closed.** Three distinct cases, and the distinction is
load-bearing:

- `mr.notes.list` raised — **skip the run**, log a warning. Do not review.
- The list returned and contains no marker — a genuine first review. Proceed
  with an empty ledger.
- A marker is present but its JSON will not parse — **skip the run**, log an
  error.

Falling back to an empty ledger on failure would mean a full re-review of the
whole MR, which is the incident this design prevents. Silence is the safe
failure mode.

**The ledger is saved incrementally**, after each reviewed file rather than
once at the end of the run. A crash mid-run would otherwise leave comments
posted but unrecorded, and the next push would post them all again. The cost is
roughly 40 note updates spread over an eight-minute run.

The oversized and budget-exhausted paths save the ledger before returning.

## Refactoring

`pipeline.py` is 431 lines and this design adds ledger handling and budget
accounting to it. The Sidorovich voice code — `VoiceState`, `flavor_review`,
`prefer_ukrainian`, `_voice_chat`, `preserves_findings`, `_fence_bodies`, and
the `VOICE_DEADLINE_S` / `VOICE_FAILURE_LIMIT` / `FENCE_BODY_RE` constants,
about 90 lines — moves to a new `reviewer/voice.py`. It is a self-contained
concern with one entry point, `flavor_review`, and moving it leaves `pipeline`
as orchestration. `tests/test_pipeline.py` imports those symbols and updates to
the new module.

No other refactoring is in scope.

## Testing

**`tests/test_ledger.py`** (new)

- Marker round-trip: serialise a `Ledger`, parse it back, get the same values.
- Malformed JSON in the marker raises rather than returning an empty ledger.
- A notes list with no marker yields a fresh empty ledger.
- `hunk_key` is unchanged when `new_start` and `old_start` shift but hunk lines
  are identical. This is the proof that a rebase does not re-trigger review.
- `hunk_key` differs when a hunk line changes.
- Recording past `LEDGER_MAX_HUNKS` drops the oldest keys and keeps the newest.

**`tests/test_pipeline_ledger.py`** (new)

- Hunks already in the ledger are filtered out of the review set.
- Every hunk known → `post_note` and `post_inline` are called zero times.
- `mr.notes.list` raising → the run is skipped, nothing posted.
- Budget exhausted → exactly one final note, and `muted` becomes true.
- Oversized MR → one note, zero inline comments; a second run posts nothing.
- `/review` on a muted MR clears the mute and resets `posted`.
- Skip reasons from `partition_reviewable` (lockfile, asset, generated) still
  reach the summary note after the signature change.

**`tests/test_webhook.py`** (extend)

- `SIDOROVICH_ENABLED=false` → the event is ignored.
- `/sidorovich stop` → a job with `command == "mute"`.

**`tests/test_review_backend.py`** (new)

- `review_file` calls the OpenRouter review client and never `ollama_client.chat`.
- The request body carries `OPENROUTER_REVIEW_MODEL` and
  `REVIEW_MAX_OUTPUT_TOKENS`, not the voice model or `OPENROUTER_MAX_TOKENS`.
- `OPENROUTER_REVIEW_FALLBACK_MODELS` populates `body["models"]` when set, and
  the key is absent when the list is empty.
- `input_token_budget` derives from `REVIEW_CONTEXT_TOKENS` and
  `REVIEW_MAX_OUTPUT_TOKENS`; changing `OLLAMA_NUM_CTX` does not move it.
- Two consecutive `ratelimit` outcomes stop the run; remaining files are
  reported as skipped and their hunks are absent from the ledger.
- A missing `OPENROUTER_API_KEY` fails the run cleanly rather than silently
  reviewing nothing.

**`tests/test_pipeline.py`** (update) — `select_files` signature change, voice
imports move to `reviewer.voice`, and `ChatResult` imports move to
`reviewer.chat_types`. Existing tests that stub `ollama_client.chat` for the
review path restub the OpenRouter review client.

## Expected outcome

The incident: 300 files exceeds the 60-file threshold, so one note is posted
and the `oversized` flag stops any repeat. 592 comments becomes 1.

A 30-file MR pushed fifteen times: the first run posts up to 30 comments and
hits the budget; every later push finds its hunks already in the ledger and
posts nothing. The total stays at 30 regardless of push count.
