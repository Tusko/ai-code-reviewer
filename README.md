# AI Code Reviewer with Cloudflare Tunnel

This project sets up an AI Code Review bot that integrates with GitLab Merge Requests using Docker, OpenRouter, and Cloudflare Tunnel. Code review itself runs on OpenRouter (see [Review model](#review-model)) — a local Ollama model is optional and, by default, used only as a fallback for the Sidorovich voice.

## Prerequisites

1.  **Docker & Docker Compose** installed.
2.  **GitLab Account** (or self-hosted instance).
3.  **Cloudflare Account** (for the tunnel).

## Setup

1.  **Clone this repository** (if you haven't already).
2.  **Copy the environment file:**
    ```bash
    cp .env.example .env
    ```
3.  **Configure `.env`:**
    *   `GITLAB_TOKEN`: Create a [Personal Access Token](https://gitlab.com/-/profile/personal_access_tokens) with `api` scope.
    *   `WEBHOOK_SECRET`: Generate a random string (e.g., `openssl rand -hex 12`).
    *   `OPENROUTER_API_KEY`: **Required.** Code review runs on OpenRouter, not
        Ollama — without a key every file review comes back as an error. Get
        one at https://openrouter.ai/keys.
    *   `TUNNEL_TOKEN`: See step 4.

4.  **Set up Cloudflare Tunnel:**
    *   Go to [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/).
    *   Navigate to **Networks > Tunnels** and create a new tunnel.
    *   Choose **Docker** as the environment.
    *   Copy the token command, but extract just the token string (the part after `--token`). Paste it into `.env`.
    *   **Configure the Public Hostname** in the Cloudflare dashboard:
        *   **Public Hostname:** `code-review.yourdomain.com` (or whatever you choose).
        *   **Service:** `http://app:5000` (The internal docker service name and port).

5.  **Start the services:**
    ```bash
    docker compose up -d
    ```

6.  **(Optional) Pull a local Ollama model:**
    Code review itself needs no local model — it runs on OpenRouter. This
    step is only needed if you set `SIDOROVICH_OLLAMA_FALLBACK=true`, which
    lets a local model attempt the release/hotfix roast when OpenRouter is
    unavailable. Ollama runs on the **host**, not in Docker (see "Tuning for
    Mac Mini M4 16 GB" below), so pull it there directly:
    ```bash
    ollama pull qwen2.5-coder:7b
    ```
    This matches the model recommended in the tuning section and the
    `OLLAMA_MODEL` default. If you use a different model, set `OLLAMA_MODEL`
    in `.env` to match and re-pull under that exact name.

7.  **Configure GitLab Webhook:**
    *   Go to your GitLab Project > **Settings > Webhooks**.
    *   **URL:** `https://code-review.yourdomain.com/webhook` (The public hostname you set in Cloudflare).
    *   **Secret Token:** The same `WEBHOOK_SECRET` from your `.env`.
    *   **Triggers:** check **Merge request events** *and* **Note events**. Note
        events are required for `/review` and `/sidorovich stop` — without
        them neither comment command does anything.
    *   Click **Add webhook**.

## Usage

- **Automatic reviews:** Create or update a Merge Request in your GitLab project. The AI reviewer will automatically comment on the MR with feedback.
- **Manual trigger via comment:** Post a comment containing `/review`
  (case-insensitive) on any MR. The bot will re-review now, clearing any mute
  and resetting the comment counter to zero.
- **Silence a noisy MR:** Post `/sidorovich stop` to mute the bot for that MR
  only; `/review` lifts it. See [Comment guardrails](#comment-guardrails) for
  the full behaviour, including why a command has to be typed as its own word
  — `/review` inside a path like `app/review/service.py`, inside backticks,
  or in a quoted reply does not trigger it.

You can change or extend the command patterns by editing `REVIEW_RE` and
`STOP_RE` in `review_server.py` if desired.

## Comment guardrails

The bot keeps a per-MR ledger of the diff hunks it has already reviewed,
stored in a single 🔒 state note it creates once and edits in place.
**Do not edit or delete that note by hand** — deleting it makes the bot
forget the MR and review it from scratch.

| Situation | What the bot does |
|---|---|
| More than `MAX_MR_FILES` (60) reviewable files | Posts exactly one note asking for a smaller MR, the first time this happens. Per-file review is skipped on that push and on every later push while the file count stays above the threshold, with no further note. If a later push drops the file count to `MAX_MR_FILES` or below, review resumes normally. |
| A push whose hunks were all reviewed already | Posts nothing at all. |
| A push with new hunks | Reviews only the new hunks. |
| `MR_COMMENT_BUDGET` (30) comments reached | Posts one closing note and goes quiet until `/review`. |
| A rebase or force-push | Unchanged hunk content is not re-reviewed. Hunk keys hash content, not line numbers. |
| A title or assignment problem | Posts one hygiene note (see below). Costs a comment slot like any other note. |

### MR hygiene

Before any code is reviewed the bot audits the merge request itself and, when
something is wrong, posts one blunt note about it:

*   **Title.** It must start with a bare ticket key — `MONO-1628: reuse browser
    tabs`. A Conventional Commit wrapper such as `fix(MONO-1628): ...` is
    rejected: GitLab cannot link that back to the issue. A `Draft:` or `WIP:`
    prefix in front of the key is allowed. Change the rule with
    `MR_TITLE_PATTERN` in `.env`.
*   **Assignment.** An MR with neither an assignee nor a reviewer is nagged.
    Either one is enough.

The note never involves a model — the checks are local and deterministic, and
the meme on top comes from the same phrase list as the summary.

Which problems were reported is recorded in the 🔒 state note, not re-derived
from the comments. A push that changes nothing costs nothing; fixing one of two
problems produces exactly one new note about the other; fixing everything is
silent, and a relapse is nagged about again.

`release/` and `hotfix/` branches skip this along with the rest of the review —
they get the commit roast instead.

### Commands

- `/review` in an MR comment — re-review now, clear any mute, reset the
  comment counter to zero. It does **not** override the `MAX_MR_FILES`
  threshold.
- `/sidorovich stop` in an MR comment — silence the bot for that MR only. It
  acknowledges by editing the state note, not by posting a comment. `/review`
  lifts it.

A command has to be typed, not merely mentioned: `/review` inside a path like
`app/review/service.py`, inside backticks, or in a quoted reply does not
trigger it, and `/sidorovich stopwatch` is not the kill switch. Both commands
arrive as GitLab Note Hook webhooks, so **Note events** must be enabled on the
webhook (see the Setup step above) or neither command does anything.

The bot also ignores any comment it judges to be its own, so that it can't
clear its own mute or reset its own budget — its closing note literally tells
the human to type `/review`, and GitLab fires a Note Hook for the bot's own
comments too. That check needs to know which GitLab account `GITLAB_TOKEN`
authenticates as. Normally the bot asks GitLab directly the first time it
matters; if the token cannot look up its own user (for example a
project-scoped access token without permission to read its own identity),
the bot **fails closed and silently ignores every `/review` and
`/sidorovich stop` comment**, with nothing posted to explain why. If comment
commands stop working for no apparent reason, check the container logs for
`cannot tell whose comment this is`, and set `SIDOROVICH_BOT_USERNAME` in
`.env` to the bot's exact GitLab username to fix it.

### Global off switch

`SIDOROVICH_ENABLED=false` makes every webhook a no-op. Use it instead of
disabling the GitLab webhook, which silences the bot for every project on the
instance.

## Review model

Code review runs on OpenRouter (`OPENROUTER_REVIEW_MODEL`, default
`poolside/laguna-s-2.1`) — a separate model and a separate OpenRouter call
from the one behind the Sidorovich voice (`OPENROUTER_MODEL`). Findings are
generated by `OPENROUTER_REVIEW_MODEL`; when `SNARK=true`, each finding is
then rewritten in Sidorovich's voice by a second call to `OPENROUTER_MODEL`.
**`OPENROUTER_API_KEY` is now required for review to produce any output at
all**, not only for the Sidorovich voice — without it, every file comes back
as an error.

`OPENROUTER_REVIEW_FALLBACK_MODELS` is a comma-separated list of models tried
after `OPENROUTER_REVIEW_MODEL`, in the same OpenRouter request.
`REVIEW_CONTEXT_TOKENS` and `REVIEW_MAX_OUTPUT_TOKENS` bound the review
prompt and response the same way `OLLAMA_NUM_CTX` / `OLLAMA_NUM_PREDICT`
used to for the old Ollama-based review path.

Ollama is no longer used for code review at all. It remains available only
as an opt-in fallback for the Sidorovich release/hotfix roast, behind
`SIDOROVICH_OLLAMA_FALLBACK` (off by default); `OLLAMA_NUM_CTX` and
`OLLAMA_NUM_PREDICT` govern only that path now, not review.

## Upgrading from an earlier version

`docker-compose.yml` reads tuning values from `.env` with fallback defaults
(`${OLLAMA_NUM_PREDICT:-320}`), so an existing `.env` from before this refactor
keeps its old values and silently overrides the new, faster defaults — you
upgrade, see none of the intended speedup, and nothing tells you why. If your
`.env` predates this change:

1.  Delete `MAX_PROMPT_CHARS` and `MIN_OUTPUT_TOKENS` from `.env` — both keys
    no longer exist and are ignored.
2.  Set `OLLAMA_NUM_PREDICT=320` and `CONTEXT_WINDOW=15`.
3.  Add `OLLAMA_NUM_BATCH=512`.
4.  Recreate the app container so the new values take effect:
    ```bash
    docker compose up -d --build --force-recreate app
    ```

## Tuning for Mac Mini M4 16 GB

> Code review runs on OpenRouter now, not Ollama — see [Review model](#review-model).
> Everything in this section applies only if you enable
> `SIDOROVICH_OLLAMA_FALLBACK=true`, and even then Ollama only ever sees a
> short commit-list roast prompt, never a full MR diff. If you are not using
> that fallback, none of this tuning is required to run the bot.

The Flask app talks to **Ollama running on the host** (not in Docker). On 16 GB
unified memory, **context size is the main cause of a hung roast** — a 12B model
at 32K context will swap-thrash and appear stuck.

### Recommended model + context

| Model | Disk | Safe `OLLAMA_NUM_CTX` on 16 GB |
|---|---|---|
| `qwen2.5-coder:7b` | ~5 GB | 8192–16384 |
| `gemma4:12b-it-qat` | ~7 GB | **8192** (do not use 32K) |

Set in `.env`:

```bash
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

# Tone. true: a meme on the summary note, and Sidorovich voice on inline
# findings (a second OpenRouter call rewrites the dry review). false: dry
# comments end to end. Voice never invents or drops issues; a failed
# rewrite posts the original review.
SNARK=true

# Deadlines and limits.
PER_FILE_TIMEOUT_S=90
MR_TIMEOUT_S=480
MAX_FILES=40
QUEUE_MAXSIZE=32
```

### How reviews are scheduled

Webhooks return immediately after enqueuing. One background worker drains the
queue, so only one review job is ever in flight. Repeat webhooks for the
same MR coalesce into the single queued job, and an MR whose diff has not
changed since its last review is skipped entirely.

`GET /health` reports `queue_depth`, which is the fastest way to tell whether
the bot is busy or stuck.

Reviews run one file at a time. Each file gets its own request with a
`PER_FILE_TIMEOUT_S` deadline, and the whole MR is bounded by `MR_TIMEOUT_S`.
Files that cannot fit the context budget are named in the summary note rather
than dropped silently.

Finding detection runs on OpenRouter (`OPENROUTER_REVIEW_MODEL`), not Ollama —
see [Review model](#review-model). With `SNARK=true` and `OPENROUTER_API_KEY`
set, each inline finding is then rewritten as Sidorovich (surzhyk, swearing)
by a second, separate OpenRouter call (`OPENROUTER_MODEL`) afterwards — same
headings, same `*Fix:*` blocks, only the prose changes. If that rewrite
fails, rate-limits, or mutates a finding, the dry review is posted instead.
After two (`VOICE_FAILURE_LIMIT` in `reviewer/voice.py`, not configurable) consecutive voice failures (free-tier rate
limits, mostly) the rest of that MR stays dry, so one merge request never
mixes voiced and dry comments. `SNARK=false` keeps comments professional.

Two situations skip a **full** file-by-file review:

*   The merge request's source branch starts with `release/` or `hotfix/`.
    Full review is skipped on purpose, but the bot still posts a short
    Sidorovich-style commit summary (surzhyk, swearing, 100–150 words) so
    the team can see what landed without waiting on the per-file loop.
    That roast uses OpenRouter (`OPENROUTER_MODEL`, default
    `google/gemini-2.5-flash-lite`) when `OPENROUTER_API_KEY` is set —
    local coder models cannot write surzhyk, and letting them try produces
    gibberish under Sidorovich's name. Without a key the bot posts a plain
    commit digest instead; set `SIDOROVICH_OLLAMA_FALLBACK=true` to let the
    local model attempt the roast anyway. A transient OpenRouter error
    posts nothing and retries on the next webhook.
*   The merge request's diff is byte-for-byte identical to the diff from its
    last completed review (tracked by `DEDUPE_CACHE_SIZE` most-recent
    fingerprints). This is a cost-saving skip on repeat webhooks, not a
    permanent one — it clears once the diff changes again, or once the
    fingerprint ages out of the cache.

A skipped full review is visible in `docker compose logs -f app`
(`release/hotfix summarised` / `diff unchanged since last review; skipping`).
A review that fails partway through (for example, a GitLab API error while
posting the summary) is never counted as "reviewed" for dedupe purposes —
the next identical webhook will retry it rather than being silently swallowed.

After changing context, **unload the model** so Ollama drops the old KV cache:

```bash
ollama stop gemma4:12b-it-qat
```

Then recreate the app container:

```bash
docker compose up -d --force-recreate app
```

### One-time host setup

```bash
./scripts/setup-ollama-host.sh
```

This sets, via `launchctl`:

| Env var | Value | Why |
|---|---|---|
| `OLLAMA_FLASH_ATTENTION` | `1` | Required to enable KV cache quantization. |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Halves KV cache memory. |
| `OLLAMA_KEEP_ALIVE` | `24h` | Keep model in unified memory between MRs. |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Never load a second model concurrently. |
| `OLLAMA_NUM_PARALLEL` | `1` | Serialize requests at the daemon level. |

After running, **fully quit and relaunch the Ollama app** (or
`pkill ollama && ollama serve`) so it re-reads the env.

### Recommended Docker Desktop settings

In Docker Desktop → Settings → Resources, **drop the VM memory to 2 GB**.
The reviewer container only runs Flask; it does not need more. Every GB you
take back from Docker is a GB the model can use.

### Verify

After restart, pre-warm the model and confirm everything is on GPU:

```bash
ollama run gemma4:12b-it-qat "ok" </dev/null
ollama ps
```

You should see `PROCESSOR=100% GPU` and `CONTEXT=8192` (matching `OLLAMA_NUM_CTX`).

If `PROCESSOR` shows any CPU%, you are OOM. Lower `OLLAMA_NUM_CTX` to 4096,
run `ollama stop <model>`, or switch to `qwen2.5-coder:7b`.

### Memory budget (gemma4:12b @ 8K context)

| Consumer | Approx. RAM |
|---|---|
| macOS baseline | ~3.5 GB |
| Docker Desktop VM (limit to 2 GB) | ~2.0 GB |
| Model weights (QAT) | ~7.2 GB |
| KV cache @ 8K, q8_0 | ~0.8 GB |
| **Total** | **~13.5 GB** |

At 32K context the same model needs ~3 GB of KV cache alone and will hang on 16 GB.

## Troubleshooting

*   **Logs:** Check logs with `docker compose logs -f`.
*   **Ollama:** Only relevant if `SIDOROVICH_OLLAMA_FALLBACK=true`. Ensure the
    model is pulled (`ollama list`).
*   **Tunnel:** Check Cloudflare dashboard to see if the tunnel is "Healthy".
*   **Review stuck / never finishes:** check `curl localhost:5000/health` for
    `queue_depth`. A depth above zero with no log progress most likely means
    `OPENROUTER_API_KEY` is missing or invalid, or OpenRouter itself is rate
    limiting the request — check the logs for `OpenRouter returned` /
    `Error communicating with OpenRouter`. With `SIDOROVICH_OLLAMA_FALLBACK=true`
    it can also mean Ollama is wedged — run `ollama ps` and confirm
    `PROCESSOR=100% GPU`. Individual files now abort after `PER_FILE_TIMEOUT_S`
    instead of hanging.
*   **Read timed out (Ollama fallback path only):** Ollama is partially
    CPU-offloaded. Check `ollama ps`.
*   **`/review` or `/sidorovich stop` does nothing:** Confirm **Note events**
    is enabled on the GitLab webhook (Setup step 7) — without it neither
    command reaches the bot at all. Also check the logs for
    `cannot tell whose comment this is`: if `GITLAB_TOKEN` cannot look up its
    own GitLab user, both commands fail closed and are silently ignored; set
    `SIDOROVICH_BOT_USERNAME` to fix it. See
    [Comment guardrails](#comment-guardrails).
*   **The bot suddenly stopped commenting on one MR:** Check the 🔒 state note
    on the MR — it says whether the MR is muted (`/sidorovich stop`, or the
    `MR_COMMENT_BUDGET` was reached) or marked oversized
    (`MAX_MR_FILES` exceeded). Post `/review` to clear a mute and reset the
    comment counter; an oversized MR needs fewer files instead. Also check
    `SIDOROVICH_ENABLED`, which silences every MR at once.
*   **Slow first roast (Ollama fallback path only):** Model cold-load from
    disk on a 16 GB box can take 30–90 s. The `keep_alive: 24h` setting
    prevents this on subsequent MRs.
*   **`404 Not Found for url: .../api/chat` (Ollama fallback path only):**
    Ollama is up but the `model` field in the request points at a model that
    is not currently pulled. Two common causes:
    1.  You changed `OLLAMA_MODEL` in `.env` but used `docker compose restart`,
        which does **not** re-read `.env`. Always use
        `docker compose up -d --force-recreate app` after editing `.env`.
    2.  The model in `.env` was uninstalled (`ollama rm ...`). Re-pull it
        or pick another model from `ollama list`.
