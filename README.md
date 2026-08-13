# AI Code Reviewer with Ollama & Cloudflare Tunnel

This project sets up a local AI Code Review bot that integrates with GitLab Merge Requests using Docker, Ollama, and Cloudflare Tunnel.

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

6.  **Pull the AI Model:**
    Wait for the containers to start, then run:
    ```bash
    docker compose exec ollama ollama pull codellama
    ```
    (You can swap `codellama` for `llama3`, `mistral`, etc., in `.env` and here).

    Using qwen2.5-coder
    -------------------
    If you want to use the `qwen2.5-coder:7b-instruct-q4_K_M` model, set the `OLLAMA_MODEL` value in your `.env` file:

    ```dotenv
    OLLAMA_MODEL=qwen2.5-coder:7b-instruct-q4_K_M
    ```

    Then pull that model inside the `ollama` container:

    ```bash
    docker compose exec ollama ollama pull qwen2.5-coder:7b-instruct-q4_K_M
    ```

    Notes:
    - The model string may include quantization or "instruct" suffixes — keep the exact name in `.env`.
    - After pulling, the service will use the model referenced by `OLLAMA_MODEL` when handling reviews.
    - Start or restart the services if you change `.env` so the new model name is picked up.

7.  **Configure GitLab Webhook:**
    *   Go to your GitLab Project > **Settings > Webhooks**.
    *   **URL:** `https://code-review.yourdomain.com/webhook` (The public hostname you set in Cloudflare).
    *   **Secret Token:** The same `WEBHOOK_SECRET` from your `.env`.
    *   **Triggers:** check **Merge request events** *and* **Note events** (so the bot can also respond to review comments).
    *   Click **Add webhook**.

## Usage

- **Automatic reviews:** Create or update a Merge Request in your GitLab project. The AI reviewer will automatically comment on the MR with feedback.
- **Manual trigger via comment:** Post a comment containing `/review` (case-insensitive) on any MR. The bot will fetch the current diff and post its AI review again.

You can change or extend the keyword by editing `review_server.py` if desired.

## Tuning for Mac Mini M4 16 GB

The Flask app talks to **Ollama running on the host** (not in Docker). On 16 GB
unified memory, **context size is the main cause of hung reviews** — a 12B model
at 32K context plus a large MR diff will swap-thrash and appear stuck.

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

# Deadlines and limits.
PER_FILE_TIMEOUT_S=90
MR_TIMEOUT_S=480
MAX_FILES=40
QUEUE_MAXSIZE=32
```

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

Two situations skip a review entirely and **post no comment on the MR**:

*   The merge request's source branch starts with `release/`. This is a
    deliberate, permanent skip — release branches are never reviewed.
*   The merge request's diff is byte-for-byte identical to the diff from its
    last completed review (tracked by `DEDUPE_CACHE_SIZE` most-recent
    fingerprints). This is a cost-saving skip on repeat webhooks, not a
    permanent one — it clears once the diff changes again, or once the
    fingerprint ages out of the cache.

In both cases the reason is visible in `docker compose logs -f app` (`targets
a release branch; skipping` / `diff unchanged since last review; skipping`)
even though nothing appears on the MR itself. A review that fails partway
through (for example, a GitLab API error while posting the summary) is never
counted as "reviewed" for dedupe purposes — the next identical webhook will
retry it rather than being silently swallowed.

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
*   **Ollama:** Ensure the model is pulled (`ollama list`).
*   **Tunnel:** Check Cloudflare dashboard to see if the tunnel is "Healthy".
*   **Review stuck / never finishes:** check `curl localhost:5000/health` for
    `queue_depth`. A depth above zero with no log progress means Ollama is
    wedged — run `ollama ps` and confirm `PROCESSOR=100% GPU`. Individual files
    now abort after `PER_FILE_TIMEOUT_S` instead of hanging.
*   **Read timed out:** Ollama is partially CPU-offloaded. Check `ollama ps`.
*   **Slow first review:** Model cold-load from disk on a 16 GB box can take
    30–90 s. The `keep_alive: 24h` setting prevents this on subsequent MRs.
*   **`404 Not Found for url: .../api/chat`:** Ollama is up but the
    `model` field in the request points at a model that is not currently
    pulled. Two common causes:
    1.  You changed `OLLAMA_MODEL` in `.env` but used `docker compose restart`,
        which does **not** re-read `.env`. Always use
        `docker compose up -d --force-recreate app` after editing `.env`.
    2.  The model in `.env` was uninstalled (`ollama rm ...`). Re-pull it
        or pick another model from `ollama list`.
