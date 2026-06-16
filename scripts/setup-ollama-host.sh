#!/usr/bin/env bash
#
# One-time host configuration for Ollama on macOS (M4 16 GB).
# Tunes the daemon for 7B–12B models at 8K context (safe default for 16 GB).
#
# Why each setting matters on a 16 GB unified-memory box:
#   OLLAMA_FLASH_ATTENTION=1   - required to enable KV cache quantization.
#   OLLAMA_KV_CACHE_TYPE=q8_0  - halves KV cache memory at 8K–16K context
#                                with negligible quality loss vs. fp16.
#   OLLAMA_KEEP_ALIVE=24h      - keeps the model in unified memory between
#                                requests so we don't pay a multi-minute cold
#                                reload from disk per MR.
#   OLLAMA_MAX_LOADED_MODELS=1 - never load a second model concurrently; we
#                                cannot afford it.
#   OLLAMA_NUM_PARALLEL=1      - serialize requests at the daemon level
#                                (the Flask app already holds review_lock,
#                                this is belt-and-braces).
#
# After running this script, fully quit and relaunch the Ollama app
# (or `brew services restart ollama` / `pkill ollama && ollama serve`)
# so it picks up the new env.

set -euo pipefail

echo "Setting Ollama environment variables via launchctl..."

launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE   q8_0
launchctl setenv OLLAMA_KEEP_ALIVE      24h
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
launchctl setenv OLLAMA_NUM_PARALLEL    1

echo "Done. Current values:"
for var in OLLAMA_FLASH_ATTENTION OLLAMA_KV_CACHE_TYPE OLLAMA_KEEP_ALIVE \
           OLLAMA_MAX_LOADED_MODELS OLLAMA_NUM_PARALLEL; do
  printf "  %-26s = %s\n" "$var" "$(launchctl getenv "$var")"
done

cat <<'EOF'

Next steps:
  1. Quit the Ollama menubar app (or: pkill ollama).
  2. Relaunch Ollama (or: ollama serve).
  3. Pre-warm the model so the first MR doesn't pay the cold load:
       ollama run <your-model> "ok" </dev/null
  4. Verify it's 100% on GPU and using your context length:
       ollama ps
     You want PROCESSOR=100% GPU and CONTEXT matching OLLAMA_NUM_CTX (default 8192).

If `ollama ps` shows any CPU% in the PROCESSOR column you are out of memory.
Either:
  - Lower Docker Desktop RAM (Settings > Resources > Memory) to 2 GB, or
  - Lower OLLAMA_NUM_CTX in .env (try 8192 or 4096), then:
       ollama stop <your-model>
    (Ollama keeps the old KV cache allocation until the model is unloaded.)
  - Switch OLLAMA_KV_CACHE_TYPE to q4_0 (more aggressive but lossier).
  - Use a smaller model (qwen2.5-coder:7b instead of gemma4:12b).
EOF
