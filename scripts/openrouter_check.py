#!/usr/bin/env python3
"""Diagnose OpenRouter 429s: report the key's real limits, then test the model.

    OPENROUTER_API_KEY=sk-or-... python3 scripts/openrouter_check.py
    python3 scripts/openrouter_check.py --env .env
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def load_env_file(path):
    """Minimal KEY=VALUE reader, so the script works without python-dotenv."""
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    except FileNotFoundError:
        print(f"! {path} not found, using the current environment", file=sys.stderr)


def call(path, key, body=None):
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode() if body else None,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": "Sidorovich",
        },
        method="POST" if body else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, dict(response.headers), json.load(response)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {"error": {"message": raw[:500]}}
        return exc.code, dict(exc.headers), payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None, help="read KEY=VALUE from this file")
    parser.add_argument("--model", default=None, help="override OPENROUTER_MODEL")
    args = parser.parse_args()

    if args.env:
        load_env_file(args.env)

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set")
    model = args.model or os.environ.get(
        "OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free",
    )
    print(f"key ...{key[-6:]}   model {model}\n")

    status, _, payload = call("/key", key)
    data = payload.get("data") or payload
    print(f"--- GET /key ({status}) ---")
    if status == 200:
        # is_free_tier=True is the usual cause of a 100% 429 rate: the daily
        # `:free` allowance is gated on credits *purchased*, not on balance.
        for field in ("label", "is_free_tier", "limit", "limit_remaining", "usage"):
            if field in data:
                print(f"  {field}: {data[field]}")
        if data.get("rate_limit"):
            print(f"  rate_limit: {data['rate_limit']}")
        if data.get("is_free_tier"):
            print("  => free tier. `:free` models are capped hard per day.")
            print("     Buying 10 credits raises that cap; a paid model skips it.")
    else:
        print(f"  {json.dumps(payload)[:400]}")

    print(f"\n--- POST /chat/completions ({model}) ---")
    status, headers, payload = call("/chat/completions", key, {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 16,
    })
    print(f"  status: {status}")
    for header in ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"):
        if header in headers:
            print(f"  {header}: {headers[header]}")
    if status == 200:
        choice = (payload.get("choices") or [{}])[0]
        print(f"  served by: {payload.get('model')}")
        print(f"  content: {((choice.get('message') or {}).get('content') or '').strip()!r}")
        print("\n=> The model answers. The 429s are intermittent provider errors.")
    else:
        error = payload.get("error") or {}
        message = error.get("message") if isinstance(error, dict) else error
        print(f"  error: {message}")
        if status == 429:
            print("\n=> Hard rate limit, not a blip. Set OPENROUTER_FALLBACK_MODELS")
            print("   to a paid model (drop the `:free` suffix) and redeploy.")
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
