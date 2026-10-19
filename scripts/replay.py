#!/usr/bin/env python3
"""Replay commits of this repo through draft_file and count what would post.

    .venv/bin/python scripts/replay.py .omc/replay/baseline.jsonl
    .venv/bin/python scripts/replay.py out.jsonl 332d017 b55d4bf

No GitLab: each commit stands in for an MR, file context comes from git. The
voice is off, so the counts are the dry review. The review model is forced to
--model (default: the config default) because a local .env may be stale.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHAS = "332d017 b55d4bf 9698c19 8f54fcf 929144a eebe994 7aee763 0e1c210".split()
TAG_RE = re.compile(r"\[(BLOCKER|SUGGESTION|NIT)\]")


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
        pass


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout


def changes(sha):
    """mr.changes()-shaped dicts for one commit."""
    out = []
    for chunk in re.split(r"^(?=diff --git )", git("show", "--format=", "--no-color", sha), flags=re.M):
        if not chunk.startswith("diff --git "):
            continue
        old = re.search(r"^--- (?:a/)?(.+)$", chunk, re.M)
        new = re.search(r"^\+\+\+ (?:b/)?(.+)$", chunk, re.M)
        if not old or not new:
            continue
        out.append({
            "old_path": old.group(1), "new_path": new.group(1),
            "new_file": old.group(1) == "/dev/null",
            "deleted_file": new.group(1) == "/dev/null",
            "renamed_file": False, "diff": chunk,
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("shas", nargs="*", default=DEFAULT_SHAS)
    parser.add_argument("--model", default="anthropic/claude-sonnet-5")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    os.environ["OPENROUTER_REVIEW_MODEL"] = args.model
    os.environ["SNARK"] = "false"
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set")
    sys.path.insert(0, str(ROOT))
    from reviewer import config, pipeline
    from reviewer.diff_parser import file_diff_from_change

    jobs = []
    for sha in args.shas:
        kept, _ = pipeline.partition_reviewable(
            [file_diff_from_change(c) for c in changes(sha)],
        )
        jobs.extend((sha, fd) for fd in kept)

    def draft(job):
        sha, fd = job
        context = git("show", f"{sha}:{fd.new_path}") if config.INCLUDE_FILE_CONTEXT else ""
        return pipeline.draft_file(fd, context)

    with ThreadPoolExecutor(max_workers=config.REVIEW_CONCURRENCY) as pool:
        results = list(pool.map(draft, jobs))

    tags, per_mr, statuses = Counter(), Counter(), Counter()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for (sha, fd), result in zip(jobs, results):
            if isinstance(result, pipeline.FileOutcome):
                statuses[result.status] += 1
                continue
            statuses["reviewed"] += 1
            for line, text in result.comments:
                per_mr[sha] += 1
                found = TAG_RE.findall(text) or ["NONE"]
                tags.update(found)
                handle.write(json.dumps({
                    "sha": sha, "path": fd.new_path, "line": line,
                    "tag": ",".join(found), "text": text,
                }, ensure_ascii=False) + "\n")

    print(f"model: {config.OPENROUTER_REVIEW_MODEL}")
    print(f"files: {len(jobs)} {dict(statuses)}")
    print(f"comments: {sum(per_mr.values())}")
    print(f"findings by tag: {dict(tags)}")
    print("comments per MR: " + " ".join(f"{sha}={per_mr[sha]}" for sha in args.shas))


if __name__ == "__main__":
    main()
