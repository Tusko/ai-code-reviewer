import os
import re
import time
import logging
import requests
import threading
import gitlab
import random
from flask import Flask, request, jsonify

app = Flask(__name__)

meme_phrases = [
    "Ше й в'єбав",
    "Сцеплєр йобаний",
    "Та за шо?!",
    "Пішов нахуй, я директор",
    "Сука, руль вирвало",
    "Батько, тобі нормально?",
    "Якого хуя, блять, відбувається?",
    "Хуярить, як дурний",
    "Тобі пизда, тікай з городу",
    "А нахуя, а головне — навіщо?",
    "Єбать ти лох, звичайно",
    "Чисто на похуях",
    "Він єбанутий, єй-богу",
    "Всьо, пізда рулю",
    "Ну і хулі ти мені зробиш?",
    "Завали єбало, шеф працює",
    "Якого милого, блять?",
    "Тупо найкращий, єбать його в рот",
    "Шо ти лисий, плачеш?",
    "Це просто якийсь сюр, нахуй",
    "Nihuyasobi na oborot.",
    "Ти кому дзвониш, синок?",
    "Єбать, я роз'їбався",
    "Шо ти мелеш, сука?",
    "А я зараз вам покажу, звідки на Білорусь...",
    "Ну це вже повний пиздець, панове",
    "Він хуйні не скаже",
    "Якого прутня тут коїться?",
    "Всьо хуйня, давай по новій",
    "Ти шо, єбанувся туди лізти?",
    "Проєбали спалах",
    "Тягни кота за яйця, хулі ти чекаєш",
    "Нам пизда, але є нюанс",
    "Я єбав таку роботу",
    "Шо ти, блєдіна, накурився?",
    "Чисто під пивко потягне",
    "Тупо в гівно",
    "Сука, аж трисе",
    "Хулі ти вилупився?",
    "До пизди карі очі",
    "Та йди ти нахуй зі своїми порадами"
]

# ==========================================
# CONFIGURATION
# ==========================================
GITLAB_URL = os.environ.get('GITLAB_URL', 'https://gitlab.com')
GITLAB_TOKEN = os.environ.get('GITLAB_TOKEN')
WEBHOOK_SECRET = os.environ.get('WEBHOOK_SECRET')
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://host.docker.internal:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')
# 16 GB unified-memory budget: 8K ctx fits gemma4:12b / qwen2.5-coder:7b without swap.
OLLAMA_NUM_CTX = int(os.environ.get('OLLAMA_NUM_CTX', '8192'))
OLLAMA_NUM_PREDICT = int(os.environ.get('OLLAMA_NUM_PREDICT', '500'))
CONTEXT_WINDOW = int(os.environ.get('CONTEXT_WINDOW', '25'))
# Upper bound on user payload; effective cap is also derived from num_ctx − num_predict.
MAX_PROMPT_CHARS = int(os.environ.get('MAX_PROMPT_CHARS', '120000'))
# Reserve tokens so input + output fit inside num_ctx (avoids eval≈8 cutoffs).
PROMPT_TOKEN_BUFFER = int(os.environ.get('PROMPT_TOKEN_BUFFER', '128'))
MIN_OUTPUT_TOKENS = int(os.environ.get('MIN_OUTPUT_TOKENS', '30'))

# Thread lock to prevent overloading the Mac Mini M4 16GB RAM
review_lock = threading.Lock()

# ==========================================
# LOGGING SETUP
# ==========================================
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
LOG_FILE = os.environ.get('LOG_FILE')

handlers = [logging.StreamHandler()]
if LOG_FILE:
    handlers.append(logging.FileHandler(LOG_FILE))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=handlers,
)

logging.info(
    "Ollama config: host=%s model=%s num_ctx=%s num_predict=%s max_prompt_chars=%s context_window=%s",
    OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, MAX_PROMPT_CHARS, CONTEXT_WINDOW,
)

# ==========================================
# GITLAB CLIENT INIT
# ==========================================
gl = None
if GITLAB_TOKEN:
    gl = gitlab.Gitlab(GITLAB_URL, private_token=GITLAB_TOKEN)
else:
    logging.warning("GITLAB_TOKEN not provided. Application will fail to authenticate.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def annotate_diff_for_ai(diff_text):
    """
    Adds explicit labels to diff lines so the AI doesn't hallucinate 
    about deleted code still being active.
    """
    annotated_lines = []
    for line in diff_text.splitlines():
        if line.startswith('---') or line.startswith('+++'):
            annotated_lines.append(line)
        elif line.startswith('-'):
            # Додаємо явний маркер, щоб модель не чіплялася до синтаксису чи неявних змінних
            annotated_lines.append(line + " // ❌ [THIS LINE WAS DELETED]")
        elif line.startswith('+'):
            annotated_lines.append(line + " // ✨ [NEWLY ADDED LINE]")
        else:
            annotated_lines.append(line)
    return "\n".join(annotated_lines)
    
def get_surgical_context(full_text, diff_hunk, window=None):
    """
    Extracts multiple context chunks if there are multiple diff hunks.
    Window is in lines on each side of the hunk; tuned to OLLAMA_NUM_CTX budget.
    """
    if window is None:
        window = CONTEXT_WINDOW
    try:
        lines = full_text.splitlines()
        context_blocks = []
        
        # Знаходимо всі місця змін у diff-файлі
        matches = re.finditer(r'@@ -\d+,\d+ \+(\d+)(?:,\d+)? @@', diff_hunk)
        
        for match in matches:
            start_line = int(match.group(1))
            start_idx = max(0, start_line - 1 - window)
            end_idx = min(len(lines), start_line - 1 + window)
            
            chunk = "\n".join(lines[start_idx:end_idx])
            context_blocks.append(f"Lines {start_idx+1}-{end_idx}:\n{chunk}")
            
        if not context_blocks:
            return "Context extraction unavailable."
            
        return "\n...\n".join(context_blocks)
    except Exception as e:
        logging.debug(f"Failed to extract surgical context: {e}")
        return "Context extraction failed. Review diff directly."


def _estimate_tokens(text: str) -> int:
    """Conservative char→token estimate for code-heavy prompts."""
    return max(1, len(text) // 3)


def _effective_prompt_char_cap(system_prompt: str) -> int:
    """Max user-prompt chars that still leaves room for model output in num_ctx."""
    system_tokens = _estimate_tokens(system_prompt)
    input_token_budget = (
        OLLAMA_NUM_CTX - OLLAMA_NUM_PREDICT - system_tokens - PROMPT_TOKEN_BUFFER
    )
    if input_token_budget < 512:
        logging.warning(
            "num_ctx=%s leaves only ~%s input tokens after system+output reserve; "
            "raise OLLAMA_NUM_CTX or lower OLLAMA_NUM_PREDICT",
            OLLAMA_NUM_CTX, input_token_budget,
        )
    char_cap = max(2000, input_token_budget * 3)
    return min(MAX_PROMPT_CHARS, char_cap)


SYSTEM_PROMPT = """Act as a strict Principal Software Engineer code reviewer.
Your ONLY job is to find logic errors, security vulnerabilities (like SQL injections, XSS), or severe performance bugs in the ADDED (+) or MODIFIED lines.

### ABSOLUTE BANS (CRITICAL TO OBEY):
1. NEVER complain about "unused", "undeclared", or "missing" variables, methods, or imports. You only see a fragment of the file; assume they are used elsewhere.
2. NEVER complain about "duplicated methods" or "duplicate blocks". The diff format repeats context. Ignore it.
3. NEVER flag code formatting, missing docstrings, naming conventions, or style issues in the analyzed code.

### YOUR FORMATTING RULES:
1. Use rich Markdown formatting for your response (paragraphs, bold text, bullet points, and code blocks) so it is highly readable in GitLab.
2. Do NOT add any introductory or concluding remarks (like "Here is the review" or "Hope this helps").

### RESPONSE FORMAT
Review the code and output ONLY using this exact structure:

### 📄 `<file_path>`

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

EFFECTIVE_PROMPT_CHAR_CAP = _effective_prompt_char_cap(SYSTEM_PROMPT)
logging.info(
    "Prompt budget: effective_char_cap=%s (num_ctx=%s, num_predict=%s, max_prompt_chars=%s)",
    EFFECTIVE_PROMPT_CHAR_CAP, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, MAX_PROMPT_CHARS,
)


def _format_file_diff(file_path: str, diff: str) -> str:
    return (
        f"\n=== START DIFF TO REVIEW: {file_path} ===\n"
        f"{diff}\n"
        f"=== END DIFF TO REVIEW ===\n"
    )


def _format_file_block(file_path: str, context: str, diff: str, include_context: bool) -> str:
    block = ""
    if include_context and context:
        block += (
            f"\n=== START FILE CONTEXT: {file_path} ===\n"
            f"{context}\n"
            f"=== END FILE CONTEXT ===\n"
        )
    block += _format_file_diff(file_path, diff)
    return block


def build_review_prompt(file_chunks, include_context=True):
    """
    Build a prompt that fits the token budget. Diffs are prioritized over file context;
    later files are dropped before truncating mid-diff when possible.
    """
    max_chars = EFFECTIVE_PROMPT_CHAR_CAP
    parts = []
    skipped = []

    for chunk in file_chunks:
        block = _format_file_block(
            chunk["path"], chunk.get("context", ""), chunk["diff"], include_context,
        )
        candidate = "".join(parts) + block
        if len(candidate) <= max_chars:
            parts.append(block)
            continue

        # Drop context for this file and retry.
        if include_context:
            block = _format_file_block(chunk["path"], "", chunk["diff"], False)
            candidate = "".join(parts) + block
            if len(candidate) <= max_chars:
                parts.append(block)
                logging.info("Omitted file context for %s to fit prompt budget", chunk["path"])
                continue

        skipped.append(chunk["path"])
        logging.warning("Skipping %s — diff alone exceeds remaining prompt budget", chunk["path"])

    payload = "".join(parts)
    if skipped:
        payload += (
            f"\n... [SKIPPED {len(skipped)} FILE(S) — PROMPT BUDGET: "
            f"{', '.join(skipped[:5])}"
            f"{'...' if len(skipped) > 5 else ''}] ...\n"
        )

    if len(payload) > max_chars:
        logging.warning(
            "Prompt %s chars still exceeds cap %s after file pruning, truncating tail",
            len(payload), max_chars,
        )
        payload = (
            payload[: max_chars - 40]
            + "\n... [TRUNCATED FOR CONTEXT SAFETY] ..."
        )

    est_input_tokens = _estimate_tokens(SYSTEM_PROMPT) + _estimate_tokens(payload)
    logging.info(
        "Built review prompt: chars=%s est_input_tokens≈%s cap=%s context=%s files=%s/%s",
        len(payload), est_input_tokens, max_chars, include_context,
        len(parts), len(file_chunks),
    )
    return payload


def get_ollama_review(prompt_payload):
    """Sends the optimized payload to the local Ollama API."""
    try:
        logging.info(
            "Sending to Ollama chat API: model=%s prompt_chars=%s num_ctx=%s",
            OLLAMA_MODEL, len(prompt_payload), OLLAMA_NUM_CTX,
        )

        # 16 GB tuning: keep num_ctx low (default 8192). Run scripts/setup-ollama-host.sh
        # and restart Ollama so KV cache quant is active. After changing num_ctx, unload
        # the model (`ollama stop <model>`) or Ollama keeps the old context allocation.
        # Use /api/chat (NOT /api/generate) for harmony-style models; think=False strips
        # channel control tokens that /api/generate would leak into the review.
        started = time.monotonic()
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_payload},
                ],
                "think": False,
                "stream": False,
                "keep_alive": "24h",
                "options": {
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_predict": OLLAMA_NUM_PREDICT,
                    "num_batch": 128,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "repeat_penalty": 1.05,
                    "seed": 42,
                },
            },
            timeout=600,
        )
        if not response.ok:
            logging.error(
                "Ollama %s returned %s: %s",
                response.url,
                response.status_code,
                response.text[:500],
            )
        response.raise_for_status()

        elapsed = time.monotonic() - started
        data = response.json()
        prompt_eval = data.get('prompt_eval_count') or 0
        eval_count = data.get('eval_count') or 0
        logging.info(
            "Ollama done in %.1fs (prompt_eval=%s, eval=%s tokens)",
            elapsed, prompt_eval, eval_count,
        )

        remaining_ctx = OLLAMA_NUM_CTX - prompt_eval
        if remaining_ctx < MIN_OUTPUT_TOKENS:
            logging.warning(
                "Context nearly full: prompt_eval=%s num_ctx=%s (~%s tokens left for output)",
                prompt_eval, OLLAMA_NUM_CTX, remaining_ctx,
            )

        review_text = data.get('message', {}).get('content', '').strip()

        # Failsafe: strip any leaked harmony channel tokens if the model emits them.
        review_text = re.sub(r'<\|[^>]*\|?>', '', review_text).strip()

        if review_text == "The":
            review_text = random.choice(meme_phrases)

        if "[LGTM]" in review_text and not any(
            tag in review_text for tag in ["[BLOCKER]", "[SUGGESTION]", "[NIT]"]
        ):
            review_text = "LGTM. The changes are clean and follow best practices."

        return review_text, {"prompt_eval_count": prompt_eval, "eval_count": eval_count}

    except Exception as e:
        logging.error(f"Error communicating with Ollama: {e}")
        return f"Error communicating with AI Reviewer: {e}", {"prompt_eval_count": 0, "eval_count": 0}


def _response_looks_truncated(review_text: str, stats: dict) -> bool:
    if stats.get("eval_count", 0) < MIN_OUTPUT_TOKENS:
        return True
    if not review_text or review_text.startswith("Error communicating"):
        return False
    if "[LGTM]" in review_text:
        return False
    # Mid-header or mid-fence usually means generation hit the context wall.
    if review_text.rstrip().endswith(("`", "###", "**", "*")):
        return True
    return len(review_text) < 80

def review_merge_request(project_id, mr_iid):
    """Fetches changes, builds prompt, and posts review to GitLab."""
    with review_lock: # Prevents multiple MRs from crashing the RAM simultaneously
        try:
            project = gl.projects.get(project_id)
            mr = project.mergerequests.get(mr_iid)
            source_branch = mr.source_branch 
            changes = mr.changes().get('changes', [])
            
            file_chunks = []

            for change in changes:
                file_path = change['new_path']
                diff = change['diff']

                if change.get('deleted_file'):
                    continue

                context = ""
                try:
                    gl_file = project.files.get(file_path=file_path, ref=source_branch)
                    full_file_content = gl_file.decode().decode('utf-8')
                    context = get_surgical_context(full_file_content, diff)
                except Exception as e:
                    logging.warning(f"Fallback: Could not fetch {file_path}. Error: {e}")

                file_chunks.append({"path": file_path, "context": context, "diff": diff})

            if not file_chunks:
                logging.info(f"No changes found for MR !{mr_iid}")
                return

            prompt_payload = build_review_prompt(file_chunks, include_context=True)
            review_comment, stats = get_ollama_review(prompt_payload)

            if _response_looks_truncated(review_comment, stats):
                logging.warning(
                    "Review output looks truncated (eval=%s, len=%s); retrying diff-only",
                    stats.get("eval_count"), len(review_comment),
                )
                prompt_payload = build_review_prompt(file_chunks, include_context=False)
                review_comment, stats = get_ollama_review(prompt_payload)
                if _response_looks_truncated(review_comment, stats):
                    logging.error(
                        "Review still truncated after diff-only retry (eval=%s, preview=%r)",
                        stats.get("eval_count"), review_comment[:120],
                    )
                    review_comment = (
                        "⚠️ Review could not be completed: the MR diff is too large for the "
                        f"configured context window (`OLLAMA_NUM_CTX={OLLAMA_NUM_CTX}`). "
                        "Try a smaller MR, lower `CONTEXT_WINDOW`, or raise `OLLAMA_NUM_CTX`."
                    )

            mr.notes.create({'body': f"## 🤖 AI Code Review\n\n{review_comment}"})
            logging.info(f"Review posted successfully to MR !{mr_iid}")
            
        except Exception as e:
            logging.error(f"Critical error in background review thread: {e}")


# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/webhook', methods=['POST'])
def webhook():
    # Verify Secret Token
    token = request.headers.get('X-Gitlab-Token')
    if WEBHOOK_SECRET and token != WEBHOOK_SECRET:
        return jsonify({'error': 'Invalid token'}), 403

    event_type = request.headers.get('X-Gitlab-Event')
    data = request.json
    project_id = None
    mr_iid = None

    if event_type == 'Note Hook':
        attrs = data.get('object_attributes', {})
        if attrs.get('noteable_type') == 'MergeRequest' and '/review' in attrs.get('note', '').lower():
            project_id = data['project']['id']
            mr_iid = data['merge_request']['iid']
            
    elif event_type == 'Merge Request Hook':
        obj = data.get('object_attributes', {})
        if obj.get('action') in ['open', 'reopen', 'update']:
            project_id = data['project']['id']
            mr_iid = obj['iid']

    if project_id and mr_iid:
        # Branch validation failsafe
        try:
            if gl:
                project = gl.projects.get(project_id)
                mr_obj = project.mergerequests.get(mr_iid)
                if 'release/' in (getattr(mr_obj, 'source_branch', '') or ''):
                    return jsonify({'message': 'Ignored release branch'}), 200
        except Exception:
            pass

        # Fire and Forget Threading
        thread = threading.Thread(target=review_merge_request, args=(project_id, mr_iid))
        thread.start()
        
        return jsonify({'message': 'Review started in background'}), 202

    return jsonify({'message': 'Ignored event'}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)