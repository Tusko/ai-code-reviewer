import os
import re
import logging
import requests
import threading
import gitlab
from flask import Flask, request, jsonify

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
GITLAB_URL = os.environ.get('GITLAB_URL', 'https://gitlab.com')
GITLAB_TOKEN = os.environ.get('GITLAB_TOKEN')
WEBHOOK_SECRET = os.environ.get('WEBHOOK_SECRET')
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://host.docker.internal:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')

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

logging.info(f"Resolved OLLAMA_HOST: {OLLAMA_HOST}, OLLAMA_MODEL: {OLLAMA_MODEL}")

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
    
def get_surgical_context(full_text, diff_hunk, window=30):
    """
    Extracts multiple context chunks if there are multiple diff hunks.
    """
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

def get_ollama_review(prompt_payload):
    """Sends the optimized payload to the local Ollama API."""
    
    system_prompt = """Act as a strict Principal Software Engineer code reviewer.
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
    
    try:
        logging.info("Sending optimized prompt to Ollama...")

        #logging.info(f"========== PAYLOAD TO OLLAMA ==========\n{prompt_payload}\n=======================================")

        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "system": system_prompt,
                "prompt": prompt_payload,
                "stream": False,
                "options": {
                    "num_ctx": 8192,
                    "temperature": 0.1,
                    "num_predict": 500, # Зменшено, бо нам не потрібні довгі роздуми
                    "top_p": 0.9
                }
            },
            timeout=600
        )
        response.raise_for_status()
        
        review_text = response.json().get('response', '').strip()
        
        # Якщо модель відповіла тільки LGTM
        if "[LGTM]" in review_text and "[BLOCKER]" not in review_text and "[SUGGESTION]" not in review_text and "[NIT]" not in review_text:
            return "LGTM. The changes are clean and follow best practices."

        return review_text

    except Exception as e:
        logging.error(f"Error communicating with Ollama: {e}")
        return f"Error communicating with AI Reviewer: {e}"

def review_merge_request(project_id, mr_iid):
    """Fetches changes, builds prompt, and posts review to GitLab."""
    with review_lock: # Prevents multiple MRs from crashing the RAM simultaneously
        try:
            project = gl.projects.get(project_id)
            mr = project.mergerequests.get(mr_iid)
            source_branch = mr.source_branch 
            changes = mr.changes().get('changes', [])
            
            prompt_payload = ""
            
            for change in changes:
                file_path = change['new_path']
                diff = change['diff']
                
                if change.get('deleted_file'):
                    continue
                    
                try:
                    gl_file = project.files.get(file_path=file_path, ref=source_branch)
                    full_file_content = gl_file.decode().decode('utf-8') 
                    
                    # APPLY SURGICAL CONTEXT HERE
                    surgical_context = get_surgical_context(full_file_content, diff)
                    
                    # prompt_payload += f"\n--- FILE: {file_path} (Context Window) ---\n"
                    # prompt_payload += f"```\n{surgical_context}\n```\n"
                    # prompt_payload += f"\n--- DIFF TO REVIEW: {file_path} ---\n"
                    # prompt_payload += f"```diff\n{diff}\n```\n"
                    prompt_payload += f"\n=== START FILE CONTEXT: {file_path} ===\n"
                    prompt_payload += f"{surgical_context}\n"
                    prompt_payload += f"=== END FILE CONTEXT ===\n"
                    
                    prompt_payload += f"\n=== START DIFF TO REVIEW: {file_path} ===\n"
                    prompt_payload += f"{diff}\n"
                    prompt_payload += f"=== END DIFF TO REVIEW ===\n"
                    
                except Exception as e:
                    logging.warning(f"Fallback: Could not fetch {file_path}. Error: {e}")
                    prompt_payload += f"\n--- DIFF ONLY: {file_path} ---\n```diff\n{diff}\n```\n"

            if not prompt_payload.strip():
                logging.info(f"No changes found for MR !{mr_iid}")
                return

            # Check if payload is still too large (failsafe)
            if len(prompt_payload) > 25000:
                prompt_payload = prompt_payload[:25000] + "\n... [TRUNCATED FOR MEMORY SAFETY] ..."

            review_comment = get_ollama_review(prompt_payload)

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