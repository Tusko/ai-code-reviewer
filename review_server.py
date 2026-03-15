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
def get_surgical_context(full_text, diff_hunk, window=30):
    """
    Extracts only the relevant lines around the diff to prevent prompt truncation.
    Saves massive amounts of memory and ensures the AI actually reads the system prompt.
    """
    try:
        # Find the line number where the change starts (e.g., @@ -14,5 +14,6 @@)
        match = re.search(r'@@ -\d+,\d+ \+(\d+)(?:,\d+)? @@', diff_hunk)
        if not match:
            return "Context extraction unavailable."

        start_line = int(match.group(1))
        lines = full_text.splitlines()

        # Calculate a 30-line window above and below the change
        start_idx = max(0, start_line - 1 - window)
        end_idx = min(len(lines), start_line - 1 + window)

        return "\n".join(lines[start_idx:end_idx])
    except Exception as e:
        logging.debug(f"Failed to extract surgical context: {e}")
        return "Context extraction failed. Review diff directly."

def get_ollama_review(prompt_payload):
    """Sends the optimized payload to the local Ollama API."""
    prompt = f"""Act as a Principal Software Engineer performing a strict code review. 
You will receive "Surgical Context" (a small chunk of the file) and the GIT DIFF hunks (the actual changes).

### CORE DIRECTIVE: NO SUMMARIES
1. DO NOT explain what the code does. 
2. DO NOT summarize the pull request, the diff, or the file. 
3. DO NOT output preamble, intro, or concluding remarks.
Your SOLE PURPOSE is to find specific defects in the ADDED (+) or MODIFIED lines. If you find no actionable defects, you must only output "LGTM."

### What to Review For
Critique the DIFF strictly for:
- Logic errors, unhandled edge cases, or potential crashes.
- Security vulnerabilities (e.g., injection, bad validation).
- Performance bottlenecks (e.g., N+1 queries, inefficient loops).

### Crucial Constraints
1. **CONTEXT ONLY:** Use the Surgical Context solely to understand surrounding definitions. DO NOT review lines outside the diff.
2. **DELETIONS:** Ignore removed lines (starting with `-`). Do not flag them as "unused".
3. **UNUSED CODE:** DO NOT flag methods or variables as "unused". The code provided is only a fragment of the application. Assume it is used elsewhere.
4. **HALLUCINATIONS:** ONLY mention files explicitly listed in the diff headers.
5. **NO FORMATTING NITS:** DO NOT flag whitespace, comma spacing, or other style issues. The project uses ESLint/Prettier for formatting. Focus only on logic, security, and performance.

### Strict Response Format
You must respond ONLY using the exact structure below. 

FILE: <file_path>
- [BLOCKER] One-sentence description of the critical bug. Code fix.
- [SUGGESTION] One-sentence description of the issue. Code fix.
- [NIT] One-sentence description of styling issue.

If there are NO actionable issues across ALL diffs, output exactly:
LGTM. The changes are clean and follow best practices.
---
{prompt_payload}
"""
    
    try:
        logging.info("Sending optimized prompt to Ollama HTTP API...")
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 8192,
                    "temperature": 0.0,     # Zero creativity, strict facts only
                    "num_predict": 800,     # Prevent endless rambling
                    "top_p": 0.9
                }
            },
            timeout=600 # 10 minute timeout
        )
        response.raise_for_status()
        return response.json().get('response', 'No response from AI.')
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
                    
                    prompt_payload += f"\n--- FILE: {file_path} (Context Window) ---\n"
                    prompt_payload += f"```\n{surgical_context}\n```\n"
                    prompt_payload += f"\n--- DIFF TO REVIEW: {file_path} ---\n"
                    prompt_payload += f"```diff\n{diff}\n```\n"
                    
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