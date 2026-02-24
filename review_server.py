import os
import logging
import requests
from ollama import Client
import threading
import gitlab
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configuration
GITLAB_URL = os.environ.get('GITLAB_URL', 'https://gitlab.com')
GITLAB_TOKEN = os.environ.get('GITLAB_TOKEN')
WEBHOOK_SECRET = os.environ.get('WEBHOOK_SECRET')
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://host.docker.internal:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'codellama')
logging.info(f"Resolved OLLAMA_HOST: {OLLAMA_HOST}")

logging.basicConfig(level=logging.INFO)

# Initialize Ollama client. Prefer local modes, fall back to host HTTP API.
client = None
in_docker = os.path.exists('/.dockerenv') or os.environ.get('IN_DOCKER') == '1'
init_attempts = [
    {'use_local': True},
    {'useLocal': True},
    {'local': True},
    {'host': OLLAMA_HOST},
    {}
]
for opts in init_attempts:
    try:
        client = Client(**opts)
        logging.info(f"Initialized Ollama Client with options: {opts}")
        break
    except TypeError:
        # Some client versions may not accept these kwargs; try the next
        continue
    except Exception as e:
        logging.debug(f"Ollama Client init with {opts} failed: {e}")

if client is None:
    logging.warning(
        "Could not initialize Ollama Client with local or host options. "
        "If Ollama is running on the macOS host and this app runs in Docker, "
        "set OLLAMA_HOST=http://host.docker.internal:11434 or run the server on the host."
    )

# Initialize GitLab client
if GITLAB_TOKEN:
    gl = gitlab.Gitlab(GITLAB_URL, private_token=GITLAB_TOKEN)
else:
    logging.warning("GITLAB_TOKEN not provided. Application will fail to authenticate.")

def get_ollama_review(prompt_payload):
    """Send the full file context and diff to Ollama for review."""
    
    prompt = f"""Act as a Principal Software Engineer. You are performing a code review. 
I will provide you with the FULL code of the files being modified for context, followed by the specific GIT DIFF.

### Contextual Instructions (Crucial):
1. **FOCUS:** ONLY critique the lines of code added or modified in the GIT DIFF. Use the FULL FILE solely to understand the surrounding context (like variable definitions, class structures, or imports).
2. **DELETIONS:** If the diff shows code being REMOVED (lines starting with `-`), do NOT flag those lines as "unused".
3. **DEDUPLICATION:** Do not create multiple points for the same root cause. 
4. **LGTM:** If the code in the diff is high-quality, respond ONLY with: "LGTM. The changes are clean and follow best practices."

### Response Format:
- **No Preamble:** Do not say "Here is my review" or "I have analyzed the code."
- **Tone:** Professional, direct, and actionable.
- **Severity Tags:** Use only: [BLOCKER], [SUGGESTION], or [NIT].

### Constraints:
- Max 300-500 words total, depends on context. 
- No emojis. 
- Provide code snippets for fixes only when necessary for clarity.

### CRITICAL RULES:
1. **NO UNUSED CODE SEARCHING:** Do NOT flag methods or variables as "unused" unless you are 100% certain they are not used anywhere in the FULL FILE CONTENT provided. 
2. **STAGED CHANGES:** If a method is defined in the file but not called yet, assume it is being exported for use in other files or is part of a larger feature. Do not mark it for removal.
3. **ONLY REVIEW THE DIFF:** Your critique must focus on the logic inside the GIT DIFF sections.

---
{prompt_payload}
"""
    
    try:
        if client:
            resp = client.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                stream=False,
                options={
                    "num_ctx": 8192,
                    "temperature": 0.0,
                    "num_predict": 800,
                    "top_p": 0.9,
                }
            )

            # Try to extract common response fields from the client result
            if hasattr(resp, 'response'):
                return resp.response
            elif isinstance(resp, dict):
                return resp.get('response') or resp.get('output') or resp.get('text') or str(resp)
            return str(resp)
        else:
            # Fallback to HTTP API if client isn't available
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 8192,
                        "temperature": 0.0,
                        "num_predict": 800,
                        "top_p": 0.9
                    }
                },
                timeout=600
            )
            response.raise_for_status()
            return response.json().get('response', 'No response from AI.')
    except Exception as e:
        logging.error(f"Error communicating with Ollama: {e}")
        return f"Error communicating with AI Reviewer: {e}"

@app.route('/webhook', methods=['POST'])
def webhook():
    # 1. Verify Secret Token (Immediate)
    token = request.headers.get('X-Gitlab-Token')
    if WEBHOOK_SECRET and token != WEBHOOK_SECRET:
        return jsonify({'error': 'Invalid token'}), 403

    event_type = request.headers.get('X-Gitlab-Event')
    data = request.json

    # 2. Identify the work to be done
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

    # 3. START BACKGROUND THREAD (The Fix)
    if project_id and mr_iid:
        # If possible, check the MR's source branch and ignore release/* branches
        try:
            if 'gl' in globals() and gl:
                project = gl.projects.get(project_id)
                mr_obj = project.mergerequests.get(mr_iid)
                source_branch = getattr(mr_obj, 'source_branch', '') or ''
                if 'release/' in source_branch:
                    logging.info(f"Ignoring MR !{mr_iid} from release branch {source_branch}")
                    return jsonify({'message': 'Ignored release branch'}), 200
        except Exception as e:
            logging.debug(f"Could not evaluate branch name before starting review: {e}")

        # We start the review process in a background thread
        thread = threading.Thread(target=review_merge_request, args=(project_id, mr_iid))
        thread.start()
        # Immediately return 202 Accepted to GitLab
        return jsonify({'message': 'Review started in background'}), 202

    return jsonify({'message': 'Ignored event'}), 200


def review_merge_request(project_id, mr_iid):
    project = gl.projects.get(project_id)
    mr = project.mergerequests.get(mr_iid)
    
    # Use the branch where the NEW code lives
    source_branch = mr.source_branch 
    changes = mr.changes().get('changes', [])
    
    prompt_payload = ""
    
    for change in changes:
        file_path = change['new_path']
        diff = change['diff']
        
        if change.get('deleted_file'):
            continue
            
        try:
            # Fetching the file from the branch being reviewed
            gl_file = project.files.get(file_path=file_path, ref=source_branch)
            full_file_content = gl_file.decode().decode('utf-8') 
            
            prompt_payload += f"\n--- FILE: {file_path} (Full Content from {source_branch}) ---\n"
            prompt_payload += f"```\n{full_file_content}\n```\n"
            prompt_payload += f"\n--- DIFF TO REVIEW: {file_path} ---\n"
            prompt_payload += f"```diff\n{diff}\n```\n"
            
        except Exception as e:
            logging.warning(f"Fallback: Could not fetch {file_path}. Error: {e}")
            prompt_payload += f"\n--- DIFF ONLY: {file_path} ---\n```diff\n{diff}\n```\n"

    if not prompt_payload.strip():
        return jsonify({'message': 'No changes found'}), 200

    # Pass this to your get_ollama_review function
    review_comment = get_ollama_review(prompt_payload)

    try:
        mr.notes.create({'body': f"## 🤖 AI Code Review\n\n{review_comment}"})
        logging.info(f"Review posted successfully to MR !{mr_iid}")
    except Exception as e:
        logging.error(f"Failed to post comment to GitLab: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
