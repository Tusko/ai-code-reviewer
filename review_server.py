import os
import logging
import requests
import gitlab
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configuration
GITLAB_URL = os.environ.get('GITLAB_URL', 'https://gitlab.com')
GITLAB_TOKEN = os.environ.get('GITLAB_TOKEN')
WEBHOOK_SECRET = os.environ.get('WEBHOOK_SECRET')
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://ollama:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'codellama')

logging.basicConfig(level=logging.INFO)

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

---
{prompt_payload}
"""
    
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 8192 # Increases context window for M4 Mac (requires ~8GB+ RAM)
                }
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json().get('response', 'No response from AI.')
    except Exception as e:
        logging.error(f"Error communicating with Ollama: {e}")
        return f"Error communicating with AI Reviewer: {e}"

@app.route('/webhook', methods=['POST'])
def webhook():
    # Verify Secret Token
    token = request.headers.get('X-Gitlab-Token')
    if WEBHOOK_SECRET and token != WEBHOOK_SECRET:
        return jsonify({'error': 'Invalid token'}), 403

    event_type = request.headers.get('X-Gitlab-Event')

    data = request.json

    # If it's a comment on an MR and the text contains e.g. "/review"
    if event_type == 'Note Hook':
        attrs = data.get('object_attributes', {})
        if attrs.get('noteable_type') == 'MergeRequest':
            note = attrs.get('note', '').strip().lower()
            if '/review' in note:
                project_id = data['project']['id']
                mr_iid = data['merge_request']['iid']

                # reuse the same code to fetch diffs & post a review
                return review_merge_request(project_id, mr_iid)
        return jsonify({'message': 'Note ignored'}), 200

    # existing MR‑opened/updated handling
    if event_type == 'Merge Request Hook':
        object_attributes = data.get('object_attributes', {})
        action = object_attributes.get('action')
        if action not in ['open', 'reopen', 'update']:
            return jsonify({'message': f'Ignored action: {action}'}), 200

        project_id = data['project']['id']
        mr_iid = object_attributes['iid']
        return review_merge_request(project_id, mr_iid)

    return jsonify({'message': 'Ignored event type'}), 200


def review_merge_request(project_id, mr_iid):
    project = gl.projects.get(project_id)
    mr = project.mergerequests.get(mr_iid)
    changes = mr.changes().get('changes', [])
    
    prompt_payload = ""
    
    for change in changes:
        file_path = change['new_path']
        diff = change['diff']
        
        # Skip fully deleted files (nothing to review)
        if change.get('deleted_file'):
            continue
            
        try:
            # Fetch the entire file content from the MR's source branch
            gl_file = project.files.get(file_path=file_path, ref=mr.source_branch)
            # decode() gets the bytes, the second decode('utf-8') turns it into a string
            full_file_content = gl_file.decode().decode('utf-8') 
            
            prompt_payload += f"\n### FILE CONTEXT: {file_path} ###\n```\n{full_file_content}\n```\n"
            prompt_payload += f"\n### GIT DIFF: {file_path} ###\n```diff\n{diff}\n```\n"
            
        except gitlab.exceptions.GitlabGetError:
            # Fallback if the file can't be fetched (e.g., brand new files might behave differently)
            logging.warning(f"Could not fetch full file {file_path}. Using diff only.")
            prompt_payload += f"\n### GIT DIFF (No Context Found): {file_path} ###\n```diff\n{diff}\n```\n"

    if not prompt_payload.strip():
        return jsonify({'message': 'No changes to review'}), 200

    logging.info(f"Requesting review for MR !{mr_iid}. Payload size: {len(prompt_payload)} chars.")
    
    review_comment = get_ollama_review(prompt_payload)
    
    mr.notes.create({'body': f"## 🤖 AI Code Review\n\n{review_comment}"})
    logging.info(f"Posted review for MR !{mr_iid}")
    
    return jsonify({'message': 'Review posted successfully'}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
