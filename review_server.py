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

def get_ollama_review(diff_text):
    """Send the diff to Ollama for review."""
    prompt = f"""Act as a Principal Software Engineer. Review the following git diff for logic errors, security vulnerabilities, and maintainability.

### Contextual Instructions (Crucial):
1. **DELETIONS:** If the diff shows code being REMOVED (lines starting with `-`), do NOT flag those lines as "unused" or "needing removal." Assume the author is already cleaning them up.
2. **SCOPE:** Only critique the code that remains or is newly added. If a removal causes a breaking change elsewhere, flag the impact, not the deletion itself.
3. **DEDUPLICATION:** Do not create multiple points for the same root cause. Group related issues into a single concise bullet.
4. **LGTM:** If the code is high-quality or the cleanup is correct, respond ONLY with: "LGTM. The changes are clean and follow best practices."

### Response Format:
- **No Preamble:** Do not say "Here is my review" or "I have analyzed the code."
- **Tone:** Professional, direct, and actionable.
- **Address Author:** Use the name found in the patch; otherwise, start with "Reviewing the changes..."
- **Severity Tags:** Use only these three:
    - [BLOCKER]: Critical bugs, security risks, or broken logic.
    - [SUGGESTION]: Performance or architectural improvements.
    - [NIT]: Minor style or readability issues.

### Constraints:
- Max 300 words total. 
- No emojis. 
- Provide code snippets for fixes only when necessary for clarity.

---

Diff:
{diff_text}
"""
    
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
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
    full_diff = ''
    for change in changes:
        full_diff += f"File: {change['new_path']}\n{change['diff']}\n\n"
    if not full_diff.strip():
        return jsonify({'message': 'No changes to review'}), 200
    review_comment = get_ollama_review(full_diff)
    mr.notes.create({'body': f"## 🤖 AI Code Review\n\n{review_comment}"})
    return jsonify({'message': 'Review posted successfully'}), 200


def review_merge_request(project_id, mr_iid):
    project = gl.projects.get(project_id)
    mr = project.mergerequests.get(mr_iid)
    # Get the changes (diff)
    changes = mr.changes().get('changes', [])
    full_diff = ""
    for change in changes:
        full_diff += f"File: {change['new_path']}\n"
        full_diff += change['diff'] + "\n\n"
    if not full_diff.strip():
        return jsonify({'message': 'No changes to review'}), 200

    logging.info(f"Requesting review for MR !{mr_iid} in project {project_id}. Diff size: {len(full_diff)} characters.")
    review_comment = get_ollama_review(full_diff)
    mr.notes.create({'body': f"## 🤖 AI Code Review\n\n{review_comment}"})
    logging.info(f"Posted review for MR !{mr_iid}")
    return jsonify({'message': 'Review posted successfully'}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
