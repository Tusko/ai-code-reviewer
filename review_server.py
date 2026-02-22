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
    prompt = f"""You are a senior software engineer conducting a code review.
Please review the following git diff for bugs, security vulnerabilities, and code style issues.
Focus on critical issues and provide concise, actionable feedback.
Do not explain what the code does, just critique it.

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
            timeout=120
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
    if event_type != 'Merge Request Hook':
        return jsonify({'message': 'Ignored event type'}), 200

    data = request.json
    object_attributes = data.get('object_attributes', {})
    
    # We only care when an MR is opened or updated
    action = object_attributes.get('action')
    if action not in ['open', 'reopen', 'update']:
        return jsonify({'message': f'Ignored action: {action}'}), 200

    project_id = data.get('project', {}).get('id')
    mr_iid = object_attributes.get('iid')

    if not project_id or not mr_iid:
        return jsonify({'error': 'Missing project_id or mr_iid'}), 400

    try:
        # Fetch the MR
        project = gl.projects.get(project_id)
        mr = project.mergerequests.get(mr_iid)
        
        # Get the changes (diff)
        changes = mr.changes()
        diffs = changes.get('changes', [])
        
        # Combine diffs into a single string (limit size if needed)
        full_diff = ""
        for change in diffs:
            full_diff += f"File: {change['new_path']}\n"
            full_diff += change['diff'] + "\n\n"

        if not full_diff.strip():
            return jsonify({'message': 'No changes found to review'}), 200

        # Get AI Review
        logging.info(f"Requesting review for MR !{mr_iid} in project {project_id}. Diff size: {len(full_diff)} characters.")
        review_comment = get_ollama_review(full_diff)

        # Post the comment to the MR
        mr.notes.create({'body': f"## 🤖 AI Code Review\n\n{review_comment}"})
        logging.info(f"Posted review for MR !{mr_iid}")

        return jsonify({'message': 'Review posted successfully'}), 200

    except gitlab.exceptions.GitlabGetError as e:
        logging.error(f"GitLab API Error: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logging.exception("Unexpected error processing webhook")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
