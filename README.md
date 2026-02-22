# AI Code Reviewer with Ollama & Cloudflare Tunnel

This project sets up a local AI Code Review bot that integrates with GitLab Merge Requests using Docker, Ollama, and Cloudflare Tunnel.

## Prerequisites

1.  **Docker & Docker Compose** installed.
2.  **GitLab Account** (or self-hosted instance).
3.  **Cloudflare Account** (for the tunnel).

## Setup

1.  **Clone this repository** (if you haven't already).
2.  **Copy the environment file:**
    ```bash
    cp .env.example .env
    ```
3.  **Configure `.env`:**
    *   `GITLAB_TOKEN`: Create a [Personal Access Token](https://gitlab.com/-/profile/personal_access_tokens) with `api` scope.
    *   `WEBHOOK_SECRET`: Generate a random string (e.g., `openssl rand -hex 12`).
    *   `TUNNEL_TOKEN`: See step 4.

4.  **Set up Cloudflare Tunnel:**
    *   Go to [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/).
    *   Navigate to **Networks > Tunnels** and create a new tunnel.
    *   Choose **Docker** as the environment.
    *   Copy the token command, but extract just the token string (the part after `--token`). Paste it into `.env`.
    *   **Configure the Public Hostname** in the Cloudflare dashboard:
        *   **Public Hostname:** `code-review.yourdomain.com` (or whatever you choose).
        *   **Service:** `http://app:5000` (The internal docker service name and port).

5.  **Start the services:**
    ```bash
    docker compose up -d
    ```

6.  **Pull the AI Model:**
    Wait for the containers to start, then run:
    ```bash
    docker compose exec ollama ollama pull codellama
    ```
    (You can swap `codellama` for `llama3`, `mistral`, etc., in `.env` and here).

7.  **Configure GitLab Webhook:**
    *   Go to your GitLab Project > **Settings > Webhooks**.
    *   **URL:** `https://code-review.yourdomain.com/webhook` (The public hostname you set in Cloudflare).
    *   **Secret Token:** The same `WEBHOOK_SECRET` from your `.env`.
    *   **Trigger:** Check **Merge request events**.
    *   Click **Add webhook**.

## Usage

Create a Merge Request in your GitLab project. The AI reviewer will automatically comment on the MR with feedback.

## Troubleshooting

*   **Logs:** Check logs with `docker compose logs -f`.
*   **Ollama:** Ensure the model is pulled (`docker compose exec ollama ollama list`).
*   **Tunnel:** Check Cloudflare dashboard to see if the tunnel is "Healthy".
