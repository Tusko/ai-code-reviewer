FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use gunicorn for production. 
# --timeout 300 matches our internal 5-minute timeout for Ollama.
# --workers 2 allows handling a few concurrent webhooks.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "--workers", "2", "review_server:app"]
