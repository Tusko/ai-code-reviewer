FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY . .

# One worker only. The review queue and dedupe cache live in process memory;
# forking a second worker would allow two concurrent Ollama inferences and
# exhaust unified memory on a 16 GB host (defect B3).
# Threads handle webhook concurrency; actual reviews are serialized by the queue.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", \
     "--workers", "1", "--threads", "4", "review_server:app"]
