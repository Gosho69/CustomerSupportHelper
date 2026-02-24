# ─────────────────────────────────────────────
# Stage 1: Builder — install all Python deps
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev ffmpeg libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install PyTorch CPU-only first (smaller; Railway runs on CPU by default)
# Switch to the CUDA index URL if you upgrade to a Railway GPU plan
RUN pip install --upgrade pip && \
    pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Stage 2: Runtime image
# ─────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root user
RUN groupadd -r app && useradd -r -g app app

WORKDIR /app

COPY --from=builder /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY . /app/backend/

RUN mkdir -p /app/backend/api/media /app/backend/api/staticfiles && \
    chown -R app:app /app

USER app

ENV PYTHONPATH="/app/backend:/app/backend/api"
ENV DJANGO_SETTINGS_MODULE="api.settings"

EXPOSE 8000

# Migrate, collect static, then start Gunicorn
# Workers: 2 (WhisperX + PyTorch use ~2–4 GB RAM each; 2 workers fits 8 GB Railway plan)
# Timeout: 300s (long audio transcription can take 2–3 min)
CMD ["sh", "-c", \
  "cd /app/backend/api && \
   python manage.py migrate --noinput && \
   python manage.py collectstatic --noinput && \
   gunicorn api.wsgi:application \
     --bind 0.0.0.0:${PORT:-8000} \
     --workers 2 \
     --threads 2 \
     --timeout 300 \
     --max-requests 500 \
     --max-requests-jitter 50 \
     --log-file - \
     --access-logfile -"]
