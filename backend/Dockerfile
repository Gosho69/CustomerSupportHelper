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

# Install PyTorch CPU-only (x86_64) — much smaller than the default CUDA wheel.
# Hetzner runs x86; no GPU needed for inference.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# torchaudio MUST be installed AFTER requirements.txt and from PyPI (not CPU index)
# because pyannote.audio 3.4+ requires torchaudio's AudioDecoder which needs ffmpeg
# bindings — the CPU-only wheels don't include them. Installing last ensures nothing
# overwrites it with the ffmpeg-less CPU-only version.
RUN pip install --no-cache-dir torchaudio

# ─────────────────────────────────────────────
# Stage 2: Runtime image
# ─────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root user
RUN groupadd -r app && useradd -r -g app app

WORKDIR /app

COPY --from=builder /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY . /app/backend/

RUN mkdir -p /app/backend/api/media /app/backend/api/staticfiles /app/.cache/huggingface && \
    chown -R app:app /app

USER app

ENV PYTHONPATH="/app/backend:/app/backend/api"
ENV DJANGO_SETTINGS_MODULE="api.settings"
# Redirect HuggingFace + PyTorch caches to /app which the app user owns
ENV HOME=/app
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

EXPOSE 8000

# Migrate, collect static, then start Gunicorn
# Workers: 2 (WhisperX + PyTorch use ~2–4 GB RAM each; Hetzner CX42 has 16 GB)
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
