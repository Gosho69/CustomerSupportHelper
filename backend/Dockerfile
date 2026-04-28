# ─────────────────────────────────────────────
# Stage 1: Builder — install all Python deps
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev ffmpeg libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install PyTorch CPU-only BEFORE copying requirements.txt so it lives in its
# own cached layer. This layer is only invalidated when the torch version pin
# changes — NOT on every requirements.txt change (which saves ~15-20 min CI time).
# Hetzner runs x86; no GPU needed for inference.
# IMPORTANT: torch + torchaudio are NOT listed in requirements.txt to avoid pip
# re-resolving them from default PyPI and installing the CUDA build on top.
RUN pip install --upgrade pip && \
    pip install \
      "torch==2.5.1+cpu" \
      "torchaudio==2.5.1+cpu" \
      "torchvision==0.20.1+cpu" \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    # Re-enforce CPU-only torch: pyannote.audio 3.4+ pulls torchcodec which can
    # upgrade torch to a CUDA build from default PyPI.  Force CPU wheels back.
    pip install \
      "torch==2.5.1+cpu" \
      "torchaudio==2.5.1+cpu" \
      "torchvision==0.20.1+cpu" \
      --index-url https://download.pytorch.org/whl/cpu \
      --force-reinstall --no-deps

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

ENV PYTHONPATH="/app/backend:/app/backend/api:/app/backend/AI_modules"
ENV DJANGO_SETTINGS_MODULE="api.settings"
# Redirect HuggingFace + PyTorch caches to /app which the app user owns
ENV HOME=/app
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

EXPOSE 8000

# Startup sequence:
#   1. preload_models.py  — downloads PyAnnote + WhisperX models into the
#      hf_cache volume (fast no-op on subsequent boots; slow only on first run).
#      If pyannote fails (exit 2) we still start gunicorn in degraded mode.
#   2. migrate + collectstatic
#   3. gunicorn
# Workers: 2 (WhisperX + PyTorch use ~2–4 GB RAM each; Hetzner CX42 has 16 GB)
# Timeout: 600s (audio transcription + model warmup can take 3–5 min first time)
CMD ["sh", "-c", \
  "python /app/backend/preload_models.py; ret=$?; \
   if [ $ret -eq 1 ]; then echo '[startup] WARN: HF_TOKEN missing — diarization will be low quality'; fi; \
   cd /app/backend/api && \
   python manage.py migrate --noinput && \
   python manage.py collectstatic --noinput && \
   gunicorn api.wsgi:application \
     --bind 0.0.0.0:${PORT:-8000} \
     --workers 2 \
     --threads 2 \
     --timeout 600 \
     --max-requests 500 \
     --max-requests-jitter 50 \
     --log-file - \
     --access-logfile -"]
