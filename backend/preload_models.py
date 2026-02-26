#!/usr/bin/env python3
"""
preload_models.py — run ONCE at container startup, before gunicorn starts.

What it does:
  1. Downloads (or verifies cached) pyannote/speaker-diarization-3.1 model.
  2. Downloads (or verifies cached) WhisperX "medium" ASR model.
  3. Downloads the WhisperX alignment model for English.

All models land in HF_HOME (/app/.cache/huggingface) which is a named Docker
volume, so the download only happens on the very first boot.  Subsequent
restarts / deploys are instant.

Exit codes:
  0   — everything ready
  1   — HF_TOKEN missing / wrong format (pyannote will use synthetic fallback)
  2   — pyannote model download failed (operator action required)
"""

import os
import sys
import time

# ── resolve HF token ─────────────────────────────────────────────────────────
# We intentionally do NOT use dotenv here — docker-compose already passes all
# vars from the .env file as real environment variables.
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip().strip('"').strip("'")

if not HF_TOKEN:
    print(
        "[preload] ERROR: HF_TOKEN is not set.\n"
        "  PyAnnote speaker diarization requires a HuggingFace token.\n"
        "  Without it the system will fall back to the low-quality\n"
        "  silence-gap diarization and transcriptions will be WRONG.\n"
        "  → Add  HF_TOKEN=hf_...  to /opt/agentsights/.env and restart.",
        file=sys.stderr,
    )
    sys.exit(1)

if not HF_TOKEN.startswith("hf_"):
    print(
        f"[preload] ERROR: HF_TOKEN does not start with 'hf_' (got: {HF_TOKEN[:10]}...).\n"
        "  Check the value in /opt/agentsights/.env.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"[preload] HF_TOKEN detected (hf_...{HF_TOKEN[-4:]})", flush=True)

# ── 1. Download / verify pyannote model ──────────────────────────────────────
print("[preload] Loading pyannote/speaker-diarization-3.1 model …", flush=True)
t0 = time.time()

try:
    from pyannote.audio import Pipeline

    # Try current API first, fall back for older pyannote versions
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN,
        )

    elapsed = round(time.time() - t0, 1)
    print(f"[preload] ✓ PyAnnote diarization model ready ({elapsed}s)", flush=True)
    del pipeline  # free memory — gunicorn workers will reload independently

except Exception as exc:
    elapsed = round(time.time() - t0, 1)
    print(
        f"[preload] FATAL: Could not load pyannote/speaker-diarization-3.1 "
        f"after {elapsed}s: {type(exc).__name__}: {exc}\n"
        "\n"
        "  Most common causes:\n"
        "  1. You have NOT accepted the model licence on HuggingFace:\n"
        "     → Visit https://huggingface.co/pyannote/speaker-diarization-3.1\n"
        "       and click 'Accept' while logged in as the account that owns the token.\n"
        "     → Also accept: https://huggingface.co/pyannote/segmentation-3.0\n"
        "  2. HF_TOKEN belongs to a different HuggingFace account than the one\n"
        "     that accepted the licence.\n"
        "  3. Network timeout — retry with:  docker compose restart backend\n"
        "\n"
        "  The service will start but transcriptions will use the fallback\n"
        "  silence-gap diarization (low quality).",
        file=sys.stderr,
    )
    # Don't hard-fail: gunicorn starts and serves other endpoints.  The
    # transcription path will fall back gracefully.
    sys.exit(2)

# ── 2. Download / verify WhisperX ASR model ──────────────────────────────────
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")

print(f"[preload] Loading WhisperX '{WHISPER_MODEL}' ASR model …", flush=True)
t0 = time.time()

try:
    import whisperx

    asr_model = whisperx.load_model(WHISPER_MODEL, "cpu", compute_type="int8")
    elapsed = round(time.time() - t0, 1)
    print(f"[preload] ✓ WhisperX '{WHISPER_MODEL}' model ready ({elapsed}s)", flush=True)
    del asr_model

except Exception as exc:
    elapsed = round(time.time() - t0, 1)
    print(
        f"[preload] WARNING: WhisperX model load failed after {elapsed}s: "
        f"{type(exc).__name__}: {exc}\n"
        "  Model will be downloaded on first transcription request (slower first call).",
        file=sys.stderr,
    )

# ── 3. Download / verify alignment model (English) ───────────────────────────
print("[preload] Loading WhisperX alignment model (en) …", flush=True)
t0 = time.time()

try:
    import whisperx

    align_model, metadata = whisperx.load_align_model("en", "cpu")
    elapsed = round(time.time() - t0, 1)
    print(f"[preload] ✓ Alignment model ready ({elapsed}s)", flush=True)
    del align_model, metadata

except Exception as exc:
    elapsed = round(time.time() - t0, 1)
    print(
        f"[preload] WARNING: Alignment model load failed after {elapsed}s: "
        f"{type(exc).__name__}: {exc}",
        file=sys.stderr,
    )

print("[preload] All model checks complete — starting gunicorn …", flush=True)
sys.exit(0)
