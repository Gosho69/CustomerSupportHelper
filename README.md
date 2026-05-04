# AgentSights

AI-powered platform for recording, transcribing, and analyzing customer support calls. Provides per-turn emotion tracking, vocal stress analysis, behavioral metrics, predictive CSAT scoring, automated coaching tips, performance reports, and dashboards to improve agent performance and customer experience.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [AI & ML Pipeline](#ai--ml-pipeline)
- [Features](#features)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Getting Started](#getting-started)
- [API Overview](#api-overview)
- [Authentication](#authentication)
- [Deployment](#deployment)

---

## Overview

AgentSights ingests audio recordings of support calls, runs them through a multi-stage ML pipeline, and surfaces structured insights for agents, team leads, and executives. The analysis runs asynchronously in the background via Celery so the upload endpoint returns immediately and the UI polls for results.

A built-in **Mock Call Center** module simulates an external telephony platform — recordings can be uploaded to it via API and are automatically synced into AgentSights every 5 minutes, enabling end-to-end integration testing without a real call center system.

---

## Architecture

```
Browser (Next.js)
       │
       ▼
Django REST API  ──► Celery Worker (AI Pipeline)
       │                    │
       ▼                    ▼
  PostgreSQL           Redis (broker)
 (Supabase)
                       ┌───────────────────────────────┐
                       │  AI Pipeline Stages            │
                       │  1. WhisperX transcription     │
                       │  2. PyAnnote diarization       │
                       │  3. Emotion analysis           │
                       │  4. Behavioral metrics         │
                       │  5. Summarization (GPT-4)      │
                       │  6. Coaching tips (Llama 3.2)  │
                       │  7. Topic analysis             │
                       │  8. Vocal stress analysis      │
                       │  9. CSAT prediction (ML)       │
                       └───────────────────────────────┘

Mock Call Center (separate Django app)
       │
       ▼  Celery Beat (every 5 min)
Django REST API ──► auto-import to Call model
```

All nine pipeline stages run in parallel inside the Celery worker via `ThreadPoolExecutor` (max 5 workers). CSAT prediction is a fire-and-forget post-analysis task that does not block call completion.

---

## Tech Stack

### Backend

| Layer | Technology |
|---|---|
| Framework | Django 5.2 + Django REST Framework 3.16 |
| Auth | djangorestframework-simplejwt 5.3 (JWT, HttpOnly cookies) |
| Task queue | Celery 5.3 |
| Message broker | Redis 7.2 |
| Database | PostgreSQL (Supabase) via psycopg2-binary |
| Server | Gunicorn (2 workers, 2 threads) |
| Static files | WhiteNoise |
| CORS | django-cors-headers |
| Email | Resend API |

### Frontend

| Layer | Technology |
|---|---|
| Framework | Next.js 16 (App Router) |
| Language | TypeScript |
| UI library | React 19 |
| State management | Zustand 5 |
| HTTP client | Axios |
| Charts | Recharts 3 |
| Icons | Lucide React |
| Styling | Tailwind CSS v4 |

### Infrastructure

| Component | Technology |
|---|---|
| Containerization | Docker + docker-compose |
| Local LLM serving | Ollama 0.6 |
| Model cache | HuggingFace Hub (volume-mounted) |

---

## AI & ML Pipeline

### 1. Speech Recognition — WhisperX + Faster-Whisper

- **Models**: `faster-whisper` with configurable size (base / medium / large-v2)
- **Library**: [WhisperX](https://github.com/m-bain/whisperX) + `ctranslate2` for optimized inference
- Produces per-word timestamps used for alignment and diarization

### 2. Speaker Diarization — PyAnnote 3.1

- **Model**: `pyannote/speaker-diarization-3.1` (HuggingFace, requires license acceptance)
- Assigns each utterance to either the **agent** or the **customer**
- Requires `HF_TOKEN` with accepted pyannote model terms

### 3. Emotion & Sentiment Analysis — HuggingFace Transformers

- **Emotion model**: `j-hartmann/emotion-english-distilroberta-base`
- **Sentiment model**: HuggingFace `sentiment-analysis` pipeline
- Runs on every turn of the transcript
- Outputs: emotion label, sentiment, emotional journey, customer satisfaction level, agent empathy score, overall call tone

### 4. Behavioral Metrics — Custom Analysis

Custom rule-based and statistical analysis producing:
- Words per minute (agent and customer separately)
- Interruption count and detection
- Silence duration, gaps, and percentage
- Response times (agent and customer)
- Question frequency and patterns
- Active listening acknowledgment detection
- Behavioral score (0–100)

### 5. Summarization — GPT-4o-mini / Fine-tuned FLAN-T5

Two modes, selected via `summarization_model` config:

| Mode | Model | Notes |
|---|---|---|
| `gpt4` (default) | `gpt-4o-mini` via OpenAI API | Requires `OPENAI_API_KEY` |
| `local` | Fine-tuned FLAN-T5 | Stored at `backend/AI_modules/model/final/` |

Extracts: issue summary, resolution status, helpfulness / respect / clarity / adherence / overall ratings (1–5 scale).

### 6. Coaching Tips — Ollama (Llama 3.2) / Rule-based Fallback

- **Primary**: Llama 3.2 via local Ollama server
- **Fallback**: Rule-based engine (triggers on rudeness, dismissiveness, interruptions, poor resolution, lack of active listening)
- Configurable via `OLLAMA_API_URL` and `OLLAMA_MODEL`

### 7. Topic Analysis

Extracts primary topics discussed in the call and tracks resolution rates per topic across historical calls.

### 8. Vocal Stress & Prosody Analysis — Custom Signal Processing

Per-utterance acoustic analysis using raw audio waveforms:
- **Features extracted**: energy (dB), zero-crossing rate (ZCR), spectral centroid, speech rate (WPM)
- **Stress scoring**: call-relative 0–1 scale using z-score normalization across all utterances
- **Pattern detection**: stress escalation (customer stress increasing toward call end) and calm recovery events
- Results stored in `Call.vocal_analysis` (JSONField)
- Frontend: "Voice Analysis" tab in call detail with a stress timeline chart
- No additional dependencies — uses `scipy` and `numpy` already in requirements

### 9. Predictive CSAT — scikit-learn Gradient Boosting

- **Model**: `GradientBoostingClassifier` trained on 10 features derived from emotional and behavioral metrics
- **Output**: predicted customer satisfaction score (1–5) and label stored in `Call.predicted_csat` / `Call.predicted_csat_label`
- **Cold-start fallback**: rule-based scoring when fewer than 30 labeled training examples exist
- **Retraining**: Celery Beat task runs every Sunday at 04:00 UTC to update the model with new call data
- Frontend: CSAT badge on call cards and full prediction display in call detail

---

## Features

- **Call Upload** — upload audio files; analysis runs in the background
- **Full Transcript** — turn-by-turn with timestamps, speaker labels, and emotion tags
- **Emotional Journey** — satisfaction trajectory from call start to end
- **Behavioral Dashboard** — WPM, silences, interruptions, talk ratio
- **Vocal Stress Analysis** — per-utterance acoustic stress scoring with timeline chart
- **AI Summaries** — structured summaries with quality ratings
- **Coaching Tips** — actionable AI-generated suggestions per call
- **Topic Tracking** — recurring issue identification and resolution rates
- **Predictive CSAT** — ML-predicted customer satisfaction score per call
- **Performance Reports** — weekly/monthly aggregated metrics per agent with team rankings, percentiles, strengths/weaknesses
- **Role-based Access** — Admin, Head of Department, Agent
- **Company Management** — multi-tenant organization support
- **Automated Reports** — scheduled daily at 23:55 UTC; backfills missed dates on startup (up to join date)
- **Mock Call Center** — built-in external telephony simulator with single/bulk upload and auto-sync
- **Call Center Portal** — frontend UI for the mock call center integration
- **Transactional Email** — email notifications via Resend for report delivery

---

## Project Structure

```
CustomerSupportHelper/
├── backend/
│   ├── api/
│   │   ├── calls/              # Call models, views, serializers, tasks
│   │   ├── users/              # Custom user model, JWT auth
│   │   ├── companies/          # Company/org management
│   │   ├── reports/            # Performance report generation
│   │   └── mock_callcenter/    # External call center simulator
│   ├── AI_modules/
│   │   ├── Whisperer/              # WhisperX transcription
│   │   ├── Emotion_analyzation/    # Emotion & sentiment analysis
│   │   ├── behaviour_analyzer/     # Behavioral metrics
│   │   ├── summarization/          # GPT-4 / FLAN-T5 summaries
│   │   ├── coaching_tips/          # Ollama / rule-based tips
│   │   ├── topic_analyzer/         # Topic extraction
│   │   ├── vocal_analysis/         # Vocal stress & prosody analysis
│   │   ├── csat_predictor/         # ML CSAT prediction & retraining
│   │   ├── orchestrator.py         # Main pipeline coordinator
│   │   └── model/final/            # Fine-tuned FLAN-T5 weights
│   ├── core/                   # Django settings, URLs, WSGI
│   ├── preload_models.py       # Warm-up script run at worker start
│   ├── scheduler_runner.py     # Daily report scheduler with catchup
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/app/
│   │   ├── dashboard/          # Protected routes
│   │   │   ├── page.tsx        # Main dashboard
│   │   │   ├── upload-call/    # Upload + analysis results
│   │   │   ├── calls/          # All calls list
│   │   │   ├── my-reports/     # Agent's own reports
│   │   │   ├── reports/        # All reports (admin/head)
│   │   │   ├── team/           # Team management
│   │   │   ├── users/          # User management
│   │   │   ├── companies/      # Company management
│   │   │   ├── profile/        # User profile
│   │   │   └── call-center/    # Mock call center portal
│   │   └── login/
│   ├── package.json
│   └── tsconfig.json
├── docker-compose.prod.yml
└── docker-compose.yml          # Minimal dev setup (Redis only)
```

---

## Environment Variables

### Backend

Copy `backend/.env.example` to `backend/.env` and fill in the values.

#### Required

| Variable | Description |
|---|---|
| `SECRET_KEY` | Django secret key |
| `DEBUG` | `True` for development, `False` for production |
| `ALLOWED_HOSTS` | Comma-separated allowed hostnames |
| `CORS_ALLOWED_ORIGINS` | Frontend origin(s), e.g. `http://localhost:3000` |
| `DB_NAME` | PostgreSQL database name |
| `DB_USER` | PostgreSQL user |
| `DB_PASSWORD` | PostgreSQL password |
| `DB_HOST` | PostgreSQL host (Supabase pooler URL) |
| `DB_PORT` | PostgreSQL port (e.g. `6543` for Supabase transaction mode) |
| `HF_TOKEN` | HuggingFace token — must have accepted pyannote model license |
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o-mini summaries |

#### Optional

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_API_URL` | Ollama API base URL | `http://ollama:11434` |
| `OLLAMA_URL` | Ollama service URL | `http://ollama:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llama3.2` |
| `DJANGO_LOG_LEVEL` | Django log level | `INFO` |
| `GITHUB_USERNAME` | GitHub username for Docker image pulls | — |
| `CALL_CENTER_INTEGRATION` | Enable mock call center sync (`mock` or `disabled`) | `disabled` |
| `MOCK_CALLCENTER_API_KEY` | API key for the mock call center endpoint | — |
| `CALL_SYNC_INTERVAL_SECONDS` | How often to sync external calls (seconds) | `300` |
| `RESEND_API_KEY` | Resend API key for transactional email | — |
| `RESEND_FROM_EMAIL` | Sender address for emails | — |
| `FRONTEND_URL` | Frontend URL used in email links | `http://localhost:3000` |

### Frontend

Copy `frontend/.env.example` to `frontend/.env.local` and fill in the values.

| Variable | Description |
|---|---|
| `NEXT_PUBLIC_API_URL` | Backend API base URL, e.g. `http://localhost:8000/api` |

---

## Getting Started

### Prerequisites

- Docker and Docker Compose
- A PostgreSQL database (Supabase free tier works)
- OpenAI API key
- HuggingFace account with pyannote model license accepted

### Local Development

```bash
# 1. Clone the repo
git clone https://github.com/your-org/CustomerSupportHelper.git
cd CustomerSupportHelper

# 2. Set up backend environment
cp backend/.env.example backend/.env
# Edit backend/.env with your values

# 3. Set up frontend environment
cp frontend/.env.example frontend/.env.local
# Edit NEXT_PUBLIC_API_URL if needed

# 4. Start all services
docker compose -f docker-compose.prod.yml up --build

# 5. The API will be available at http://localhost:8000
# 6. The frontend dev server at http://localhost:3000
```

On first boot the Celery worker preloads Whisper, PyAnnote, and the emotion models so the first call analysis doesn't pay a cold-start penalty. HuggingFace model weights are cached to a persistent Docker volume (`hf_cache`).

### Running Without Docker

```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

# In separate terminals:

# Celery worker
celery -A core worker --loglevel=info --concurrency=1

# Celery Beat (scheduled tasks — CSAT retraining, call sync, etc.)
celery -A core beat --loglevel=info

# Daily report scheduler
python scheduler_runner.py

# Frontend
cd frontend
npm install
npm run dev
```

---

## API Overview

All endpoints are prefixed with `/api/`.

### Calls

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/calls/upload/` | Upload audio file, triggers async analysis |
| `GET` | `/calls/my-calls/` | List calls for the current agent |
| `GET` | `/calls/queue-status/` | Get current analysis queue status |
| `GET` | `/calls/<id>/` | Retrieve call with full analysis results |
| `GET` | `/calls/<id>/status/` | Poll analysis status |

### Users

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/users/login/` | Obtain JWT tokens (set as HttpOnly cookies) |
| `POST` | `/users/logout/` | Logout and clear cookies |
| `POST` | `/users/token/refresh/` | Refresh access token |
| `GET` | `/users/me/` | Current user profile |
| `POST` | `/users/create-admin/` | Create admin user |
| `POST` | `/users/create-head/` | Create head of department |
| `POST` | `/users/create-agent/` | Create agent |
| `GET` | `/users/all/` | List all users |
| `GET` | `/users/heads/` | List heads of department |
| `GET` | `/users/subordinates/` | List subordinates of current user |
| `GET` | `/users/<id>/` | User detail |

### Reports

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reports/generate/` | Manually trigger report generation |
| `GET` | `/reports/my-reports/` | Current user's reports |
| `GET` | `/reports/agent/<id>/` | Reports for a specific agent |
| `GET` | `/reports/all/` | All reports (admin/head only) |
| `GET` | `/reports/<id>/` | Report detail |

### Companies

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/companies/create/` | Create company |
| `GET` | `/companies/all/` | List companies |
| `GET` | `/companies/<id>/` | Company detail |
| `POST` | `/companies/<id>/assign-head/` | Assign head to company |
| `GET` | `/companies/<id>/employees/` | List company employees |
| `POST` | `/companies/keywords/` | Manage company keywords |

### Mock Call Center

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/mock-callcenter/calls/upload/` | Upload a single recording |
| `POST` | `/mock-callcenter/calls/bulk-upload/` | Batch upload recordings |
| `GET` | `/mock-callcenter/calls/` | List calls (filter with `?analyzed=false`) |
| `GET` | `/mock-callcenter/calls/<uuid>/audio/` | Download call audio |
| `POST` | `/mock-callcenter/calls/<uuid>/` | Mark call as analyzed/imported |

Authenticate with `X-API-Key: <MOCK_CALLCENTER_API_KEY>` header.

### Other

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/version/` | Build SHA / version info |

---

## Authentication

- JWT tokens issued by `djangorestframework-simplejwt`
- Tokens stored as **HttpOnly, Secure, SameSite=None** cookies
- Access token lifetime: **1 hour**; Refresh token: **7 days** (auto-rotated)
- Custom `CookieJWTAuthentication` reads from the `access_token` cookie; falls back to the `Authorization: Bearer` header for API clients
- Three roles: **Admin**, **Head of Department**, **Agent** — permissions enforced per endpoint

---

## Deployment

The production setup is fully defined in `docker-compose.prod.yml`.

### Services

| Service | Image | Role |
|---|---|---|
| `redis` | `redis:7.2-alpine` | Celery message broker |
| `backend` | Custom Django image | REST API (Gunicorn) |
| `celery_worker` | Same Django image | AI analysis worker (concurrency 1) |
| `celery_beat` | Same Django image | Periodic task scheduler (CSAT retraining, call sync) |
| `scheduler` | Same Django image | Daily report scheduler (23:55 UTC) |
| `ollama` | `ollama/ollama:0.6.2` | Local Llama 3.2 serving |

### Volumes

| Volume | Purpose |
|---|---|
| `media_files` | Uploaded audio recordings |
| `hf_cache` | HuggingFace model weights (downloaded once) |
| `ollama_data` | Ollama model storage |
| `model` | Fine-tuned FLAN-T5 weights |

### Notes

- The Celery worker runs with `--concurrency 1` because the AI models are CPU/RAM intensive; it recycles after each task to prevent memory accumulation
- Model weights survive container restarts via persistent volumes — no re-download on redeploy
- The **scheduler** backfills missed report dates for every agent back to their join date on startup
- **Celery Beat** manages periodic tasks: CSAT model retraining (Sunday 04:00 UTC) and mock call center sync (every 5 minutes when `CALL_CENTER_INTEGRATION=mock`)
- All services include health checks with extended `start_period` to allow model preloading before traffic is accepted
