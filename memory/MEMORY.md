# AgentSights Project Memory

## Project Overview
AI-powered call analysis platform. Django/DRF backend, Next.js/TypeScript frontend, Docker Compose deployment on Hetzner.

## Key Architecture
- Backend: `backend/api/` (Django), AI pipeline: `backend/AI_modules/`
- Frontend: `frontend/src/` (Next.js 16, Zustand, Axios)
- Production: `docker-compose.prod.yml` — backend, celery_worker, scheduler, redis, ollama
- PYTHONPATH set in Dockerfile: `/app/backend:/app/backend/api` — no sys.path hacks needed

## Auth Pattern (Post-Refactor)
- JWT stored in httpOnly cookies (`access_token`, `refresh_token`) set by Django login view
- Custom `CookieJWTAuthentication` in `backend/api/users/authentication.py` reads from cookie, falls back to Authorization header
- Axios uses `withCredentials: true` — no tokens in localStorage
- `authStore.ts` stores only `user` in memory (no persist, no tokens)
- `hydrateFromServer()` called on dashboard mount to re-fetch user after page reload
- `middleware.ts` checks `access_token` cookie for server-side route protection

## Async Call Processing (Post-Refactor)
- Upload returns 202 with `call_id` immediately
- Celery worker processes analysis in background
- Call model has `status` field: pending/processing/completed/failed
- Frontend polls `/calls/<id>/status/` every 5s until completed/failed

## User Preferences
- No emojis in output
- Concise responses
