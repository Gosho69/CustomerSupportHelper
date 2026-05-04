"""
Lightweight scheduler that runs auto_generate_reports daily at 23:55.

This replaces the need for cron or Celery Beat inside the Docker container.
It sleeps until the target time, runs the management command, then loops.

On every startup it runs a per-user catch-up pass:
  auto_generate_reports --catchup
This walks from each agent's account-creation date to today and generates any
missing weekly/monthly reports — so every user's history is always complete
regardless of how long the container was down, and new users never receive
reports for periods before they existed.
"""

import subprocess
import sys
import time
from datetime import datetime, timedelta

TARGET_HOUR = 23
TARGET_MINUTE = 55


# ─── helpers ─────────────────────────────────────────────────────────────────

def seconds_until_target():
    """Return seconds from now until the next occurrence of TARGET_HOUR:TARGET_MINUTE."""
    now = datetime.now()
    target = now.replace(hour=TARGET_HOUR, minute=TARGET_MINUTE, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def run_catchup() -> int:
    """
    Run the per-user catch-up pass on startup.
    For each active agent, generates every missing report from their join date
    to today.  Already-existing reports are silently skipped (idempotent).
    """
    result = subprocess.run(
        [sys.executable, "manage.py", "auto_generate_reports", "--catchup"],
        cwd="/app/backend/api",
        capture_output=False,
    )
    return result.returncode


def run_daily() -> int:
    """Run the daily trigger (today's date) — fires at 23:55 each night."""
    result = subprocess.run(
        [sys.executable, "manage.py", "auto_generate_reports"],
        cwd="/app/backend/api",
        capture_output=False,
    )
    return result.returncode


# ─── startup catch-up ────────────────────────────────────────────────────────

def run_startup_catchup():
    """
    Called once on startup.  Fills every agent's complete report history from
    their join date — not a fixed lookback window — so the result is the same
    whether the server has been running for one day or six months.
    """
    print("[scheduler] Startup catch-up: filling per-user report history …", flush=True)
    rc = run_catchup()
    if rc != 0:
        print(f"[scheduler] Catch-up exited with code {rc}", flush=True)
    else:
        print("[scheduler] Catch-up finished successfully.", flush=True)


# ─── main loop ───────────────────────────────────────────────────────────────

def main():
    print("[scheduler] Report scheduler started", flush=True)
    print(
        f"[scheduler] Will run auto_generate_reports daily at "
        f"{TARGET_HOUR:02d}:{TARGET_MINUTE:02d}",
        flush=True,
    )

    # Fill any gaps in every agent's history since they joined.
    run_startup_catchup()

    while True:
        wait = seconds_until_target()
        next_run = datetime.now() + timedelta(seconds=wait)
        print(
            f"[scheduler] Next run at {next_run.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(sleeping {int(wait)}s)",
            flush=True,
        )

        time.sleep(wait)

        print(f"[scheduler] Running auto_generate_reports at {datetime.now()}", flush=True)
        rc = run_daily()
        if rc != 0:
            print(f"[scheduler] Command exited with code {rc}", flush=True)
        else:
            print("[scheduler] Command completed successfully", flush=True)

        # Sleep 120 s to avoid re-triggering within the same minute.
        time.sleep(120)


if __name__ == "__main__":
    main()
