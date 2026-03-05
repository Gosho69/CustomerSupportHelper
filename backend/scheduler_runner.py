"""
Lightweight scheduler that runs auto_generate_reports daily at 23:55.

This replaces the need for cron or Celery Beat inside the Docker container.
It sleeps until the target time, runs the management command, then loops.

On every startup it also runs a backfill pass covering the past BACKFILL_DAYS
days so that any reports missed while the container was down are generated
immediately rather than waiting until the next trigger window.
"""

import calendar
import subprocess
import sys
import time
from datetime import date as date_type
from datetime import datetime, timedelta

TARGET_HOUR = 23
TARGET_MINUTE = 55

# How many calendar days to look back for missed trigger dates on startup.
BACKFILL_DAYS = 60


# ─── helpers ─────────────────────────────────────────────────────────────────

def seconds_until_target():
    """Return seconds from now until the next occurrence of TARGET_HOUR:TARGET_MINUTE."""
    now = datetime.now()
    target = now.replace(hour=TARGET_HOUR, minute=TARGET_MINUTE, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _is_trigger_date(d: date_type) -> bool:
    """Return True if `d` is a Sunday or the last day of its month."""
    is_sunday = d.weekday() == 6
    is_last_day = d.day == calendar.monthrange(d.year, d.month)[1]
    return is_sunday or is_last_day


def get_backfill_dates(lookback_days: int = BACKFILL_DAYS):
    """
    Return a sorted list of past dates (up to `lookback_days` ago, exclusive of
    today) that are Sundays or last days of their month.  The management command
    is idempotent, so re-running it for already-generated reports is safe.
    """
    today = date_type.today()
    seen = set()
    result = []
    for offset in range(1, lookback_days + 1):
        d = today - timedelta(days=offset)
        if d not in seen and _is_trigger_date(d):
            seen.add(d)
            result.append(d)
    return sorted(result)


def run_command_for_date(d: date_type) -> int:
    """Run auto_generate_reports for a specific date. Returns the exit code."""
    result = subprocess.run(
        [sys.executable, "manage.py", "auto_generate_reports", "--date", str(d)],
        cwd="/app/backend/api",
        capture_output=False,
    )
    return result.returncode


def run_command() -> int:
    """Run the Django management command for today. Returns the exit code."""
    result = subprocess.run(
        [sys.executable, "manage.py", "auto_generate_reports"],
        cwd="/app/backend/api",
        capture_output=False,
    )
    return result.returncode


# ─── startup backfill ────────────────────────────────────────────────────────

def run_backfill():
    """
    Called once on startup.  Iterates over all trigger dates in the past
    BACKFILL_DAYS days and calls auto_generate_reports for each one.
    Already-existing reports are skipped automatically by the command itself.
    """
    dates = get_backfill_dates()
    if not dates:
        print("[scheduler] Backfill: no trigger dates found in the lookback window.", flush=True)
        return

    print(
        f"[scheduler] Backfill: found {len(dates)} trigger date(s) in the past "
        f"{BACKFILL_DAYS} days — checking for missing reports …",
        flush=True,
    )

    errors = 0
    for d in dates:
        print(f"[scheduler] Backfill: processing {d} …", flush=True)
        rc = run_command_for_date(d)
        if rc != 0:
            print(f"[scheduler] Backfill: command for {d} exited with code {rc}", flush=True)
            errors += 1

    if errors:
        print(f"[scheduler] Backfill finished with {errors} error(s).", flush=True)
    else:
        print("[scheduler] Backfill finished successfully.", flush=True)


# ─── main loop ───────────────────────────────────────────────────────────────

def main():
    print("[scheduler] Report scheduler started", flush=True)
    print(
        f"[scheduler] Will run auto_generate_reports daily at "
        f"{TARGET_HOUR:02d}:{TARGET_MINUTE:02d}",
        flush=True,
    )

    # Catch up on any reports that were missed while the container was down.
    run_backfill()

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
        rc = run_command()
        if rc != 0:
            print(f"[scheduler] Command exited with code {rc}", flush=True)
        else:
            print("[scheduler] Command completed successfully", flush=True)

        # Sleep 120 s to avoid re-triggering within the same minute.
        time.sleep(120)


if __name__ == "__main__":
    main()
