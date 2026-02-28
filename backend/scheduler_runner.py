"""
Lightweight scheduler that runs auto_generate_reports daily at 23:55.

This replaces the need for cron or Celery Beat inside the Docker container.
It sleeps until the target time, runs the management command, then loops.
"""

import subprocess
import sys
import time
from datetime import datetime, timedelta

TARGET_HOUR = 23
TARGET_MINUTE = 55


def seconds_until_target():
    """Return seconds from now until the next occurrence of TARGET_HOUR:TARGET_MINUTE."""
    now = datetime.now()
    target = now.replace(hour=TARGET_HOUR, minute=TARGET_MINUTE, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def run_command():
    """Run the Django management command."""
    result = subprocess.run(
        [sys.executable, "manage.py", "auto_generate_reports"],
        cwd="/app/backend/api",
        capture_output=False,
    )
    return result.returncode


def main():
    print("[scheduler] Report scheduler started", flush=True)
    print(f"[scheduler] Will run auto_generate_reports daily at {TARGET_HOUR:02d}:{TARGET_MINUTE:02d}", flush=True)

    while True:
        wait = seconds_until_target()
        next_run = datetime.now() + timedelta(seconds=wait)
        print(f"[scheduler] Next run at {next_run.strftime('%Y-%m-%d %H:%M:%S')} (sleeping {int(wait)}s)", flush=True)

        time.sleep(wait)

        print(f"[scheduler] Running auto_generate_reports at {datetime.now()}", flush=True)
        rc = run_command()
        if rc != 0:
            print(f"[scheduler] Command exited with code {rc}", flush=True)
        else:
            print("[scheduler] Command completed successfully", flush=True)

        # Sleep 120s to avoid re-triggering within the same minute
        time.sleep(120)


if __name__ == "__main__":
    main()
