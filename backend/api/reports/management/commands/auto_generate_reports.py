"""
auto_generate_reports
─────────────────────
Run this command daily (e.g. at 23:55 via cron / task-scheduler).
It inspects today's date and:

  • On Sundays       → generates a weekly  report for Mon–Sun (the week ending today)
  • On the last day
    of the month     → generates a monthly report for the 1st–last-day (the month ending today)

Both checks happen independently, so the last Sunday of a short month
(e.g. Feb 28 on a Sunday) triggers *both* reports.

The command is fully idempotent: it skips any report that already exists
for the exact date range.

Optional flags
──────────────
  --date   YYYY-MM-DD   Simulate a specific date instead of today
                        (useful for backfills and testing)
  --agent-id  <int>     Only process one agent (useful for manual reruns)
"""
from datetime import date as date_type
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from reports.models import PerformanceReport
from reports.report_factory import calculate_and_create_report
from reports.date_utils import (
    is_sunday,
    is_last_day_of_month,
    get_week_date_range,
    get_month_date_range,
)
from users.models import MyUser


class Command(BaseCommand):
    help = (
        "Automatically generate weekly reports on Sundays and monthly reports "
        "on the last day of the month for all active agents."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--date",
            type=str,
            default=None,
            help="Override today's date (YYYY-MM-DD). Defaults to today.",
        )
        parser.add_argument(
            "--agent-id",
            type=int,
            default=None,
            help="Only process this agent (by primary key).",
        )

    def handle(self, *args, **options):
        # ── Resolve the reference date ────────────────────────────────────
        if options["date"]:
            try:
                today = date_type.fromisoformat(options["date"])
            except ValueError:
                raise CommandError(
                    f"Invalid date format: '{options['date']}'. Use YYYY-MM-DD."
                )
        else:
            today = timezone.now().date()

        self.stdout.write(f"auto_generate_reports running for date: {today}")

        # ── Decide which period(s) need reports ───────────────────────────
        tasks = []  # list of (report_type, start_date, end_date)

        if is_sunday(today):
            start, end = get_week_date_range(today)
            tasks.append(("weekly", start, end))
            self.stdout.write(f"  → Sunday detected: will generate weekly report {start} – {end}")

        if is_last_day_of_month(today):
            start, end = get_month_date_range(today)
            tasks.append(("monthly", start, end))
            self.stdout.write(f"  → Last day of month: will generate monthly report {start} – {end}")

        if not tasks:
            self.stdout.write(
                self.style.WARNING(
                    f"  No reports due on {today} "
                    f"(not a Sunday, not the last day of the month). Nothing to do."
                )
            )
            return

        # ── Fetch agents ──────────────────────────────────────────────────
        if options["agent_id"]:
            agents = MyUser.objects.filter(
                id=options["agent_id"], role="agent", is_active=True
            )
        else:
            agents = MyUser.objects.filter(role="agent", is_active=True)

        if not agents.exists():
            self.stdout.write(self.style.WARNING("No active agents found. Exiting."))
            return

        # ── Process each period × each agent ────────────────────────────
        total_generated = 0
        total_skipped = 0
        total_errors = 0

        for report_type, start_date, end_date in tasks:
            self.stdout.write(
                f"\nProcessing {report_type} reports for {start_date} – {end_date} …"
            )
            generated = skipped = errors = 0

            for agent in agents:
                # Idempotency guard
                already_exists = PerformanceReport.objects.filter(
                    agent=agent,
                    report_type=report_type,
                    start_date=start_date,
                    end_date=end_date,
                ).exists()

                if already_exists:
                    self.stdout.write(
                        self.style.WARNING(
                            f"  SKIP  {agent.username} — {report_type} report already exists"
                        )
                    )
                    skipped += 1
                    continue

                try:
                    report = calculate_and_create_report(
                        agent=agent,
                        report_type=report_type,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"  OK    {agent.username} — {report_type} report created (ID: {report.id})"
                        )
                    )
                    generated += 1
                except Exception as exc:
                    self.stdout.write(
                        self.style.ERROR(
                            f"  ERROR {agent.username} — {exc}"
                        )
                    )
                    errors += 1

            self.stdout.write(
                f"  {report_type}: generated={generated}  skipped={skipped}  errors={errors}"
            )
            total_generated += generated
            total_skipped += skipped
            total_errors += errors

        # ── Summary ───────────────────────────────────────────────────────
        self.stdout.write(
            self.style.SUCCESS(
                f"\nDone. Total — generated: {total_generated}  "
                f"skipped: {total_skipped}  errors: {total_errors}"
            )
        )
