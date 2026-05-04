"""
auto_generate_reports
─────────────────────
Two modes of operation:

DAILY MODE (default / --date)
  Run this command daily at 23:55 via the scheduler.
  On Sundays       → generates a weekly  report for Mon–Sun (the week ending today).
  On the last day
  of the month     → generates a monthly report for the 1st–last (the month ending today).
  Both checks are independent, so the last Sunday of a short month triggers both.
  The command is idempotent: existing reports are silently skipped.

CATCH-UP MODE (--catchup)
  Used on startup (and safe to run any time).
  For EACH active agent individually, walks every calendar day from the agent's
  account-creation date up to today, finds every Sunday and every month-end in
  that window, and generates any missing weekly/monthly reports.
  Because the window is anchored to each agent's join date — not a fixed lookback —
  no agent ever receives reports from before they existed, and no report from their
  active history is missed regardless of how long the server was down.
  Email notifications are suppressed for historical reports (only current-period
  reports, where end_date >= today, generate a notification).

Optional flags
──────────────
  --date      YYYY-MM-DD  (daily mode) Simulate a specific date instead of today.
  --agent-id  <int>       Process only this agent (both modes).
  --catchup               Run catch-up mode instead of daily mode.
"""

from datetime import date as date_type, timedelta

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


def _trigger_tasks_for_date(d: date_type):
    """
    Return a list of (report_type, start_date, end_date) tuples that are due
    on date `d`.  A Sunday triggers a weekly report; the last day of the month
    triggers a monthly report; both can fire on the same day.
    """
    tasks = []
    if is_sunday(d):
        tasks.append(("weekly", *get_week_date_range(d)))
    if is_last_day_of_month(d):
        tasks.append(("monthly", *get_month_date_range(d)))
    return tasks


class Command(BaseCommand):
    help = (
        "Generate weekly (Sunday) and monthly (last-day-of-month) performance reports. "
        "Use --catchup on startup to fill every agent's history from their join date."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--date",
            type=str,
            default=None,
            help="(daily mode) Override today's date (YYYY-MM-DD).",
        )
        parser.add_argument(
            "--agent-id",
            type=int,
            default=None,
            help="Only process this agent (by primary key). Works in both modes.",
        )
        parser.add_argument(
            "--catchup",
            action="store_true",
            default=False,
            help=(
                "Catch-up mode: for each agent, generate every missing report "
                "from their account-creation date to today."
            ),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────────────────────

    def handle(self, *args, **options):
        if options["agent_id"]:
            agents = MyUser.objects.filter(
                id=options["agent_id"], role="agent", is_active=True
            )
        else:
            agents = MyUser.objects.filter(role="agent", is_active=True)

        if not agents.exists():
            self.stdout.write(self.style.WARNING("No active agents found. Exiting."))
            return

        if options["catchup"]:
            self._handle_catchup(agents)
        else:
            self._handle_daily(agents, options)

    # ──────────────────────────────────────────────────────────────────────────
    # Daily mode
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_daily(self, agents, options):
        if options["date"]:
            try:
                today = date_type.fromisoformat(options["date"])
            except ValueError:
                raise CommandError(
                    f"Invalid date format: '{options['date']}'. Use YYYY-MM-DD."
                )
        else:
            today = timezone.now().date()

        self.stdout.write(f"auto_generate_reports (daily) for date: {today}")

        tasks = _trigger_tasks_for_date(today)
        if not tasks:
            self.stdout.write(self.style.WARNING(
                f"  No reports due on {today} "
                f"(not a Sunday, not the last day of the month). Nothing to do."
            ))
            return

        total_generated = total_skipped = total_errors = 0

        for report_type, start_date, end_date in tasks:
            self.stdout.write(f"\nProcessing {report_type} {start_date} – {end_date} …")
            generated = skipped = errors = 0

            for agent in agents:
                result = self._generate_one(
                    agent, report_type, start_date, end_date,
                    send_notification=True,
                )
                if result == "generated":
                    generated += 1
                elif result == "skipped":
                    skipped += 1
                else:
                    errors += 1

            self.stdout.write(
                f"  {report_type}: generated={generated}  skipped={skipped}  errors={errors}"
            )
            total_generated += generated
            total_skipped += skipped
            total_errors += errors

        self.stdout.write(self.style.SUCCESS(
            f"\nDone. Total — generated: {total_generated}  "
            f"skipped: {total_skipped}  errors: {total_errors}"
        ))

    # ──────────────────────────────────────────────────────────────────────────
    # Catch-up mode  (per-user, anchored to join date)
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_catchup(self, agents):
        today = timezone.now().date()
        self.stdout.write(
            f"auto_generate_reports (catch-up) up to {today}\n"
            f"Each agent's history is filled from their own join date forward."
        )

        total_generated = total_skipped = total_errors = 0

        for agent in agents:
            join_date = agent.created_at.date()
            self.stdout.write(f"\n── {agent.username}  (joined {join_date}) ──")

            # Walk every day from the agent's join date to today
            current = join_date
            while current <= today:
                for report_type, start_date, end_date in _trigger_tasks_for_date(current):
                    # Only notify for reports whose period ends today or in the future
                    # (i.e. the most recent / current period). Historical ones are silent.
                    is_current = end_date >= today

                    result = self._generate_one(
                        agent, report_type, start_date, end_date,
                        send_notification=is_current,
                    )
                    if result == "generated":
                        total_generated += 1
                    elif result == "skipped":
                        total_skipped += 1
                    else:
                        total_errors += 1

                current += timedelta(days=1)

        self.stdout.write(self.style.SUCCESS(
            f"\nCatch-up done. "
            f"generated={total_generated}  "
            f"skipped={total_skipped}  "
            f"errors={total_errors}"
        ))

    # ──────────────────────────────────────────────────────────────────────────
    # Shared helper
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_one(
        self,
        agent: MyUser,
        report_type: str,
        start_date: date_type,
        end_date: date_type,
        *,
        send_notification: bool,
    ) -> str:
        """
        Try to generate a single report.  Returns 'generated', 'skipped', or 'error'.

        Skips if:
          • The period ended before the agent was created (no data can exist).
          • An identical report already exists (idempotent).
        """
        # Guard: period must not have ended before the agent was created.
        if end_date < agent.created_at.date():
            self.stdout.write(self.style.WARNING(
                f"  SKIP  {agent.username} — {report_type} {start_date}–{end_date} "
                f"predates account creation ({agent.created_at.date()})"
            ))
            return "skipped"

        # Idempotency: skip if the report already exists.
        if PerformanceReport.objects.filter(
            agent=agent,
            report_type=report_type,
            start_date=start_date,
            end_date=end_date,
        ).exists():
            self.stdout.write(self.style.WARNING(
                f"  SKIP  {agent.username} — {report_type} {start_date}–{end_date} already exists"
            ))
            return "skipped"

        try:
            report = calculate_and_create_report(
                agent=agent,
                report_type=report_type,
                start_date=start_date,
                end_date=end_date,
            )
            self.stdout.write(self.style.SUCCESS(
                f"  OK    {agent.username} — {report_type} {start_date}–{end_date} "
                f"(ID: {report.id})"
            ))

            if send_notification:
                try:
                    from notifications import send_report_notification
                    sent = send_report_notification(report)
                    if sent:
                        self.stdout.write(f"  MAIL  {agent.username} — notification sent")
                except Exception as mail_exc:
                    self.stdout.write(self.style.WARNING(
                        f"  MAIL  {agent.username} — notification failed: {mail_exc}"
                    ))

            return "generated"

        except Exception as exc:
            self.stdout.write(self.style.ERROR(
                f"  ERROR {agent.username} — {report_type} {start_date}–{end_date}: {exc}"
            ))
            return "error"
