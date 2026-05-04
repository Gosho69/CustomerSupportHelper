"""
purge_predated_reports
──────────────────────
One-off (and safe to re-run) command that deletes every PerformanceReport
whose period ended before the agent's account was created.

These reports are meaningless (they can only contain zeros) and were produced
by the backfill logic running for agents that didn't exist during those periods.

Usage:
    python manage.py purge_predated_reports           # dry-run — shows what would be deleted
    python manage.py purge_predated_reports --commit  # actually deletes them
"""

from django.core.management.base import BaseCommand
from django.db.models import F
from django.db.models.functions import TruncDate

from reports.models import PerformanceReport


class Command(BaseCommand):
    help = (
        "Delete PerformanceReport rows whose period ended before the agent was created. "
        "Runs as a dry-run by default; pass --commit to delete."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--commit",
            action="store_true",
            default=False,
            help="Actually delete the stale reports (default is dry-run).",
        )

    def handle(self, *args, **options):
        # Find reports where end_date < agent.created_at (as a date)
        stale = (
            PerformanceReport.objects
            .annotate(agent_created=TruncDate("agent__created_at"))
            .filter(end_date__lt=F("agent_created"))
            .select_related("agent")
        )

        count = stale.count()

        if count == 0:
            self.stdout.write(self.style.SUCCESS("No predated reports found. Nothing to do."))
            return

        self.stdout.write(f"Found {count} predated report(s):\n")
        for r in stale.order_by("agent__username", "start_date"):
            self.stdout.write(
                f"  [{r.id}] {r.agent.username} — {r.report_type} "
                f"{r.start_date} – {r.end_date}  "
                f"(account created {r.agent.created_at.date()})"
            )

        if options["commit"]:
            stale.delete()
            self.stdout.write(
                self.style.SUCCESS(f"\nDeleted {count} predated report(s).")
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    f"\nDry-run: nothing was deleted. "
                    f"Run with --commit to delete these {count} report(s)."
                )
            )
