import sys
import os
from pathlib import Path
from django.core.management.base import BaseCommand
from calls.models import Call

# Ensure the AI_modules directory is on sys.path
backend_dir = Path(__file__).resolve().parents[4]  # .../backend
ai_modules_dir = backend_dir / "AI_modules"
if str(ai_modules_dir) not in sys.path:
    sys.path.insert(0, str(ai_modules_dir))

from behaviour_analyzer.behavioral_analyzer import behavioral_analyze_call  # type: ignore


class Command(BaseCommand):
    help = (
        "Backfill missing behavioral_analysis (and behavioral_score) for all calls "
        "that have a transcript but no behavioral_analysis or are missing the score key."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--call-id",
            type=int,
            help="Backfill a single call by its ID",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-compute behavioral_analysis even if it already exists",
        )

    def handle(self, *args, **options):
        if options["call_id"]:
            calls = Call.objects.filter(id=options["call_id"])
        else:
            calls = Call.objects.all()

        updated = 0
        skipped = 0
        errors = 0

        for call in calls:
            # Decide whether to process this call
            has_score = (
                call.behavioral_analysis
                and call.behavioral_analysis.get("behavioral_score") is not None
            )
            if has_score and not options["force"]:
                skipped += 1
                continue

            transcript = call.transcript
            if not transcript or not transcript.get("utterances"):
                self.stdout.write(
                    self.style.WARNING(
                        f"Call {call.id}: no transcript/utterances — skipping"
                    )
                )
                skipped += 1
                continue

            try:
                result = behavioral_analyze_call(transcript)
                call.behavioral_analysis = result
                call.save(update_fields=["behavioral_analysis"])
                score = result.get("behavioral_score", "?")
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Call {call.id}: behavioral_score = {score}"
                    )
                )
                updated += 1
            except Exception as exc:
                self.stdout.write(
                    self.style.ERROR(
                        f"Call {call.id}: error — {exc}"
                    )
                )
                errors += 1

        self.stdout.write(
            f"\nDone. Updated: {updated}  Skipped: {skipped}  Errors: {errors}"
        )
