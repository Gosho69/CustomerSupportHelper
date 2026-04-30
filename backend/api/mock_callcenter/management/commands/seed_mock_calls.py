"""
Management command to seed the mock call center with synthetic ExternalCall records.

Usage:
    python manage.py seed_mock_calls
    python manage.py seed_mock_calls --count 10
    python manage.py seed_mock_calls --count 3 --agent-email agent@company.com

This is a convenience tool for testing the sync pipeline without needing real audio.
The primary production-like flow is via POST /api/mock-callcenter/calls/upload/.
"""
import io
import math
import random
import struct
import uuid
import wave
from datetime import timedelta

from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone


def _generate_wav_bytes(duration_sec: int, sample_rate: int = 16000) -> bytes:
    """
    Generates a stereo 16-bit PCM WAV with two distinct sine tones (one per
    channel) to simulate a two-speaker call.
      Channel 0: 440 Hz (agent)
      Channel 1: 330 Hz (customer)

    Minimum 10 seconds enforced so WhisperX / pyannote have enough frames to
    detect speaker turns. Sample rate 16 kHz matches WhisperX's expectation.
    """
    num_frames = sample_rate * max(duration_sec, 10)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        frames = []
        for i in range(num_frames):
            t = i / sample_rate
            agent_sample = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * t))
            customer_sample = int(32767 * 0.3 * math.sin(2 * math.pi * 330 * t))
            frames.append(struct.pack('<hh', agent_sample, customer_sample))
        wf.writeframes(b''.join(frames))
    return buf.getvalue()


class Command(BaseCommand):
    help = (
        'Seed the mock call center with synthetic ExternalCall records. '
        'Each record gets a generated WAV audio file and is marked as unanalyzed '
        'so the sync task will pick it up automatically.'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--count',
            type=int,
            default=5,
            help='Number of synthetic calls to create (default: 5).',
        )
        parser.add_argument(
            '--agent-email',
            type=str,
            default=None,
            help=(
                'Restrict seeding to this specific agent email. '
                'If omitted, picks randomly from all active agents.'
            ),
        )

    def handle(self, *args, **options):
        from users.models import MyUser
        from mock_callcenter.models import ExternalCall

        count = options['count']
        agent_email = options.get('agent_email')

        agents_qs = MyUser.objects.filter(role='agent', is_active=True)
        if agent_email:
            agents_qs = agents_qs.filter(email=agent_email)

        agents = list(agents_qs)
        if not agents:
            if agent_email:
                raise CommandError(
                    f"No active agent found with email '{agent_email}'. "
                    "Make sure the agent exists and has role='agent'."
                )
            raise CommandError(
                "No active agents found in the database. "
                "Create at least one user with role='agent' first."
            )

        created = 0
        for _ in range(count):
            agent = random.choice(agents)
            duration = random.randint(30, 300)
            started_at = timezone.now() - timedelta(
                seconds=random.randint(3600, 86400 * 7)
            )

            wav_bytes = _generate_wav_bytes(duration)
            ext_id = uuid.uuid4()
            filename = f'seed_{ext_id}.wav'

            ec = ExternalCall(
                external_id=ext_id,
                agent_email=agent.email,
                caller_id=f'+1555{random.randint(1000000, 9999999)}',
                call_started_at=started_at,
                duration_seconds=duration,
            )
            ec.audio_file.save(filename, ContentFile(wav_bytes), save=False)
            ec.save()
            created += 1
            self.stdout.write(f"  Created ExternalCall {ext_id} for {agent.email}")

        self.stdout.write(
            self.style.SUCCESS(
                f"\nDone. Created {created} ExternalCall record(s). "
                "They will be imported automatically on the next sync cycle "
                "(or trigger manually: celery -A api call calls.integration_tasks.sync_external_calls)."
            )
        )
