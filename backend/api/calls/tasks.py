import os
import logging
import datetime
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.utils import timezone

logger = logging.getLogger(__name__)


def _mark_failed(call_id: int, reason: str) -> None:
    """Best-effort: write status=failed to DB. Never raises."""
    try:
        from .models import Call
        call = Call.objects.get(id=call_id)
        call.status = 'failed'
        call.error_message = reason[:1000]
        call.save(update_fields=['status', 'error_message'])
    except Exception:
        logger.exception(f"Could not mark call {call_id} as failed in DB")


@shared_task(
    bind=True,
    max_retries=0,
    acks_late=True,           # Only remove from queue after success/failure — NOT on receipt.
    reject_on_worker_lost=True,  # If the worker process is killed (OOM), requeue the task.
)
def analyze_call_task(
    self,
    call_id: int,
    temp_path: str,
    summarization_model: str,
    whisper_model_size: str,
    local_model_path: str,
    gpt4_max_tokens: int,
):
    from .models import Call
    from AI_modules.orchestrator.orchestrator import analyze_call

    try:
        call = Call.objects.get(id=call_id)
        call.status = 'processing'
        call.save(update_fields=['status'])

        analysis_results = analyze_call(
            audio_path=temp_path,
            summarization_model=summarization_model,
            whisper_model_size=whisper_model_size,
            device="cpu",
            compute_type="int8",
            local_model_path=local_model_path,
            gpt4_max_tokens=gpt4_max_tokens,
        )

        # Extract duration from transcript
        duration = 0
        transcript = analysis_results.get('transcript', {})
        if 'duration_sec' in transcript:
            duration = int(transcript['duration_sec'])
        elif 'utterances' in transcript and transcript['utterances']:
            last_utterance = max(transcript['utterances'], key=lambda u: u.get('end', 0))
            duration = int(last_utterance.get('end', 0))
        elif 'segments' in transcript and transcript['segments']:
            duration = int(transcript['segments'][-1].get('end', 0))

        call.transcript = analysis_results['transcript']
        call.transcript_summary = analysis_results['call_summary']
        call.emotional_analysis = analysis_results['emotion_analysis']
        call.emotional_summary = analysis_results['emotion_summary']
        call.behavioral_analysis = analysis_results['behavioral_analysis']
        call.behavioral_summary = analysis_results['behavioral_summary']
        call.coaching_tips = analysis_results['coaching_tips']
        call.topic_analysis = analysis_results['topic_analysis']
        call.vocal_analysis = analysis_results.get('vocal_analysis')
        if duration > 0:
            call.duration = duration
        call.status = 'completed'
        call.save()

        # Fire-and-forget CSAT prediction — failure must NOT affect the completed call.
        from .csat_tasks import predict_csat_for_call
        predict_csat_for_call.delay(call.id)

        logger.info(f"analyze_call_task completed successfully for call_id={call_id}")

    except SoftTimeLimitExceeded:
        logger.error(f"analyze_call_task soft time limit exceeded for call_id={call_id}")
        _mark_failed(call_id, "Analysis timed out — the audio file may be too long.")
        raise

    except Exception as exc:
        logger.exception(f"analyze_call_task failed for call_id={call_id}")
        _mark_failed(call_id, str(exc))
        raise

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@shared_task
def cleanup_stuck_calls():
    """
    Runs every 10 minutes via Celery Beat.
    Marks any call stuck in 'processing' for longer than the hard task
    time limit (35 min) as 'failed'. This catches worker OOM kills and
    any other crash that bypasses Python exception handlers.
    """
    from .models import Call

    cutoff = timezone.now() - datetime.timedelta(minutes=35)
    stuck = Call.objects.filter(status='processing', updated_at__lt=cutoff)
    count = stuck.count()
    if count:
        stuck.update(
            status='failed',
            error_message='Analysis timed out — worker may have run out of memory.',
        )
        logger.warning(f"cleanup_stuck_calls: marked {count} stuck call(s) as failed")
    return count
