import os
import logging
from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=0)
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
        if duration > 0:
            call.duration = duration
        call.status = 'completed'
        call.save()

        logger.info(f"analyze_call_task completed successfully for call_id={call_id}")

    except Exception as exc:
        logger.exception(f"analyze_call_task failed for call_id={call_id}")
        try:
            from .models import Call as CallModel
            call = CallModel.objects.get(id=call_id)
            call.status = 'failed'
            call.error_message = str(exc)[:1000]
            call.save(update_fields=['status', 'error_message'])
        except Exception:
            pass
        raise

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
