"""
Celery task: sync_external_calls

Polls the configured call center integration for new, unanalyzed calls and
imports them into the main platform for analysis.

Scheduled via Celery Beat every CALL_SYNC_INTERVAL_SECONDS (default: 300s / 5 min).
Can also be triggered manually:
  celery -A api call calls.integration_tasks.sync_external_calls
"""
import logging
import os
import tempfile

from celery import shared_task
from django.conf import settings

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    reject_on_worker_lost=True,
)
def sync_external_calls(self):
    """
    Discover unanalyzed calls from the configured call center integration,
    import each as a Call record, and enqueue analysis.

    Returns a dict: {'imported': N, 'skipped': N, 'errors': N}
    Returns {'skipped': True} when integration is disabled.

    Idempotent: calls with an existing platform_call link are skipped.
    """
    from calls.call_center_integration import get_integration_service
    from calls.models import Call
    from users.models import MyUser

    integration = get_integration_service()
    if integration is None:
        logger.info("sync_external_calls: integration disabled (CALL_CENTER_INTEGRATION=disabled), skipping.")
        return {'skipped': True}

    # Fetch unanalyzed calls — retry the whole task if this fails
    try:
        unanalyzed = integration.get_unanalyzed_calls()
    except Exception as exc:
        logger.exception("sync_external_calls: failed to fetch unanalyzed calls from integration")
        raise self.retry(exc=exc)

    if not unanalyzed:
        logger.info("sync_external_calls: no unanalyzed calls found.")
        return {'imported': 0, 'skipped': 0, 'errors': 0}

    logger.info(f"sync_external_calls: found {len(unanalyzed)} unanalyzed call(s)")

    # Resolve paths once — same pattern as UploadCallView
    local_model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'AI_modules', 'model', 'final',
    )
    whisper_model_size = os.environ.get('WHISPER_MODEL', 'medium')

    imported = 0
    skipped = 0
    errors = 0

    for call_meta in unanalyzed:
        external_id = call_meta['external_id']

        # Idempotency guard: re-check DB state inside the loop.
        # ExternalCall.platform_call is a OneToOneField — if already linked, skip.
        try:
            from mock_callcenter.models import ExternalCall
            ext_call = ExternalCall.objects.get(external_id=external_id)
        except ExternalCall.DoesNotExist:
            logger.warning(f"sync_external_calls: ExternalCall {external_id} not found in DB, skipping")
            skipped += 1
            continue

        if ext_call.platform_call_id is not None:
            logger.debug(f"sync_external_calls: {external_id} already imported (platform_call={ext_call.platform_call_id}), skipping")
            skipped += 1
            continue

        # Match agent: must be an active agent-role user
        agent = MyUser.objects.filter(
            email=call_meta['agent_email'],
            role='agent',
            is_active=True,
        ).first()

        if agent is None:
            logger.warning(
                f"sync_external_calls: no active agent with email "
                f"{call_meta['agent_email']!r} — skipping call {external_id}"
            )
            skipped += 1
            continue

        # Download audio bytes from the integration
        try:
            audio_bytes = integration.get_audio_content(external_id)
        except Exception as exc:
            logger.error(f"sync_external_calls: audio download failed for {external_id}: {exc}")
            errors += 1
            continue

        # Write to a temp file in MEDIA_ROOT for the analysis pipeline.
        # analyze_call_task reads temp_path and deletes it in its finally block.
        try:
            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.wav',
                dir=settings.MEDIA_ROOT,
            )
            tmp.write(audio_bytes)
            tmp.close()
            temp_path = tmp.name
        except Exception as exc:
            logger.error(f"sync_external_calls: failed to write temp file for {external_id}: {exc}")
            errors += 1
            continue
        # Create the Call record and enqueue analysis. If anything fails after the
        # temp file was created but before the task is successfully scheduled,
        # remove the temp file to avoid orphaned files.
        try:
            from django.core.files.base import ContentFile
            audio_cf = ContentFile(audio_bytes, name=f'imported_{external_id}.wav')
            call = Call.objects.create(
                agent=agent,
                audio_file=audio_cf,
                status='pending',
            )

            # Enqueue the full analysis pipeline
            from calls.tasks import analyze_call_task
            analyze_call_task.delay(
                call_id=call.id,
                temp_path=temp_path,
                summarization_model='gpt4',
                whisper_model_size=whisper_model_size,
                local_model_path=local_model_path,
                gpt4_max_tokens=1500,
            )
        except Exception as exc:
            logger.error(f"sync_external_calls: failed to import/schedule analysis for {external_id}: {exc}")
            try:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                logger.exception("sync_external_calls: failed to remove temp file after error")
            errors += 1
            continue

        # Mark as analyzed in the external system
        try:
            integration.mark_as_analyzed(external_id)
        except Exception as exc:
            logger.warning(
                f"sync_external_calls: mark_as_analyzed failed for {external_id}: {exc}. "
                "The call has been imported — this only affects the external platform state."
            )

        # Link the FK back to the ExternalCall record.
        # OneToOneField: if two concurrent runs race here, the second raises IntegrityError.
        try:
            ext_call.platform_call = call
            ext_call.save(update_fields=['platform_call'])
        except Exception:
            logger.warning(
                f"sync_external_calls: could not link platform_call for {external_id} "
                "— likely a race condition with a concurrent sync run. Safe to ignore."
            )

        logger.info(
            f"sync_external_calls: imported ExternalCall {external_id} "
            f"→ Call id={call.id} for agent {agent.email}"
        )
        imported += 1

    result = {'imported': imported, 'skipped': skipped, 'errors': errors}
    logger.info(f"sync_external_calls: complete — {result}")
    return result
