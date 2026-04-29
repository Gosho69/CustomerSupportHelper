import logging
from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task
def predict_csat_for_call(call_id: int) -> None:
    """
    Fire-and-forget. Predicts CSAT for a single call and persists the result.
    Called from analyze_call_task after the main pipeline completes.

    After saving the prediction, checks whether enough new calls have arrived
    since the last model training run and fires retrain_csat_model.delay() if so.
    This makes the model self-improving automatically — no manual scripts needed.

    Failure here must never affect the call's completed status.
    """
    from .models import Call
    from AI_modules.csat_predictor.predictor import predict, should_auto_retrain

    try:
        call = Call.objects.get(id=call_id)
        score, label = predict(call)
        call.predicted_csat       = score
        call.predicted_csat_label = label
        call.save(update_fields=["predicted_csat", "predicted_csat_label"])
        logger.info(f"[CSAT] call_id={call_id} predicted={score} ({label})")
    except Exception:
        logger.exception(f"[CSAT] predict_csat_for_call failed for call_id={call_id}")
        return  # Don't proceed to retrain check on failure

    # Auto-retrain: count eligible calls and trigger retraining if the threshold is met.
    # This fires at most once every RETRAIN_EVERY_N_CALLS new calls, after MIN_TRAINING_SAMPLES
    # are reached — so the model silently improves in the background as calls accumulate.
    try:
        from .models import Call as _Call
        total = _Call.objects.filter(
            status="completed",
            emotional_summary__isnull=False,
        ).count()
        if should_auto_retrain(total):
            logger.info(f"[CSAT] Auto-retrain triggered — {total} eligible calls in history")
            retrain_csat_model.delay()
    except Exception:
        logger.exception("[CSAT] Auto-retrain check failed (non-fatal)")


@shared_task
def retrain_csat_model() -> dict:
    """
    Retrains the CSAT model on all completed calls with available labels.

    Triggered two ways:
      1. Automatically by predict_csat_for_call() when RETRAIN_EVERY_N_CALLS new
         calls have accumulated since the last training run.
      2. On a fixed schedule (every Sunday 04:00 UTC) via Celery Beat as a backstop.

    Skips silently if fewer than MIN_TRAINING_SAMPLES labeled calls exist yet.
    """
    from .models import Call
    from AI_modules.csat_predictor.predictor import train, MIN_TRAINING_SAMPLES

    calls = list(
        Call.objects.filter(
            status="completed",
            emotional_summary__isnull=False,
        ).only(
            "emotional_summary", "behavioral_analysis",
            "behavioral_summary", "transcript_summary",
        )
    )
    total = len(calls)

    if total < MIN_TRAINING_SAMPLES:
        logger.info(
            f"[CSAT] Only {total} labeled calls — skipping retrain "
            f"(need {MIN_TRAINING_SAMPLES})."
        )
        return {"skipped": True, "n_calls": total}

    try:
        metadata = train(calls, total_call_count=total)
        return metadata
    except ValueError as exc:
        logger.info(f"[CSAT] Retrain skipped: {exc}")
        return {"skipped": True, "reason": str(exc)}
