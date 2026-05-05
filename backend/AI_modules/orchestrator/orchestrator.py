import concurrent.futures
import logging

import phrase_loader as _pl
from Whisperer.whisperer import transcribe_audio

logger = logging.getLogger(__name__)

def _update_behavioral_score(behavioral_results: dict, behavioral_summary: dict | None, new_score: float):
    """Apply a new score to both behavioral_results and behavioral_summary."""
    behavioral_results["behavioral_score"] = new_score
    if behavioral_summary:
        behavioral_summary["overall_score"] = new_score
        if new_score >= 90:
            behavioral_summary["overall_assessment"] = "excellent"
        elif new_score >= 75:
            behavioral_summary["overall_assessment"] = "good"
        elif new_score >= 60:
            behavioral_summary["overall_assessment"] = "acceptable"
        elif new_score >= 40:
            behavioral_summary["overall_assessment"] = "needs_improvement"
        else:
            behavioral_summary["overall_assessment"] = "poor"

from Emotion_analyzation.emotion_analyzer import emotion_analyze_call
from Emotion_analyzation.summary import summarize_emotion_call
from behaviour_analyzer.behavioral_analyzer import behavioral_analyze_call
from behaviour_analyzer.summary import summarize_behavioral_call
from summarization.local_summary import analyze_call as local_summary_analyze
from summarization.gpt4_summary import analyze_call as gpt4_summary_analyze
from coaching_tips.coaching_tips import generate
from topic_analyzer.topic_analyzer import analyze_topics
from vocal_analysis.vocal_analyzer import analyze_vocal_stress, _empty_result as _vocal_empty


# Maps local model customer_tone labels to standardized emotion summary fields.
# Defaults kept here as fallback for phrase_loader.get().

_TONE_TO_SATISFACTION_DEFAULT = {
    "positive": "very_satisfied",
    "satisfied": "satisfied",
    "neutral": "neutral",
    "frustrated": "dissatisfied",
    "negative": "dissatisfied",
    "angry": "very_dissatisfied",
}

_TONE_TO_CALL_TONE_DEFAULT = {
    "positive": "positive",
    "satisfied": "positive",
    "neutral": "neutral",
    "frustrated": "negative",
    "negative": "negative",
    "angry": "negative",
}

_TONE_TO_RESOLUTION_DEFAULT = {
    "positive": "resolved",
    "satisfied": "resolved",
    "neutral": "pending",
    "frustrated": "unresolved",
    "negative": "unresolved",
    "angry": "unresolved",
}

_AGENT_TONE_TO_EMPATHY_DEFAULT = {
    "empathetic": 0.85,
    "apologetic": 0.75,
    "helpful": 0.6,
    "professional": 0.5,
    "positive": 0.55,
    "neutral": 0.3,
    "dismissive": 0.1,
}


def _tone_to_satisfaction():
    return _pl.get("tone_to_satisfaction", _TONE_TO_SATISFACTION_DEFAULT)


def _tone_to_call_tone():
    return _pl.get("tone_to_call_tone", _TONE_TO_CALL_TONE_DEFAULT)


def _tone_to_resolution():
    return _pl.get("tone_to_resolution", _TONE_TO_RESOLUTION_DEFAULT)


def _agent_tone_to_empathy():
    return _pl.get("agent_tone_to_empathy", _AGENT_TONE_TO_EMPATHY_DEFAULT)


def _apply_local_model_tone_signals(emotion_summary: dict, call_summary: dict) -> dict:
    """
    When running the local model, use the model's customer_tone and agent_tone
    outputs to refine the high-level emotion summary fields.

    Key constraint: the per-turn emotion analyzer uses distilroberta on each
    utterance and produces a trajectory-based resolution/satisfaction assessment
    that is more granular and reliable than the FLAN-T5 model's single overall
    tone label.  We therefore NEVER downgrade a call that the emotion analyzer
    has already confirmed as resolved + satisfied — the local model's negative
    tone label on such calls is almost always a misclassification.
    """
    customer_tone = (call_summary.get("customer_tone") or "").lower().strip()
    agent_tone    = (call_summary.get("agent_tone")    or "").lower().strip()

    tone_sat = _tone_to_satisfaction()
    tone_ct  = _tone_to_call_tone()
    tone_res = _tone_to_resolution()
    tone_emp = _agent_tone_to_empathy()

    current_resolution   = emotion_summary.get("resolution_status",    "pending")
    current_satisfaction = emotion_summary.get("customer_satisfaction", "neutral")
    strongly_negative    = customer_tone in ("frustrated", "negative", "angry")

    # "already confirmed positive" — emotion analyser found the call resolved
    # AND the customer was satisfied.  Protecting both fields together avoids
    # a partial override that would still trigger the satisfaction-only penalty.
    already_positive = (
        current_resolution   == "resolved" and
        current_satisfaction in ("satisfied", "very_satisfied")
    )

    if customer_tone and customer_tone in tone_sat:
        # Only apply a negative-tone override when the emotion analyser has NOT
        # already confirmed a positive outcome.  Positive overrides (e.g. model
        # says "satisfied") are always safe to apply.
        if not (strongly_negative and already_positive):
            emotion_summary["customer_satisfaction"] = tone_sat[customer_tone]
            emotion_summary["call_tone"] = tone_ct[customer_tone]
        else:
            logger.info(
                "_apply_local_model_tone_signals: skipping negative satisfaction "
                "override — emotion analyser already confirmed resolved+satisfied "
                "(customer_tone=%s)", customer_tone
            )

    if agent_tone and agent_tone in tone_emp:
        # Blend with rule-based score — take the higher of the two to avoid
        # penalising agents when keyword detection misses empathy phrases.
        rule_based_score = emotion_summary.get("agent_empathy_score", 0.0)
        model_score      = tone_emp[agent_tone]
        emotion_summary["agent_empathy_score"] = round(max(rule_based_score, model_score), 3)

    # Resolution override: only promote/demote "pending" status.
    # Never downgrade "resolved" — the emotion analyser's trajectory-based
    # "resolved" assessment is authoritative and the local model's overall tone
    # label is an unreliable signal for this specific field.
    if customer_tone and customer_tone in tone_res:
        if current_resolution == "pending":
            emotion_summary["resolution_status"] = tone_res[customer_tone]
        elif current_resolution == "resolved" and strongly_negative:
            logger.info(
                "_apply_local_model_tone_signals: preserving 'resolved' status — "
                "not overriding with model's negative tone (customer_tone=%s)",
                customer_tone
            )

    return emotion_summary


def _compute_local_model_score(behavioral_score: float, ratings: dict) -> float:
    """
    Compute the final score for local-model calls as a weighted blend of
    behavioral quality and model-rated interaction quality.

    The five model ratings (1-5 scale) are weighted by importance and mapped
    to a 0-100 quality scale, then blended with the behavioral score.

    Weights:
      helpfulness 40% — primary outcome signal: did the agent solve the problem?
      overall     25% — model's holistic judgment
      clarity     15% — communication effectiveness
      respect     10% — agent tone toward customer
      adherence   10% — procedure compliance

    Blend is outcome-scaled: when helpfulness is very low (agent failed to help),
    a good behavioral score (clear speech, pacing) must not mask the outcome failure.
      helpfulness ≤ 1  →  15 % behavioral + 85 % model quality
      helpfulness ≤ 2  →  25 % behavioral + 75 % model quality
      helpfulness ≥ 3  →  40 % behavioral + 60 % model quality  (default)
    """
    h = float(ratings.get("helpfulness", 3))
    o = float(ratings.get("overall",     3))
    c = float(ratings.get("clarity",     3))
    r = float(ratings.get("respect",     3))
    a = float(ratings.get("adherence",   3))

    weighted = 0.40*h + 0.25*o + 0.15*c + 0.10*r + 0.10*a
    model_quality = (weighted - 1.0) / 4.0 * 100.0          # 1-5 → 0-100

    if h <= 1:
        beh_w, mdl_w = 0.15, 0.85
    elif h <= 2:
        beh_w, mdl_w = 0.25, 0.75
    else:
        beh_w, mdl_w = 0.40, 0.60

    blended = beh_w * behavioral_score + mdl_w * model_quality
    return max(0.0, min(100.0, round(blended, 1)))


def analyze_call(
    audio_path,
    summarization_model="gpt4",
    whisper_model_size="base",
    device="cpu",
    compute_type="int8",
    local_model_path="../model/final",
    gpt4_model="gpt-4o-mini",
    gpt4_temperature=0.2,
    gpt4_max_tokens=1500,
):
    """
    Complete call analysis pipeline.
    
    Args:
        audio_path: Path to audio file
        summarization_model: "gpt4" or "local" for transcript summarization
        whisper_model_size: Whisper model size (tiny, base, small, medium, large-v2)
        device: Device to use (cpu, cuda)
        compute_type: Compute type (int8, float16, float32)
        local_model_path: Path to local fine-tuned model (if using local)
        gpt4_model: GPT-4 model name (if using gpt4)
        gpt4_temperature: Temperature for GPT-4 (if using gpt4)
        gpt4_max_tokens: Max tokens for GPT-4 (if using gpt4)
    
    Returns:
        Dict with all analysis results:
        - transcript: WhisperX output
        - emotion_analysis: Emotion analysis results
        - emotion_summary: Emotion analysis summary
        - behavioral_analysis: Behavioral metrics
        - behavioral_summary: Behavioral summary
        - call_summary: Call summary and ratings
        - coaching_tips: Coaching tips (if issues found)
    """
    transcript = transcribe_audio(
        audio_path=audio_path,
        model_size=whisper_model_size,
        device=device,
        compute_type=compute_type
    )

    # Run all post-transcription analyses in parallel.
    # Emotion chain, behavioral chain, summarization, topic extraction, and
    # vocal stress analysis are all independent of each other.
    def _run_emotion():
        results = emotion_analyze_call(transcript)
        summary = summarize_emotion_call(results)
        return results, summary

    def _run_behavioral():
        results = behavioral_analyze_call(transcript)
        summary = summarize_behavioral_call(results)
        return results, summary

    def _run_summary():
        if summarization_model.lower() == "gpt4":
            return gpt4_summary_analyze(
                transcript=transcript,
                model=gpt4_model,
                temperature=gpt4_temperature,
                max_tokens=gpt4_max_tokens,
            )
        return local_summary_analyze(transcript=transcript, checkpoint_path=local_model_path)

    def _run_vocal_analysis():
        try:
            return analyze_vocal_stress(audio_path, transcript)
        except Exception as exc:
            import traceback
            import sys
            print(
                f"[Orchestrator] vocal_analysis FAILED — returning empty result.\n"
                f"  {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )
            return _vocal_empty(str(exc))

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        emotion_future = executor.submit(_run_emotion)
        behavioral_future = executor.submit(_run_behavioral)
        summary_future = executor.submit(_run_summary)
        topics_future = executor.submit(analyze_topics, transcript)
        vocal_future = executor.submit(_run_vocal_analysis)

        emotion_results, emotion_summary = emotion_future.result()
        behavioral_results, behavioral_summary = behavioral_future.result()
        call_summary = summary_future.result()
        topic_analysis = topics_future.result()
        vocal_analysis = vocal_future.result()

    # Enrich emotion_summary with GPT-4 emotional assessment (sequential — needs both results)
    if summarization_model.lower() == "gpt4" and "emotional_assessment" in call_summary:
        ea = call_summary["emotional_assessment"]
        emotion_summary["summary"] = ea.get("emotional_narrative", emotion_summary["summary"])
        emotion_summary["customer_satisfaction"] = ea.get("customer_satisfaction", emotion_summary["customer_satisfaction"])
        emotion_summary["resolution_status"] = ea.get("resolution_status", emotion_summary["resolution_status"])
        emotion_summary["call_tone"] = ea.get("call_tone", emotion_summary["call_tone"])
        emotion_summary["agent_empathy_score"] = ea.get("agent_empathy_score", emotion_summary["agent_empathy_score"])
        emotion_summary["customer_frustration_level"] = ea.get("customer_frustration_level", emotion_summary["customer_frustration_level"])
        journey = emotion_summary.get("emotional_journey", {})
        journey["trajectory"] = ea.get("emotional_trajectory", journey.get("trajectory"))
        journey["start_emotion"] = ea.get("start_emotion", journey.get("start_emotion"))
        journey["end_emotion"] = ea.get("end_emotion", journey.get("end_emotion"))
        journey["description"] = ea.get("trajectory_description", journey.get("description"))
        emotion_summary["emotional_journey"] = journey

        # Resolution penalty: if the customer's request was not resolved AND they were
        # dissatisfied, deduct 25 points from the behavioral score.  The behavioral score
        # is based purely on communication metrics (WPM, pacing, interruptions) and has
        # no awareness of call outcome — this correction makes the overall score reflect
        # what actually happened.
        resolution = ea.get("resolution_status", "unknown")
        satisfaction = ea.get("customer_satisfaction", "unknown")
        raw_score = behavioral_results.get("behavioral_score", 0)
        logger.info(
            "[Orchestrator] GPT-4 path: behavioral_score_raw=%.1f, "
            "resolution=%s, satisfaction=%s",
            raw_score, resolution, satisfaction
        )
        resolution_penalty_applied = False
        if resolution == "unresolved" and satisfaction in ("dissatisfied", "very_dissatisfied"):
            penalty = 25
            old_score = raw_score
            new_score = max(0.0, round(old_score - penalty, 1))
            _update_behavioral_score(behavioral_results, behavioral_summary, new_score)
            resolution_penalty_applied = True
            logger.warning(
                "[Orchestrator] Resolution penalty applied: %.1f → %.1f "
                "(resolution=%s, satisfaction=%s)",
                old_score, new_score, resolution, satisfaction
            )
        else:
            logger.info(
                "[Orchestrator] No resolution penalty (resolution=%s, satisfaction=%s)",
                resolution, satisfaction
            )

        # Satisfaction-only penalty: fires when the customer was dissatisfied
        # but the resolution penalty above didn't (e.g. GPT-4 marked "resolved"
        # despite evidence of a bad outcome).  Deducts 15 points — less than the
        # full resolution penalty to avoid double-counting when both fire.
        if not resolution_penalty_applied and satisfaction in ("dissatisfied", "very_dissatisfied"):
            sat_penalty = 15
            old_score = behavioral_results.get("behavioral_score", 0)
            new_score = max(0.0, round(old_score - sat_penalty, 1))
            _update_behavioral_score(behavioral_results, behavioral_summary, new_score)
            logger.warning(
                "[Orchestrator] Satisfaction-only penalty applied: %.1f → %.1f "
                "(satisfaction=%s, resolution=%s)",
                old_score, new_score, satisfaction, resolution
            )

    elif summarization_model.lower() != "gpt4":
        emotion_summary = _apply_local_model_tone_signals(emotion_summary, call_summary)

        # Local model path: blend behavioral quality with model-rated quality.
        # No stacking penalty chain — each penalty was an independent deduction
        # that could compound into catastrophically wrong scores.  The model
        # ratings already encode call quality; scaling them directly and blending
        # with behavioral metrics is transparent and bounded.
        ratings = (call_summary.get("detailed_ratings") or {})
        raw_beh = behavioral_results.get("behavioral_score", 50.0)
        blended = _compute_local_model_score(raw_beh, ratings)
        _update_behavioral_score(behavioral_results, behavioral_summary, blended)
        logger.info(
            "[Orchestrator] Local model blended score: behavioral=%.1f, "
            "helpfulness=%s, overall=%s → final=%.1f",
            raw_beh,
            ratings.get("helpfulness", "?"),
            ratings.get("overall", "?"),
            blended,
        )

    # Low-helpfulness / low-overall penalties apply to the GPT-4 path only.
    # For the local model path the ratings already feed into _compute_local_model_score;
    # applying them again here would double-count and produce inflated deductions.
    if summarization_model.lower() == "gpt4":
        helpfulness = (call_summary.get("detailed_ratings") or {}).get("helpfulness")
        if helpfulness is not None and helpfulness <= 2:
            help_penalty = (3 - helpfulness) * 15
            current_score = behavioral_results.get("behavioral_score", 0)
            new_score = max(0.0, round(current_score - help_penalty, 1))
            _update_behavioral_score(behavioral_results, behavioral_summary, new_score)
            logger.warning(
                "[Orchestrator] Low helpfulness penalty: helpfulness=%d → -%.0f: %.1f → %.1f",
                helpfulness, help_penalty, current_score, new_score
            )

        overall_rating = (call_summary.get("detailed_ratings") or {}).get("overall")
        if overall_rating is not None and overall_rating <= 2:
            overall_penalty = (3 - overall_rating) * 15
            current_score = behavioral_results.get("behavioral_score", 0)
            new_score = max(0.0, round(current_score - overall_penalty, 1))
            _update_behavioral_score(behavioral_results, behavioral_summary, new_score)
            logger.warning(
                "[Orchestrator] Low overall rating penalty: overall=%d → -%.0f: %.1f → %.1f",
                overall_rating, overall_penalty, current_score, new_score
            )

    coaching_tips = generate(
        transcript=transcript,
        summary_result=call_summary,
        emotion_result=emotion_results,
        behavioral_result=behavioral_results,
    )

    return {
        "transcript": transcript,
        "emotion_analysis": emotion_results,
        "emotion_summary": emotion_summary,
        "behavioral_analysis": behavioral_results,
        "behavioral_summary": behavioral_summary,
        "call_summary": call_summary,
        "coaching_tips": coaching_tips,
        "topic_analysis": topic_analysis,
        "vocal_analysis": vocal_analysis,
    }


def transcribe_only(
    audio_path,
    model_size="base",
    device="cpu",
    compute_type="int8"
):
    """
    Transcribe audio without analysis.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        device: Device to use
        compute_type: Compute type
    
    Returns:
        WhisperX transcript dict
    """
    return transcribe_audio(
        audio_path=audio_path,
        model_size=model_size,
        device=device,
        compute_type=compute_type
    )


def analyze_transcript(
    transcript,
    summarization_model="gpt4",
    local_model_path="../model/final",
    gpt4_model="gpt-4o-mini",
    gpt4_temperature=0.2,
    gpt4_max_tokens=1500
):
    """
    Analyze existing transcript without transcription.
    
    Args:
        transcript: WhisperX transcript dict or JSON string
        summarization_model: "gpt4" or "local"
        local_model_path: Path to local model (if using local)
        gpt4_model: GPT-4 model name (if using gpt4)
        gpt4_temperature: Temperature for GPT-4
        gpt4_max_tokens: Max tokens for GPT-4
    
    Returns:
        Dict with analysis results (same as analyze_call but without transcript key)
    """
    emotion_results = emotion_analyze_call(transcript)

    emotion_summary = summarize_emotion_call(emotion_results)

    behavioral_results = behavioral_analyze_call(transcript)

    behavioral_summary = summarize_behavioral_call(behavioral_results)

    if summarization_model.lower() == "gpt4":
        call_summary = gpt4_summary_analyze(
            transcript=transcript,
            model=gpt4_model,
            temperature=gpt4_temperature,
            max_tokens=gpt4_max_tokens
        )
        if "emotional_assessment" in call_summary:
            ea = call_summary["emotional_assessment"]
            emotion_summary["summary"] = ea.get("emotional_narrative", emotion_summary["summary"])
            emotion_summary["customer_satisfaction"] = ea.get("customer_satisfaction", emotion_summary["customer_satisfaction"])
            emotion_summary["resolution_status"] = ea.get("resolution_status", emotion_summary["resolution_status"])
            emotion_summary["call_tone"] = ea.get("call_tone", emotion_summary["call_tone"])
            emotion_summary["agent_empathy_score"] = ea.get("agent_empathy_score", emotion_summary["agent_empathy_score"])
            emotion_summary["customer_frustration_level"] = ea.get("customer_frustration_level", emotion_summary["customer_frustration_level"])
            journey = emotion_summary.get("emotional_journey", {})
            journey["trajectory"] = ea.get("emotional_trajectory", journey.get("trajectory"))
            journey["start_emotion"] = ea.get("start_emotion", journey.get("start_emotion"))
            journey["end_emotion"] = ea.get("end_emotion", journey.get("end_emotion"))
            journey["description"] = ea.get("trajectory_description", journey.get("description"))
            emotion_summary["emotional_journey"] = journey

            resolution = ea.get("resolution_status", "unknown")
            satisfaction = ea.get("customer_satisfaction", "unknown")
            raw_score = behavioral_results.get("behavioral_score", 0)
            logger.info(
                "[Orchestrator] GPT-4 path (transcript): behavioral_score_raw=%.1f, "
                "resolution=%s, satisfaction=%s",
                raw_score, resolution, satisfaction
            )
            resolution_penalty_applied = False
            if resolution == "unresolved" and satisfaction in ("dissatisfied", "very_dissatisfied"):
                penalty = 25
                old_score = raw_score
                new_score = max(0.0, round(old_score - penalty, 1))
                _update_behavioral_score(behavioral_results, behavioral_summary, new_score)
                resolution_penalty_applied = True
                logger.warning(
                    "[Orchestrator] Resolution penalty applied: %.1f → %.1f",
                    old_score, new_score
                )
            else:
                logger.info(
                    "[Orchestrator] No resolution penalty (resolution=%s, satisfaction=%s)",
                    resolution, satisfaction
                )

            if not resolution_penalty_applied and satisfaction in ("dissatisfied", "very_dissatisfied"):
                sat_penalty = 15
                old_score = behavioral_results.get("behavioral_score", 0)
                new_score = max(0.0, round(old_score - sat_penalty, 1))
                _update_behavioral_score(behavioral_results, behavioral_summary, new_score)
                logger.warning(
                    "[Orchestrator] Satisfaction-only penalty applied: %.1f → %.1f "
                    "(satisfaction=%s, resolution=%s)",
                    old_score, new_score, satisfaction, resolution
                )

    else:
        call_summary = local_summary_analyze(
            transcript=transcript,
            checkpoint_path=local_model_path
        )
        emotion_summary = _apply_local_model_tone_signals(emotion_summary, call_summary)

        ratings = (call_summary.get("detailed_ratings") or {})
        raw_beh = behavioral_results.get("behavioral_score", 50.0)
        blended = _compute_local_model_score(raw_beh, ratings)
        _update_behavioral_score(behavioral_results, behavioral_summary, blended)
        logger.info(
            "[Orchestrator] Local model blended score: behavioral=%.1f, "
            "helpfulness=%s, overall=%s → final=%.1f",
            raw_beh,
            ratings.get("helpfulness", "?"),
            ratings.get("overall", "?"),
            blended,
        )

    if summarization_model.lower() == "gpt4":
        helpfulness = (call_summary.get("detailed_ratings") or {}).get("helpfulness")
        if helpfulness is not None and helpfulness <= 2:
            help_penalty = (3 - helpfulness) * 15
            current_score = behavioral_results.get("behavioral_score", 0)
            new_score = max(0.0, round(current_score - help_penalty, 1))
            _update_behavioral_score(behavioral_results, behavioral_summary, new_score)
            logger.warning(
                "[Orchestrator] Low helpfulness penalty: helpfulness=%d → -%.0f: %.1f → %.1f",
                helpfulness, help_penalty, current_score, new_score
            )

        overall_rating = (call_summary.get("detailed_ratings") or {}).get("overall")
        if overall_rating is not None and overall_rating <= 2:
            overall_penalty = (3 - overall_rating) * 15
            current_score = behavioral_results.get("behavioral_score", 0)
            new_score = max(0.0, round(current_score - overall_penalty, 1))
            _update_behavioral_score(behavioral_results, behavioral_summary, new_score)
            logger.warning(
                "[Orchestrator] Low overall rating penalty: overall=%d → -%.0f: %.1f → %.1f",
                overall_rating, overall_penalty, current_score, new_score
            )

    coaching_tips = generate(
        transcript=transcript,
        summary_result=call_summary,
        emotion_result=emotion_results,
        behavioral_result=behavioral_results
    )

    return {
        "emotion_analysis": emotion_results,
        "emotion_summary": emotion_summary,
        "behavioral_analysis": behavioral_results,
        "behavioral_summary": behavioral_summary,
        "call_summary": call_summary,
        "coaching_tips": coaching_tips
    }


def get_emotion_analysis(transcript):
    """
    Get only emotion analysis.
    
    Args:
        transcript: WhisperX transcript dict or JSON string
    
    Returns:
        Dict with emotion_analysis and emotion_summary
    """
    emotion_results = emotion_analyze_call(transcript)
    emotion_summary = summarize_emotion_call(emotion_results)
    
    return {
        "emotion_analysis": emotion_results,
        "emotion_summary": emotion_summary
    }


def get_behavioral_analysis(transcript):
    """
    Get only behavioral analysis.
    
    Args:
        transcript: WhisperX transcript dict or JSON string
    
    Returns:
        Dict with behavioral_analysis and behavioral_summary
    """
    behavioral_results = behavioral_analyze_call(transcript)
    behavioral_summary = summarize_behavioral_call(behavioral_results)
    
    return {
        "behavioral_analysis": behavioral_results,
        "behavioral_summary": behavioral_summary
    }


def get_call_summary(
    transcript,
    model="gpt4",
    local_model_path="../model/final",
    gpt4_model="gpt-4o-mini",
    gpt4_temperature=0.2,
    gpt4_max_tokens=1500
):
    """
    Get only call summary and ratings.
    
    Args:
        transcript: WhisperX transcript dict or JSON string
        model: "gpt4" or "local"
        local_model_path: Path to local model
        gpt4_model: GPT-4 model name
        gpt4_temperature: Temperature for GPT-4
        gpt4_max_tokens: Max tokens for GPT-4
    
    Returns:
        Call summary dict
    """
    if model.lower() == "gpt4":
        return gpt4_summary_analyze(
            transcript=transcript,
            model=gpt4_model,
            temperature=gpt4_temperature,
            max_tokens=gpt4_max_tokens
        )
    else:
        return local_summary_analyze(
            transcript=transcript,
            checkpoint_path=local_model_path
        )


def get_coaching_tips(
    transcript,
    summary_result=None,
    emotion_result=None,
    behavioral_result=None
):
    """
    Get coaching tips based on analysis results.
    
    Args:
        transcript: WhisperX transcript dict or JSON string
        summary_result: Optional pre-computed summary
        emotion_result: Optional pre-computed emotion analysis
        behavioral_result: Optional pre-computed behavioral analysis
    
    Returns:
        Coaching tips dict
    """
    return generate(
        transcript=transcript,
        summary_result=summary_result,
        emotion_result=emotion_result,
        behavioral_result=behavioral_result
    )
