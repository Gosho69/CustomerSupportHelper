import concurrent.futures

from Whisperer.whisperer import transcribe_audio
from Emotion_analyzation.emotion_analyzer import emotion_analyze_call
from Emotion_analyzation.summary import summarize_emotion_call
from behaviour_analyzer.behavioral_analyzer import behavioral_analyze_call
from behaviour_analyzer.summary import summarize_behavioral_call
from summarization.local_summary import analyze_call as local_summary_analyze
from summarization.gpt4_summary import analyze_call as gpt4_summary_analyze
from coaching_tips.coaching_tips import generate
from topic_analyzer.topic_analyzer import analyze_topics


# Maps local model customer_tone labels to standardized emotion summary fields
_TONE_TO_SATISFACTION = {
    "positive": "very_satisfied",
    "satisfied": "satisfied",
    "neutral": "neutral",
    "frustrated": "dissatisfied",
    "negative": "dissatisfied",
    "angry": "very_dissatisfied",
}

_TONE_TO_CALL_TONE = {
    "positive": "positive",
    "satisfied": "positive",
    "neutral": "neutral",
    "frustrated": "negative",
    "negative": "negative",
    "angry": "negative",
}

_TONE_TO_RESOLUTION = {
    "positive": "resolved",
    "satisfied": "resolved",
    "neutral": "pending",
    "frustrated": "unresolved",
    "negative": "unresolved",
    "angry": "unresolved",
}

_AGENT_TONE_TO_EMPATHY = {
    "empathetic": 0.85,
    "apologetic": 0.75,
    "helpful": 0.6,
    "professional": 0.5,
    "positive": 0.55,
    "neutral": 0.3,
    "dismissive": 0.1,
}


def _apply_local_model_tone_signals(emotion_summary: dict, call_summary: dict) -> dict:
    """
    When running the local model, use the model's customer_tone and agent_tone
    outputs to override the high-level emotion summary fields that the rule-based
    system tends to get wrong.
    """
    customer_tone = (call_summary.get("customer_tone") or "").lower().strip()
    agent_tone = (call_summary.get("agent_tone") or "").lower().strip()

    if customer_tone and customer_tone in _TONE_TO_SATISFACTION:
        emotion_summary["customer_satisfaction"] = _TONE_TO_SATISFACTION[customer_tone]
        emotion_summary["call_tone"] = _TONE_TO_CALL_TONE[customer_tone]
        # Only override resolution if the rule-based system returned "pending"
        # (so explicit unresolved/resolved from context is preserved)
        if emotion_summary.get("resolution_status") == "pending":
            emotion_summary["resolution_status"] = _TONE_TO_RESOLUTION[customer_tone]

    if agent_tone and agent_tone in _AGENT_TONE_TO_EMPATHY:
        # Blend with rule-based score — take the higher of the two to avoid
        # penalising agents when keyword detection misses empathy phrases
        rule_based_score = emotion_summary.get("agent_empathy_score", 0.0)
        model_score = _AGENT_TONE_TO_EMPATHY[agent_tone]
        emotion_summary["agent_empathy_score"] = round(max(rule_based_score, model_score), 3)

    return emotion_summary


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
    # Emotion chain, behavioral chain, summarization, and topic extraction are
    # independent of each other — especially valuable because the GPT-4 summary
    # is a network call that would otherwise block the CPU-bound analyses.
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        emotion_future = executor.submit(_run_emotion)
        behavioral_future = executor.submit(_run_behavioral)
        summary_future = executor.submit(_run_summary)
        topics_future = executor.submit(analyze_topics, transcript)

        emotion_results, emotion_summary = emotion_future.result()
        behavioral_results, behavioral_summary = behavioral_future.result()
        call_summary = summary_future.result()
        topic_analysis = topics_future.result()

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
    elif summarization_model.lower() != "gpt4":
        emotion_summary = _apply_local_model_tone_signals(emotion_summary, call_summary)

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
    else:
        call_summary = local_summary_analyze(
            transcript=transcript,
            checkpoint_path=local_model_path
        )
        emotion_summary = _apply_local_model_tone_signals(emotion_summary, call_summary)

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
