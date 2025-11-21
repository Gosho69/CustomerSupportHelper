import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Whisperer.whisperer import transcribe_audio
from Emotion_analyzation.emotion_analyzer import emotion_analyze_call
from Emotion_analyzation.summary import summarize_emotion_call
from behaviour_analyzer.behavioral_analyzer import behavioral_analyze_call
from behaviour_analyzer.summary import summarize_behavioral_call
from summarization.local_summary import analyze_call as local_summary_analyze
from summarization.gpt4_summary import analyze_call as gpt4_summary_analyze
from coaching_tips.coaching_tips import generate


def analyze_call(
    audio_path,
    summarization_model="gpt4",
    whisper_model_size="base",
    device="cpu",
    compute_type="int8",
    local_model_path="../model/final",
    gpt4_model="gpt-4o-mini",
    gpt4_temperature=0.3,
    gpt4_max_tokens=800
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
    else:
        call_summary = local_summary_analyze(
            transcript=transcript,
            checkpoint_path=local_model_path
        )
    
    coaching_tips = generate(
        transcript=transcript,
        summary_result=call_summary,
        emotion_result=emotion_results,
        behavioral_result=behavioral_results
    )
    
    return {
        "transcript": transcript,
        "emotion_analysis": emotion_results,
        "emotion_summary": emotion_summary,
        "behavioral_analysis": behavioral_results,
        "behavioral_summary": behavioral_summary,
        "call_summary": call_summary,
        "coaching_tips": coaching_tips
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
    gpt4_temperature=0.3,
    gpt4_max_tokens=800
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
    else:
        call_summary = local_summary_analyze(
            transcript=transcript,
            checkpoint_path=local_model_path
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
    gpt4_temperature=0.3,
    gpt4_max_tokens=800
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
