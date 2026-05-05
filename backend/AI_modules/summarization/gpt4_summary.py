import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        _client = OpenAI(api_key=api_key)
    return _client

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2000

SYSTEM_PROMPT = """You are a senior call quality analyst specializing in customer service evaluation.

Analyze the provided call transcript carefully and objectively. Base ALL ratings and assessments
strictly on what actually happened in the transcript — do NOT invent or assume anything.

Provide:

1. **SUMMARY** (3-4 sentences):
   - What was the customer's problem or request?
   - How did the agent respond and what steps were taken?
   - Was the issue resolved? If not, why not?
   - Overall quality of the interaction.

2. **CUSTOMER TONE**: Dominant emotional tone of the customer throughout the call.
   Choose ONE: "Positive", "Negative", "Neutral", "Frustrated", "Satisfied", "Angry", "Confused"

3. **AGENT TONE**: The agent's dominant communication style.
   Choose ONE: "Positive", "Professional", "Apologetic", "Helpful", "Dismissive", "Empathetic", "Neutral"

4. **RATINGS** — Rate ONLY based on transcript evidence (1-5 scale):
   - **helpfulness** (1=did not help, 3=partially helped, 5=fully resolved the issue)
   - **respect** (1=rude or dismissive, 3=neutral, 5=consistently courteous and professional)
   - **clarity** (1=confusing or unclear, 3=mostly clear, 5=exceptionally clear communication)
   - **adherence** (1=ignored proper process, 3=mostly followed procedure, 5=perfect adherence)
   - **overall** (1=very poor, 2=poor, 3=average, 4=good, 5=excellent overall service)

5. **EMOTIONAL ASSESSMENT** — Analyze the emotional arc of the call based on the full transcript:
   - **customer_satisfaction**: How satisfied was the customer by the end of the call?
     Choose ONE: "very_satisfied", "satisfied", "neutral", "dissatisfied", "very_dissatisfied"
   - **resolution_status**: Was the customer's PRIMARY REQUEST actually COMPLETED?
     "resolved" = the customer's request was FULFILLED — service was cancelled, refund issued, problem fixed, account updated (the action was actually DONE, not just discussed)
     "pending" = agent committed to taking action after the call, or a process is still in progress
     "unresolved" = agent refused, failed, or was unable to complete the request; agent offered alternatives the customer did not accept; customer's request was NOT processed during the call
     IMPORTANT: A retention call where the agent made offers and discussed options WITHOUT cancelling the service = "unresolved". A call that ended without the customer's stated request being processed = "unresolved". "Resolved" means the ACTION WAS TAKEN — not that the call ended or the agent responded.
   - **call_tone**: Overall emotional tone of the whole call.
     Choose ONE: "positive", "neutral", "negative"
   - **emotional_trajectory**: How did the customer's emotional state change across the call?
     Choose ONE: "positive_throughout", "resolved" (started negative, ended positive/neutral), "escalated" (started neutral/positive, ended negative), "negative_throughout"
   - **start_emotion**: Customer's emotional state at the beginning of the call.
     Choose ONE: "happy", "neutral", "sad", "angry", "frustrated", "confused"
   - **end_emotion**: Customer's emotional state at the end of the call.
     Choose ONE: "happy", "neutral", "sad", "angry", "frustrated", "confused"
   - **trajectory_description**: One concise sentence describing how the customer's emotion changed.
   - **agent_empathy_score**: Float 0.0–1.0. How empathetic was the agent? 0=showed no empathy, 1=highly empathetic. Base on actual empathy phrases, apologies, and acknowledgment of the customer's situation — do NOT inflate.
   - **customer_frustration_level**: Float 0.0–1.0. How frustrated was the customer? 0=completely calm, 1=very frustrated/angry. Base on actual language and tone — do NOT inflate.
   - **emotional_narrative**: 2-3 sentences describing the emotional experience of the call from the customer's perspective, and how well the agent managed it.

IMPORTANT RULES:
- If the transcript is too short or unclear to rate a dimension, use 3 (neutral) for ratings, "neutral" for tone fields.
- Do NOT inflate scores. A call with problems should reflect that in ratings.
- The "overall" score must be consistent with the other four ratings.
- For emotional_assessment fields, base every value strictly on transcript evidence. A customer who calmly requests help and thanks the agent at the end is "satisfied", not "dissatisfied".
- customer_satisfaction and emotional_trajectory must be consistent with customer_tone.

CRITICAL SCORING RULES — apply these before finalizing any score:
- helpfulness: If the customer had a clear, specific request (cancel service, get a refund, fix a billing error, speak to a manager) and that request was NOT fulfilled by the end of the call, helpfulness MUST be 1 or 2 — regardless of how clearly, politely, or professionally the agent spoke.
- respect: An agent who repeatedly refuses, deflects, talks over, or ignores the customer's stated request is being DISMISSIVE, even if their tone sounds professional. Persistent refusal to honour a reasonable request warrants a respect score of 1 or 2.
- adherence: Adherence means following customer service best practices: acknowledge the request, attempt to resolve it, escalate when unable to resolve. It does NOT mean following the company's internal retention script. An agent who refuses to process a cancellation or transfer to a manager is FAILING to adhere to proper customer service process.
- overall: When the customer's core request went completely unresolved, the overall score CANNOT exceed the helpfulness score. An overall of 4 or 5 is only valid when the issue was actually resolved or the customer expressed satisfaction.
- agent_empathy_score: Scripted retention phrases ("I'd hate to lose you", "let me see what offers I have") are NOT empathy. Real empathy = acknowledging the customer's situation, apologising for failures, validating frustration. Score these exchanges at 0.1–0.3 unless genuine empathy is clearly expressed.
- resolution_status: Do NOT confuse "the call ended" with "resolved". An agent who refuses to process a cancellation, who keeps offering deals instead of honouring the request, or who leaves the customer's stated need unaddressed = "unresolved" regardless of how politely or professionally the call was conducted. If the customer's core request was NOT completed by the end of the call, resolution_status MUST be "unresolved".

SPEAKER LABEL RELIABILITY NOTE:
Speaker labels (Agent/Customer) are produced by an automated diarization system and can sometimes be swapped, especially for calls that begin mid-conversation without a clear opening greeting. If the transcript shows:
- "Agent" repeatedly saying things like "I want to cancel", "I'd like a refund", or "please cancel my service"
- "Customer" using retention phrases like "before I process that", "I'd hate to lose you", "let me offer you a deal", "let me see what offers I have"
...treat the labels as SWAPPED when forming your assessment. Rate based on who is ACTUALLY behaving like a customer service agent and who is the customer — ignore the assigned label and judge by behaviour.

Output ONLY valid JSON — no extra text, no markdown, no code blocks:
{
  "summary": "string",
  "customer_tone": "string",
  "agent_tone": "string",
  "detailed_ratings": {
    "helpfulness": int,
    "respect": int,
    "clarity": int,
    "adherence": int,
    "overall": int
  },
  "emotional_assessment": {
    "customer_satisfaction": "string",
    "resolution_status": "string",
    "call_tone": "string",
    "emotional_trajectory": "string",
    "start_emotion": "string",
    "end_emotion": "string",
    "trajectory_description": "string",
    "agent_empathy_score": float,
    "customer_frustration_level": float,
    "emotional_narrative": "string"
  }
}"""

def _convert_to_transcript(data):
    if isinstance(data, str):
        if data.strip().startswith('{') or data.strip().startswith('['):
            try:
                data = json.loads(data)
            except:
                return data
        else:
            return data
    
    if isinstance(data, dict):
        if "utterances" in data:
            utterances = data["utterances"]
        else:
            return None
    elif isinstance(data, list):
        utterances = data
    else:
        return str(data)
    
    lines = []
    for u in utterances:
        role = u.get("role") or u.get("speaker") or "Unknown"
        text = u.get("text", "")
        if text:
            lines.append(f"{role}: {text}")
    
    return "\n".join(lines)

def analyze_call(transcript, model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS):
    transcript_text = _convert_to_transcript(transcript)

    if not transcript_text:
        return {
            "summary": "Invalid input format",
            "rating": 3,
            "error": "Could not parse transcript"
        }

    transcript_lines = transcript_text.split('\n')
    agent_lines = [line for line in transcript_lines if line.startswith('Agent')]
    first_agent_line = agent_lines[0][:100] if agent_lines else "(none)"
    logger.info(
        "GPT-4 request: model=%s, transcript_turns=%d, first_agent_line=\"%s\"",
        model, len(transcript_lines), first_agent_line
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcript:\n{transcript_text}"}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        results = json.loads(response.choices[0].message.content)

        if "detailed_ratings" in results and "overall" in results["detailed_ratings"]:
            results["rating"] = results["detailed_ratings"]["overall"]
        elif "ratings" in results and "overall" in results["ratings"]:
            results["rating"] = results["ratings"]["overall"]
            results["detailed_ratings"] = results.pop("ratings")

        # Log what GPT-4 actually returned — critical for diagnosing wrong scores
        ea = results.get("emotional_assessment", {})
        logger.info(
            "GPT-4 response: ratings=%s, agent_tone=%s, customer_tone=%s, "
            "resolution=%s, satisfaction=%s, call_tone=%s, "
            "empathy=%.2f, frustration=%.2f",
            results.get("detailed_ratings"),
            results.get("agent_tone"),
            results.get("customer_tone"),
            ea.get("resolution_status"),
            ea.get("customer_satisfaction"),
            ea.get("call_tone"),
            ea.get("agent_empathy_score") or 0.0,
            ea.get("customer_frustration_level") or 0.0,
        )

        return results

    except json.JSONDecodeError:
        logger.error("GPT-4 JSON decode error — raw response could not be parsed")
        return {
            "summary": "Failed to parse model response",
            "rating": 3,
            "error": "JSON decode error"
        }
    except Exception as e:
        logger.error("GPT-4 API call failed: %s", e)
        return {
            "summary": "API call failed",
            "rating": 3,
            "error": str(e)
        }

