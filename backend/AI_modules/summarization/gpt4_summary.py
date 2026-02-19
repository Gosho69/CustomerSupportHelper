import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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
   - **resolution_status**: Was the customer's issue resolved?
     Choose ONE: "resolved", "pending", "unresolved"
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
        
        return results
    
    except json.JSONDecodeError:
        return {
            "summary": "Failed to parse model response",
            "rating": 3,
            "error": "JSON decode error"
        }
    except Exception as e:
        return {
            "summary": "API call failed",
            "rating": 3,
            "error": str(e)
        }

