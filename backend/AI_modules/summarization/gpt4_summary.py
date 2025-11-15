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
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 800

SYSTEM_PROMPT = """You are an expert call quality analyst for customer service interactions.

Your task is to analyze customer service call transcripts and provide:

1. **SUMMARY** (2-3 clear sentences):
   - What was the customer's issue or need?
   - How did the agent help or respond?
   - What was the outcome (resolved, escalated, pending)?
   
2. **RATINGS** - Rate the agent's performance on a 1-5 scale:
   - **helpfulness**: Did the agent solve the customer's problem? (1=no help, 3=partial, 5=fully resolved)
   - **respect**: Was the agent courteous and professional? (1=rude, 3=neutral, 5=very respectful)
   - **clarity**: Was communication clear and easy to understand? (1=confusing, 3=adequate, 5=crystal clear)
   - **adherence**: Did the agent follow proper procedures? (1=ignored rules, 3=mostly followed, 5=perfect adherence)
   - **overall**: Overall service quality (1=very poor, 2=poor, 3=average, 4=good, 5=excellent)

Output ONLY valid JSON with these exact keys:
{
  "summary": "string",
  "ratings": {
    "helpfulness": int,
    "respect": int,
    "clarity": int,
    "adherence": int,
    "overall": int
  }
}

Be objective, concise, and professional. Focus on facts from the transcript."""

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
    """
    Analyze a customer service call using GPT-4.
    
    Args:
        transcript: Can be string, dict with 'utterances', list of utterances, or JSON string
        model: OpenAI model to use (default: gpt-4o-mini)
        temperature: Sampling temperature (default: 0.3)
        max_tokens: Maximum response tokens (default: 800)
    
    Returns:
        Dict with keys: summary, ratings (with helpfulness, respect, clarity, adherence, overall)
        Returns dict with error key if analysis fails
    """
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
        )

        results = json.loads(response.choices[0].message.content)
        
        if "ratings" in results and "overall" in results["ratings"]:
            results["rating"] = results["ratings"]["overall"]        
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

