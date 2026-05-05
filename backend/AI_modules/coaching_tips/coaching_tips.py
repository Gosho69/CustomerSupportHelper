import json
import logging
import os
from collections import Counter
from datetime import datetime
from typing import List, Dict, Optional, Any
import requests

logger = logging.getLogger(__name__)

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def _normalize_rating(val: Any) -> Optional[float]:
    try:
        f = float(val)
        if f > 10:
            return round((f / 100.0) * 5.0, 2)
        return round(f, 2)
    except Exception:
        return None


def _turn_word_counts(turns: List[Dict]) -> Dict[str, int]:
    agent = 0
    customer = 0
    for t in turns:
        wc = t.get('word_count') or len(t.get('text', '').split())
        if t.get('speaker', '').lower().startswith('agent'):
            agent += wc
        else:
            customer += wc
    return {"agent_words": agent, "customer_words": customer}


def _check_ollama_available() -> bool:
    try:
        ollama_base = OLLAMA_API_URL.replace("/api/generate", "")
        response = requests.get(f"{ollama_base}/api/tags", timeout=5)
        available = response.status_code == 200
        if not available:
            logger.warning("Ollama /api/tags returned status %s — falling back to heuristic coaching", response.status_code)
        return available
    except Exception as e:
        logger.warning("Ollama not reachable at %s — falling back to heuristic coaching. Error: %s", OLLAMA_API_URL, e)
        return False


def _generate_tips_with_ai(turns, call_summary, emotion_summary, ratings, summary_text):
    context_parts = []

    if summary_text:
        context_parts.append(f"CALL SUMMARY:\n{summary_text}")

    # Always include a transcript excerpt — long calls are truncated to first 15 + last 15
    # turns so Ollama can see how the call opened AND how it closed.
    if turns:
        _MAX = 30
        if len(turns) > _MAX:
            half = _MAX // 2
            shown = turns[:half] + [{"speaker": "…", "text": f"[{len(turns) - _MAX} turns omitted]", "emotion": "", "sentiment": ""}] + turns[-half:]
        else:
            shown = turns
        context_parts.append("\nCONVERSATION TRANSCRIPT:")
        for turn in shown:
            speaker  = turn.get("speaker",   "Unknown")
            text     = turn.get("text",       "")[:150]
            emotion  = turn.get("emotion",    "")
            sentiment = turn.get("sentiment", "")
            tag = f"({emotion}/{sentiment})" if emotion or sentiment else ""
            context_parts.append(f"[{speaker}] {tag}: {text}")

    if emotion_summary:
        context_parts.append(f"\nEMOTION DISTRIBUTION:\n{json.dumps(emotion_summary, indent=2)}")

    if ratings:
        context_parts.append(f"\nRATINGS (scale 1–5):\n{json.dumps(ratings, indent=2)}")

    context = "\n".join(context_parts)

    # Adapt the directive based on outcome quality.
    # When helpfulness or overall are ≤ 2 the call had a serious problem — tell
    # Ollama this explicitly so it doesn't return an empty array.
    helpfulness = (ratings or {}).get("helpfulness")
    overall     = (ratings or {}).get("overall")
    serious_failure = (
        (helpfulness is not None and float(helpfulness) <= 2) or
        (overall     is not None and float(overall)     <= 2)
    )

    if serious_failure:
        directive = (
            "IMPORTANT: The ratings above indicate serious quality issues "
            "(helpfulness or overall ≤ 2/5). You MUST generate actionable coaching "
            "tips — do NOT return an empty array. Identify specifically what the "
            "agent did wrong and how to improve."
        )
    else:
        directive = (
            "Return empty array [] if the call was handled well.\n"
            "Return tips array only if significant problems exist."
        )

    prompt = f"""Analyze this customer service call and identify problems the agent made.

{context}

Flag issues when the agent:
- Was rude, dismissive, or obstructive
- Showed frustration or negative emotions
- Interrupted the customer repeatedly
- Talked excessively without listening
- Failed to fulfill the customer's explicit request (e.g. cancel, refund, disconnect)
- Used stalling or retention tactics instead of helping

DO NOT flag minor politeness issues, slightly imperfect wording, or small style improvements.

{directive}

Response format:
[{{"tip": "What the agent did wrong and how to fix it", "justification": "Why this is problematic", "evidence": {{"issue": "type"}}}}]

Return ONLY a JSON array, no other text."""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 1500
        }
    }

    for attempt in range(2):
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)

            if response.status_code != 200:
                logger.warning("Ollama generate returned HTTP %s (attempt %d)", response.status_code, attempt + 1)
                return None

            response_data = response.json()
            response_text = response_data.get("response", "").strip()

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            tips = json.loads(response_text)

            if isinstance(tips, list) and len(tips) > 0:
                logger.info("Ollama generated %d coaching tip(s) on attempt %d", len(tips), attempt + 1)
                return tips
            else:
                logger.info("Ollama returned empty tips array — no significant issues found")
                return None

        except Exception as e:
            logger.warning("Ollama generate attempt %d failed: %s", attempt + 1, e)
            if attempt == 0:
                continue
            return None


def generate_coaching_tips(transcript=None, turns=None, summary=None, call_summary=None, emotion_analysis=None, emotion_summary=None, ratings=None):
    tips = []

    if emotion_analysis and not turns:
        turns = emotion_analysis
    
    if call_summary:
        if not emotion_summary and 'emotion_distribution' in call_summary:
            emotion_summary = {
                k: v['count'] for k, v in call_summary['emotion_distribution'].items()
            }
        if not summary and 'summary' in call_summary:
            summary = call_summary['summary']
        if not ratings:
            ratings = {
                'customer_satisfaction': call_summary.get('customer_satisfaction', 'unknown'),
                'agent_empathy_score': call_summary.get('agent_empathy_score', 0),
                'customer_frustration_level': call_summary.get('customer_frustration_level', 0),
            }

    norm_ratings = {k: _normalize_rating(v) for k, v in (ratings or {}).items()} if ratings else {}
    
    if emotion_summary is None and turns:
        emotions = [t.get('emotion') for t in turns if t.get('emotion')]
        emotion_summary = dict(Counter(emotions))

    talk = _turn_word_counts(turns or [])

    ollama_available = _check_ollama_available()
    ai_tips = None

    if ollama_available:
        ai_tips = _generate_tips_with_ai(
            turns=turns,
            call_summary=call_summary,
            emotion_summary=emotion_summary,
            ratings=ratings,
            summary_text=summary
        )

        if ai_tips:
            tips = ai_tips
        else:
            logger.info("Ollama returned no tips — falling back to heuristic coaching")
    else:
        logger.warning("Ollama unavailable (URL: %s) — using heuristic coaching", OLLAMA_API_URL)

    if not tips:
        agent_turns = [t for t in turns if t.get('speaker','').lower().startswith('agent')] if turns else []
        customer_turns = [t for t in turns if t.get('speaker','').lower().startswith('customer')] if turns else []
        
        if norm_ratings:
            very_low = [(k, v) for k, v in norm_ratings.items() if v is not None and v <= 2.0]
            if very_low:
                for k, v in very_low[:2]:
                    tips.append({
                        "tip": f"Critical issue with {k.replace('_',' ')}: Review communication techniques and ensure clear, helpful responses.",
                        "justification": f"The {k.replace('_',' ')} score was very low ({v}/5), indicating a significant problem that needs immediate attention.",
                        "evidence": {"rating": {k: v}, "severity": "critical"}
                    })
        
        if agent_turns and customer_turns:
            agent_negative_emotions = [t for t in agent_turns if t.get('emotion') in ['angry', 'frustrated', 'sad']]
            agent_dismissive = [t for t in agent_turns if t.get('sentiment') == 'negative' and not t.get('contains_empathy') and not t.get('contains_apology')]
            
            if len(agent_negative_emotions) >= 2:
                tips.append({
                    "tip": "Maintain composure and professionalism. Agent showed frustration or negative emotions during the call.",
                    "justification": f"Agent displayed negative emotions in {len(agent_negative_emotions)} turns. This can escalate customer frustration and damage rapport.",
                    "evidence": {"agent_negative_turns": len(agent_negative_emotions), "issue": "agent_frustration"}
                })
            
            customer_angry_turns = [t for t in customer_turns if t.get('emotion') == 'angry']
            if len(customer_angry_turns) >= 2 and len(agent_dismissive) >= 2:
                tips.append({
                    "tip": "When customer is upset, show empathy and acknowledge their concern before providing solutions.",
                    "justification": f"Customer was clearly upset ({len(customer_angry_turns)} angry turns), but agent responses appeared dismissive or lacked empathy.",
                    "evidence": {"customer_angry_turns": len(customer_angry_turns), "agent_dismissive_turns": len(agent_dismissive)}
                })
        
        if talk:
            agent_words = talk.get('agent_words', 0)
            customer_words = talk.get('customer_words', 0)
            
            if agent_words > customer_words * 3.0 and customer_words > 50:
                tips.append({
                    "tip": "Allow customer to fully explain their issue. Agent talked too much, which can frustrate customers.",
                    "justification": f"Agent spoke {agent_words} words vs customer's {customer_words} words (3:1 ratio). Over-talking prevents understanding customer needs.",
                    "evidence": {"talk_ratio": round(agent_words/max(customer_words, 1), 1), "issue": "over_talking"}
                })
            
            if call_summary and isinstance(call_summary, dict):
                interruptions = call_summary.get('interruption_analysis', {})
                agent_interrupts = interruptions.get('agent_interruptions', 0)
                
                if agent_interrupts >= 3:
                    tips.append({
                        "tip": "Stop interrupting the customer. Let them finish speaking before responding.",
                        "justification": f"Agent interrupted customer {agent_interrupts} times. This is disrespectful and prevents proper issue resolution.",
                        "evidence": {"agent_interruptions": agent_interrupts, "issue": "interrupting"}
                    })

    if not tips:
        return {
            "created_at": datetime.utcnow().isoformat() + 'Z',
            "generation_method": "ai_powered" if (ollama_available and ai_tips) else "heuristic_based",
            "quality_status": "excellent",
            "message": "Great job! This call was handled professionally with no significant issues identified. Keep up the excellent work."
        }

    created_at = datetime.utcnow().isoformat() + 'Z'

    return {
        "created_at": created_at,
        "generation_method": "ai_powered" if (ollama_available and ai_tips) else "heuristic_based",
        "quality_status": "needs_improvement",
        "tips": tips
    }


def generate(transcript, summary_result=None, emotion_result=None, behavioral_result=None):
    if isinstance(transcript, str):
        transcript = json.loads(transcript)
    
    if not isinstance(transcript, dict):
        raise ValueError("Transcript must be a dict or JSON string")
    
    utterances = transcript.get("utterances", [])
    if not utterances:
        raise ValueError("Transcript must contain 'utterances' key")
    
    turns = []
    for idx, utt in enumerate(utterances):
        turn = {
            'speaker': utt.get('role', 'Unknown'),
            'text': utt.get('text', ''),
            'word_count': len(utt.get('text', '').split())
        }
        
        if emotion_result and isinstance(emotion_result, list):
            if idx < len(emotion_result):
                emotion_turn = emotion_result[idx]
                turn.update({
                    'emotion': emotion_turn.get('emotion'),
                    'sentiment': emotion_turn.get('sentiment'),
                    'contains_empathy': emotion_turn.get('contains_empathy'),
                    'contains_apology': emotion_turn.get('contains_apology')
                })
        
        turns.append(turn)
    
    summary_text = None
    ratings = None
    
    if summary_result:
        summary_text = summary_result.get('summary')
        
        if 'detailed_ratings' in summary_result:
            ratings = summary_result['detailed_ratings']
        elif 'ratings' in summary_result:
            ratings = summary_result['ratings']
        elif 'rating' in summary_result:
            ratings = {'overall': summary_result['rating']}
    
    call_summary_dict = None
    if behavioral_result:
        call_summary_dict = {
            'behavioral_score': behavioral_result.get('behavioral_score'),
            'words_per_minute': behavioral_result.get('words_per_minute'),
            'silence_analysis': behavioral_result.get('silence_analysis'),
            'interruption_analysis': behavioral_result.get('interruption_analysis'),
            'overall_assessment': behavioral_result.get('overall_assessment')
        }
    
    emotion_summary = None
    if emotion_result and isinstance(emotion_result, list):
        emotions = [t.get('emotion') for t in emotion_result if t.get('emotion')]
        emotion_summary = dict(Counter(emotions))
    
    return generate_coaching_tips(
        transcript=transcript,
        turns=turns,
        summary=summary_text,
        call_summary=call_summary_dict,
        emotion_analysis=emotion_result,
        emotion_summary=emotion_summary,
        ratings=ratings
    )



