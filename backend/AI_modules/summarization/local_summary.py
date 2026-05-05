import json, re, torch, os, logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import phrase_loader as _pl

logger = logging.getLogger(__name__)

SYSTEM = (
    "Analyze this customer service call transcript. "
    "Output ONLY valid JSON with these exact keys: "
    "summary (2-3 sentences: what customer needed, how agent helped, outcome), "
    "customer_tone (one of: Positive, Negative, Neutral, Frustrated, Satisfied), "
    "agent_tone (one of: Professional, Helpful, Dismissive, Apologetic, Positive, Neutral), "
    "ratings (object with helpfulness, respect, clarity, adherence, overall — each 1-5). "
    "Rate based strictly on what the transcript shows."
)
TAIL = "JSON output only:"

_model = None
_tokenizer = None
_device = None

def _load_model(checkpoint_path="out-flan-sft1/final"):
    global _model, _tokenizer, _device

    if _model is None:
        source = "local" if os.path.exists(checkpoint_path) else "HuggingFace Hub"
        logger.info("Local summary model: loading from %s (%s)", checkpoint_path, source)
        try:
            if os.path.exists(checkpoint_path):
                _tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
                _model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, local_files_only=True)
            else:
                _tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
                _model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

            _device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            _model.to(_device).eval()
            logger.info("Local summary model: loaded successfully on device=%s", _device)
        except Exception as e:
            logger.error("Local summary model: FAILED to load from '%s' — %s", checkpoint_path, e)
            raise

    return _model, _tokenizer, _device

def _build_prompt(transcript):
    lines = transcript.split('\n')
    if len(lines) > 30:
        # Keep first 10 lines (problem statement) + last 20 lines (resolution).
        # 30 lines × ~12 tokens + ~100 overhead ≈ 460 tokens — safely within the
        # 512-token model budget.  The old first-30+last-20=50 line strategy caused
        # the tokenizer to silently drop the resolution from the right end.
        transcript = '\n'.join(lines[:10]) + '\n...(call continues)\n' + '\n'.join(lines[-20:])
    
    prompt = (
        f"[SYSTEM]\n{SYSTEM}\n[/SYSTEM]\n"
        f"[USER]\nTranscript:\n{transcript}\n[/USER]\n"
        f"{TAIL}"
    )
    return prompt

def _load_conversation_from_file(filepath):
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
        return transcript if transcript else None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if "utterances" in data:
            turns = data["utterances"]
        else:
            return None
    elif isinstance(data, list):
        turns = data
    else:
        return None
    
    transcript_lines = []
    for turn in turns:
        speaker = turn.get("role") or turn.get("speaker") or "Unknown"
        text = turn.get("text", "")
        transcript_lines.append(f"{speaker}: {text}")
    
    return "\n".join(transcript_lines)

def _convert_utterances_to_transcript(utterances):
    speaker_map = {}
    
    for utt in utterances:
        speaker_id = utt.get("role") or utt.get("speaker") or "Unknown"
        text = utt.get("text", "").strip()
        
        if not text:
            continue
        
        if speaker_id not in speaker_map:
            if speaker_id in ["Agent", "Customer"]:
                speaker_map[speaker_id] = speaker_id
            elif len(speaker_map) == 0:
                speaker_map[speaker_id] = "Agent"
            elif len(speaker_map) == 1:
                speaker_map[speaker_id] = "Customer"
            else:
                speaker_map[speaker_id] = speaker_id
    
    # Merge consecutive same-role utterances into one line.
    # Synthetic diarization can fragment a single speaker turn into several short
    # entries (e.g., agent pauses 0.6 s mid-sentence → two "Agent:" lines).
    # Merging keeps the line count low and gives the model coherent full turns.
    merged: list[tuple[str, str]] = []
    for turn in utterances:
        speaker_id = turn.get("role") or turn.get("speaker") or "Unknown"
        text = turn.get("text", "").strip()

        if not text:
            continue

        role = speaker_map.get(speaker_id, speaker_id)
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + " " + text)
        else:
            merged.append((role, text))

    return "\n".join(f"{role}: {text}" for role, text in merged)

def _parse_json_safe(text):
    try:
        return json.loads(text)
    except:
        pass
    
    if not text.strip().startswith('{'):
        try:
            return json.loads('{' + text + '}')
        except:
            pass
    
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    try:
        fixed = text if text.strip().startswith('{') else '{' + text + '}'
        fixed = re.sub(r'"ratings"\s*:\s*"', '"ratings": {"', fixed)
        fixed = re.sub(r'("overall"\s*:\s*\d+)(\s*[,}]|$)', r'\1}\2', fixed)
        return json.loads(fixed)
    except:
        pass
    
    return None

def _improve_summary(summary, transcript):
    if summary.startswith("The customer states:"):
        lines = transcript.split('\n')
        customer_lines = [l.replace('Customer:', '').strip() for l in lines if l.startswith('Customer:')]
        agent_lines = [l.replace('Agent:', '').strip() for l in lines if l.startswith('Agent:')]
        
        if customer_lines:
            issue = customer_lines[0][:100]
            resolution = "Agent assisted with the inquiry."
            if len(agent_lines) > 1:
                last_agent = agent_lines[-1][:80]
                if any(word in last_agent.lower() for word in ['welcome', 'help', 'else', 'thank']):
                    resolution = "Issue was addressed and call concluded positively."
            
            summary = f"Customer called about: {issue}. {resolution}"
    
    return summary

def _apply_outcome_correction(result, transcript_text):
    """
    Post-process ratings using transcript evidence alone — independent of the
    fine-tuned model's tone classification, which is unreliable for nuanced calls.

    Three signal families drive corrections:
    1. Unresolved-request density — customer repeated a specific request many times
    2. Agent refusal/retention language — broad set matching real agent speech patterns
    3. Model tone consistency — tone + dismissiveness as a secondary guard

    Rules are intentionally independent so that any single strong signal fires the
    correction, rather than requiring two signals to co-occur.
    """
    ratings = result.get("detailed_ratings", {})
    if not ratings:
        return result

    customer_tone = (result.get("customer_tone") or "").lower()
    agent_tone    = (result.get("agent_tone")    or "").lower()

    transcript_lower = transcript_text.lower() if isinstance(transcript_text, str) else ""

    # ── Customer request signals ──────────────────────────────────────────────
    # Individual words/phrases that indicate an unresolved specific request.
    # We COUNT them — a single occurrence of "cancel" is not enough; 3+ in an
    # 8-minute call means the customer kept repeating themselves (= not resolved).
    _REQUEST_WORDS_DEFAULT = [
        "cancel", "cancellation", "cancelled", "disconnect", "disconnecting",
        "close my account", "close the account", "speak to a manager",
        "speak to your supervisor", "speak to a supervisor", "transfer me",
        "refund", "billing error", "overcharged", "charged me wrong",
    ]

    # ── Agent refusal / retention language ───────────────────────────────────
    # Broad set — matches how retention agents actually speak, not ideal phrasing.
    # One match is enough to confirm the agent was blocking the request.
    _REFUSAL_SIGNALS_DEFAULT = [
        # Inability / refusal
        "i'm not able to", "i am not able to",
        "i can't do that", "i cannot do that",
        "that's not something i can", "that's not something we can",
        "that's not possible", "i don't have the ability",
        "my hands are tied",
        # Deflection / stalling
        "i understand but", "i hear you but", "i know but",
        "before i do that", "before i go ahead",
        "before i process", "before we proceed",
        # Retention offers (agent is trying to keep rather than resolve)
        "let me see what i can do for you", "let me see what i can do",
        "let me see what offers", "let me see what i can offer",
        "let me pull up some offers", "let me check what deals",
        "i have some great offers", "we have a great offer",
        "i can offer you", "what if i offered",
        "i'd hate to lose you", "i'd hate to see you go",
        "we value you as a customer",
        # Ignoring the request
        "instead of cancelling", "instead of disconnecting",
        "have you considered", "what if instead",
    ]

    # ── Compute signals ───────────────────────────────────────────────────────
    request_words = _pl.get("request_words",    _REQUEST_WORDS_DEFAULT)
    refusal_sigs  = _pl.get("refusal_signals",  _REFUSAL_SIGNALS_DEFAULT)
    request_hits  = sum(transcript_lower.count(w) for w in request_words)
    refusal_hit   = any(sig in transcript_lower for sig in refusal_sigs)

    # ── Pre-pass: tone correction based on transcript evidence ───────────────
    # The local FLAN-T5 model frequently mis-classifies tone as Satisfied/Helpful
    # in contentious retention calls.  We override with evidence-based values
    # BEFORE evaluating the tone-dependent rules below, so those rules can fire
    # even when the model was confidently wrong about tone.
    if request_hits >= 5 and customer_tone in ("satisfied", "positive", "neutral"):
        old_ct = customer_tone
        customer_tone = "frustrated"
        result["customer_tone"] = "Frustrated"
        logger.warning(
            "Outcome correction Pre-pass: customer_tone %s→Frustrated (request_hits=%d)",
            old_ct, request_hits
        )
    if request_hits >= 10 and agent_tone in ("helpful", "positive", "neutral", "professional"):
        old_at = agent_tone
        agent_tone = "dismissive"
        result["agent_tone"] = "Dismissive"
        logger.warning(
            "Outcome correction Pre-pass: agent_tone %s→Dismissive (request_hits=%d)",
            old_at, request_hits
        )

    customer_unsatisfied = customer_tone in ("frustrated", "negative", "angry")
    agent_dismissive     = agent_tone == "dismissive"

    helpfulness = ratings.get("helpfulness", 3)
    respect     = ratings.get("respect",     3)

    logger.info(
        "_apply_outcome_correction: customer_tone=%s, agent_tone=%s, "
        "request_hits=%d, refusal_hit=%s",
        customer_tone, agent_tone, request_hits, refusal_hit
    )

    # Rule 1: model says customer is unhappy AND agent is dismissive
    if customer_unsatisfied and agent_dismissive:
        ratings["helpfulness"] = min(helpfulness, 2)
        ratings["respect"]     = min(respect,     2)
        logger.warning("Outcome correction Rule 1 fired (tone+dismissive): helpfulness→%d, respect→%d",
                       ratings["helpfulness"], ratings["respect"])

    # Rule 2: customer repeated a specific request 3+ times AND agent used
    # retention/refusal language → the request was never fulfilled
    if request_hits >= 3 and refusal_hit:
        ratings["helpfulness"] = min(ratings.get("helpfulness", 3), 2)
        logger.warning("Outcome correction Rule 2 fired (request_hits=%d, refusal): helpfulness→%d",
                       request_hits, ratings["helpfulness"])

    # Rule 3: customer repeated a request 5+ times regardless of agent phrasing
    # (volume alone means the issue was never resolved)
    if request_hits >= 5:
        ratings["helpfulness"] = min(ratings.get("helpfulness", 3), 2)
        logger.warning("Outcome correction Rule 3 fired (request_hits=%d): helpfulness→%d",
                       request_hits, ratings["helpfulness"])

    # Rule 3b: extremely high request density means the agent failed to efficiently
    # address the customer — respect and adherence are also low in such calls.
    # Does NOT require customer_unsatisfied (tone may still be mis-classified).
    if request_hits >= 10:
        ratings["respect"]    = min(ratings.get("respect", 3), 2)
        ratings["adherence"]  = min(ratings.get("adherence", 3), 3)
        logger.warning(
            "Outcome correction Rule 3b fired (request_hits=%d): respect→%d, adherence→%d",
            request_hits, ratings["respect"], ratings["adherence"]
        )

    # Rule 4: overall cannot exceed helpfulness when it was forced low
    if ratings.get("helpfulness", 3) <= 2:
        ratings["overall"] = min(ratings.get("overall", 3), ratings["helpfulness"])

    # Rule 5: clarity cannot be 4-5 when the agent was dismissive.
    # "Exceptional clarity" (5) requires clearly addressing the customer's need —
    # a dismissive agent who refuses, deflects, or stonewalls cannot score that.
    # Cap at 3 (mostly clear) to reflect that the agent may speak fluently while
    # still failing to communicate anything useful to the customer.
    clarity = ratings.get("clarity", 3)
    if agent_dismissive and clarity > 3:
        ratings["clarity"] = 3
        logger.warning(
            "Outcome correction Rule 5 fired (agent_dismissive): clarity→3"
        )

    # Rule 5b: when the customer repeated a request 10+ times, the agent's
    # communication was clearly not effective — clarity should not stay at 5.
    if request_hits >= 10 and ratings.get("clarity", 3) > 3:
        ratings["clarity"] = 3
        logger.warning(
            "Outcome correction Rule 5b fired (request_hits=%d): clarity→3",
            request_hits
        )

    result["detailed_ratings"] = ratings
    result["rating"]           = ratings.get("overall", result.get("rating", 3))
    return result


def analyze_call(transcript, checkpoint_path="../model/final"):
    if transcript is None:
        return {
            "summary": "No conversation provided",
            "rating": 3,
            "error": "No input data"
        }
    
    if isinstance(transcript, str):
        if transcript.strip().startswith('{') or transcript.strip().startswith('['):
            try:
                transcript = json.loads(transcript)
            except:
                pass
    
    if isinstance(transcript, dict):
        if "utterances" in transcript:
            transcript = _convert_utterances_to_transcript(transcript["utterances"])
        else:
            return {
                "summary": "Invalid JSON format",
                "rating": 3,
                "error": "JSON must have 'utterances' key"
            }
    elif isinstance(transcript, list):
        transcript = _convert_utterances_to_transcript(transcript)
    elif not isinstance(transcript, str):
        return {
            "summary": "Invalid input type",
            "rating": 3,
            "error": f"Expected str, dict, or list, got {type(transcript).__name__}"
        }
    
    transcript_lines = transcript.split('\n')
    agent_lines    = [line for line in transcript_lines if line.startswith('Agent')]
    customer_lines = [line for line in transcript_lines if line.startswith('Customer')]
    first_agent_line = agent_lines[0][:100] if agent_lines else "(none)"
    logger.info(
        "Local model request: transcript_turns=%d, agent_turns=%d, customer_turns=%d, "
        "first_agent_line=\"%s\"",
        len(transcript_lines), len(agent_lines), len(customer_lines), first_agent_line
    )

    model, tokenizer, device = _load_model(checkpoint_path)

    x = tokenizer(_build_prompt(transcript), return_tensors="pt", truncation=True, max_length=512).to(device)
    y = model.generate(
        **x,
        max_new_tokens=300,
        num_beams=4,
        do_sample=False,
        early_stopping=True
    )
    raw_output = tokenizer.decode(y[0], skip_special_tokens=True).strip()
    logger.info("Local model raw output: %.200s", raw_output)

    result_json = _parse_json_safe(raw_output)

    if result_json:
        summary       = result_json.get("summary", "")
        customer_tone = result_json.get("customer_tone", "Unknown")
        agent_tone    = result_json.get("agent_tone", "Unknown")
        ratings       = result_json.get("ratings", {})
        overall_rating = ratings.get("overall", 3)

        logger.info(
            "Local model parsed: customer_tone=%s, agent_tone=%s, ratings=%s",
            customer_tone, agent_tone, ratings
        )

        summary = _improve_summary(summary, transcript)

        result = {
            "summary": summary,
            "rating": overall_rating,
            "customer_tone": customer_tone,
            "agent_tone": agent_tone,
            "detailed_ratings": ratings
        }
        orig_ratings = dict(ratings)  # snapshot before in-place mutation inside correction
        corrected = _apply_outcome_correction(result, transcript)

        # Log if outcome correction changed any rating
        corr_ratings = corrected.get("detailed_ratings", {})
        if orig_ratings != corr_ratings:
            logger.warning(
                "Local model outcome correction applied: %s → %s",
                orig_ratings, corr_ratings
            )
        else:
            logger.info("Local model outcome correction: no changes (ratings unchanged)")

        return corrected
    else:
        logger.error(
            "Local model FAILED to produce valid JSON. Raw output (first 300 chars): %.300s",
            raw_output
        )
        lines = transcript.split('\n')
        customer_lines_fb = [line for line in lines if line.startswith('Customer:')]

        if customer_lines_fb:
            first_issue = customer_lines_fb[0].replace('Customer:', '').strip()[:150]
            summary = f"Customer contacted support regarding: {first_issue}. Agent provided assistance."
        else:
            summary = "Customer service interaction between agent and customer."

        return {
            "summary": summary,
            "rating": 3,
            "error": "Model failed to generate valid JSON"
        }