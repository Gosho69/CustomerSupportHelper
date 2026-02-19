import json, re, torch, os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
        if os.path.exists(checkpoint_path):
            _tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
            _model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, local_files_only=True)
        else:
            _tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            _model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        
        _device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device).eval()
    
    return _model, _tokenizer, _device

def _build_prompt(transcript):
    lines = transcript.split('\n')
    if len(lines) > 50:
        # Keep first 30 lines (opening) and last 20 lines (resolution/closing)
        transcript = '\n'.join(lines[:30]) + '\n...(call continues)\n' + '\n'.join(lines[-20:])
    
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
    
    transcript_lines = []
    for turn in utterances:
        speaker_id = turn.get("role") or turn.get("speaker") or "Unknown"
        text = turn.get("text", "").strip()
        
        if not text:
            continue
        
        role = speaker_map.get(speaker_id, speaker_id)
        transcript_lines.append(f"{role}: {text}")
    
    return "\n".join(transcript_lines)

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
    
    result_json = _parse_json_safe(raw_output)
    
    if result_json:
        summary = result_json.get("summary", "")
        customer_tone = result_json.get("customer_tone", "Unknown")
        agent_tone = result_json.get("agent_tone", "Unknown")
        ratings = result_json.get("ratings", {})
        overall_rating = ratings.get("overall", 3)
        
        summary = _improve_summary(summary, transcript)
        
        return {
            "summary": summary,
            "rating": overall_rating,
            "customer_tone": customer_tone,
            "agent_tone": agent_tone,
            "detailed_ratings": ratings
        }
    else:
        lines = transcript.split('\n')
        customer_lines = [l for l in lines if l.startswith('Customer:')]
        
        if customer_lines:
            first_issue = customer_lines[0].replace('Customer:', '').strip()[:150]
            summary = f"Customer contacted support regarding: {first_issue}. Agent provided assistance."
        else:
            summary = "Customer service interaction between agent and customer."
        
        return {
            "summary": summary,
            "rating": 3,
            "error": "Model failed to generate valid JSON"
        }