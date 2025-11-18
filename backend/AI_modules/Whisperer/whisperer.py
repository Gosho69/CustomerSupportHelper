import os, json, sys
import warnings
import soundfile as sf
import dotenv

warnings.filterwarnings("ignore", category=UserWarning, message=".*torchaudio.*deprecated.*")

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def _detect_audio_properties(audio_path: str):
    try:
        data, sr = sf.read(audio_path)
        is_stereo = data.ndim > 1 and data.shape[1] >= 2
        
        if is_stereo:
            return True, 2
        else:
            return False, None
            
    except Exception as e:
        return False, None


def _convert_pyannote_to_whisperx(pyannote_annotation):
    segments = []
    for segment, track, speaker in pyannote_annotation.itertracks(yield_label=True):
        segments.append({
            'start': segment.start,
            'end': segment.end,
            'speaker': speaker
        })
    
    return {'segments': segments}

def _assign_speakers_to_words(diarize_segments, aligned):
    speaker_timeline = diarize_segments.get('segments', [])
    
    if not speaker_timeline:
        return aligned
    
    for segment in aligned.get('segments', []):
        if 'words' not in segment or not segment['words']:
            continue
            
        for word in segment['words']:
            word_start = word.get('start', 0)
            word_end = word.get('end', 0)
            
            best_speaker = 'SPEAKER_00'
            best_overlap = 0.0
            
            for spk_seg in speaker_timeline:
                spk_start = spk_seg.get('start', 0)
                spk_end = spk_seg.get('end', 0)
                
                overlap_start = max(word_start, spk_start)
                overlap_end = min(word_end, spk_end)
                overlap = max(0.0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = spk_seg.get('speaker', 'SPEAKER_00')
            
            if best_overlap == 0.0 and speaker_timeline:
                closest = min(
                    speaker_timeline,
                    key=lambda s: abs(s.get('start', 0) - word_start)
                )
                best_speaker = closest.get('speaker', 'SPEAKER_00')
            
            word['speaker'] = best_speaker
    
    return aligned


def transcribe_mono_with_diarization(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    num_speakers: int | None = None,        
    agent_hint: str | None = None   
):
    import whisperx
    import os
    import tempfile
    import subprocess

    if device not in ["cpu", "cuda"]:
        device = "cpu"

    asr_model = whisperx.load_model(model_size, device, compute_type=compute_type)
    batch_size = 16 if device != "cpu" else 4
    asr_result = asr_model.transcribe(audio_path, batch_size=batch_size)
    audio = whisperx.load_audio(audio_path)
    align_model, metadata = whisperx.load_align_model(asr_result["language"], device)
    aligned = whisperx.align(asr_result["segments"], align_model, metadata, audio, device)
    
    hf_token = HF_TOKEN
    clean_token = hf_token.strip('"').strip("'") if hf_token else None
    
    diarized = None
    diarization_failed = False
    
    audio_for_diarization = audio_path
    wav_temp_path = None
    
    if audio_path.lower().endswith(('.mp3', '.m4a', '.flac')):
        try:
            wav_temp_path = tempfile.mktemp(suffix=".wav")
            subprocess.run(
                ['ffmpeg', '-i', audio_path, '-acodec', 'pcm_s16le', '-ar', '16000', wav_temp_path, '-y'],
                capture_output=True, check=True
            )
            audio_for_diarization = wav_temp_path
        except Exception as e:
            audio_for_diarization = audio_path
    
    try:
        from pyannote.audio import Pipeline
        
        diarize_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=clean_token
        )
        
        if num_speakers is not None:
            diarize_result = diarize_pipeline(audio_for_diarization, num_speakers=num_speakers)
        else:
            diarize_result = diarize_pipeline(audio_for_diarization)
        diarize_segments = _convert_pyannote_to_whisperx(diarize_result)
        diarized = _assign_speakers_to_words(diarize_segments, aligned)
        
    except Exception as e:
        diarization_failed = True
    
    if wav_temp_path and os.path.exists(wav_temp_path):
        try:
            os.remove(wav_temp_path)
        except:
            pass
    
    if diarization_failed or diarized is None:
        diarized = _create_synthetic_diarization(aligned)

    utterances = []
    current = None
    
    for segment in diarized.get("segments", []):
        if 'words' not in segment or not segment['words']:
            continue
        
        for word in segment['words']:
            spk = word.get('speaker', 'SPEAKER_00')
            word_start = word.get('start', 0)
            word_end = word.get('end', 0)
            word_text = word.get('word', '').strip()
            
            if not word_text:
                continue
            
            if current is None or spk != current["speaker"]:
                if current:
                    current["text"] = current["text"].strip()
                    if current["text"]:
                        utterances.append(current)
                
                current = {
                    "speaker": spk,
                    "start": word_start,
                    "end": word_end,
                    "text": word_text
                }
            else:
                current["end"] = word_end
                if current["text"] and not current["text"].endswith(" "):
                    current["text"] += " "
                current["text"] += word_text
    
    if current:
        current["text"] = current["text"].strip()
        if current["text"]:
            utterances.append(current)

    def _enforce_speaker_continuity(utterances_list, max_short_utterance_duration=0.8, max_words=2):
        if not utterances_list:
            return utterances_list

        out = []
        for u in utterances_list:
            duration = u.get('end', 0) - u.get('start', 0)
            word_count = len(u.get('text', '').split()) if u.get('text') else 0

            if out and (duration <= max_short_utterance_duration or word_count <= max_words) and u['speaker'] != out[-1]['speaker']:
                prev = out[-1]
                prev['end'] = max(prev['end'], u['end'])
                if prev.get('text') and not prev['text'].endswith(' '):
                    prev['text'] += ' '
                prev['text'] += u.get('text', '')
            else:
                out.append(u)

        merged = []
        i = 0
        while i < len(out):
            u = out[i]
            duration = u.get('end', 0) - u.get('start', 0)
            word_count = len(u.get('text', '').split()) if u.get('text') else 0
            if i < len(out) - 1 and (duration <= max_short_utterance_duration or word_count <= max_words) and u['speaker'] != out[i+1]['speaker']:
                nxt = out[i+1]
                nxt['start'] = min(nxt['start'], u['start'])
                if u.get('text'):
                    if not u['text'].endswith(' '):
                        u['text'] += ' '
                    nxt['text'] = u['text'] + nxt.get('text', '')
                i += 1 
                merged.append(nxt)
                i += 1
            else:
                merged.append(u)
                i += 1

        return merged

    utterances = _enforce_speaker_continuity(utterances)

    speakers = sorted(set(u["speaker"] for u in utterances))
    if agent_hint and agent_hint in speakers:
        agent_speaker = agent_hint
    else:
        if utterances:
            first = min(utterances, key=lambda u: u["start"])["speaker"]
            agent_speaker = first
        else:
            agent_speaker = "SPEAKER_00"
    
    role_map = {agent_speaker: "Agent"}
    for spk in speakers:
        if spk not in role_map:
            role_map[spk] = "Customer"

    turns = []
    for u in utterances:
        turns.append({
            "role": role_map[u["speaker"]],
            "start": round(float(u["start"]), 2),
            "end": round(float(u["end"]), 2),
            "text": u["text"]
        })
    turns.sort(key=lambda x: x["start"])

    duration = 0.0
    if turns:
        duration = max(duration, max(t["end"] for t in turns))

    return {
        "call_id": os.path.basename(audio_path),
        "duration_sec": round(duration, 2),
        "utterances": turns
    }


def _create_synthetic_diarization(aligned):
    MIN_SILENCE_GAP = 0.5  
    
    segments = aligned.get("segments", [])
    if not segments:
        return aligned
    
    words_or_segments = []
    
    for seg in segments:
        if "words" in seg:
            words_or_segments.extend(seg["words"])
        elif "word" in seg:
            words_or_segments.append(seg)
        else:
            words_or_segments.append(seg)
    
    if not words_or_segments:
        return aligned
    
    current_speaker = "SPEAKER_01"
    
    for i, item in enumerate(words_or_segments):
        if i == 0:
            current_speaker = "SPEAKER_00"
        
        item["speaker"] = current_speaker
        
        if i < len(words_or_segments) - 1:
            current_end = item.get("end", 0)
            next_start = words_or_segments[i + 1].get("start", 0)
            gap = next_start - current_end
            
            if gap > MIN_SILENCE_GAP:
                current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
    
    result = aligned.copy()
    result["segments"] = words_or_segments
    
    return result


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    agent_hint: str | None = None
):
    if device == "mps":
        device = "cpu"
    
    is_stereo, estimated_speakers = _detect_audio_properties(audio_path)
    num_speakers = estimated_speakers if estimated_speakers is not None else 2

    return transcribe_mono_with_diarization(
        audio_path=audio_path,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        num_speakers=num_speakers,
        agent_hint=agent_hint
    )