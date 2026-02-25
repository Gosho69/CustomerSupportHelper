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
            import numpy as np
            left  = data[:, 0]
            right = data[:, 1]
            # If both channels are nearly identical, it's mono saved as stereo.
            # Use channel correlation: >0.99 means the same audio on both tracks.
            correlation = float(np.corrcoef(left, right)[0, 1])
            if correlation > 0.99:
                return False, None  # treat as mono → use diarization path
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
    
    last_speaker = speaker_timeline[0].get('speaker', 'SPEAKER_00') if speaker_timeline else 'SPEAKER_00'

    for segment in aligned.get('segments', []):
        if 'words' not in segment or not segment['words']:
            continue

        prev_word_speaker = None

        for word in segment['words']:
            word_start = word.get('start', 0)
            word_end = word.get('end', 0)
            word_mid = (word_start + word_end) / 2.0

            best_speaker = None
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

            if best_overlap == 0.0:
                # No time-overlap: prefer the segment that *contains* the word midpoint.
                # If none contains it, fall back to the previous word's speaker so we
                # don't incorrectly flip speakers mid-sentence.
                container = next(
                    (s for s in speaker_timeline
                     if s.get('start', 0) <= word_mid <= s.get('end', 0)),
                    None
                )
                if container:
                    best_speaker = container.get('speaker', 'SPEAKER_00')
                elif prev_word_speaker is not None:
                    # Same segment → very likely the same speaker continuing
                    best_speaker = prev_word_speaker
                else:
                    best_speaker = last_speaker

            word['speaker'] = best_speaker
            prev_word_speaker = best_speaker
            last_speaker = best_speaker

    return aligned


def transcribe_mono_with_diarization(
    audio_path: str,
    model_size: str = "small",
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
        print(f"[Whisperer] PyAnnote diarization failed ({type(e).__name__}): {e}", file=sys.stderr)
        diarization_failed = True
    
    if wav_temp_path and os.path.exists(wav_temp_path):
        try:
            os.remove(wav_temp_path)
        except:
            pass
    
    if diarization_failed or diarized is None:
        print("[Whisperer] Falling back to synthetic diarization (silence-gap method)", file=sys.stderr)
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

    def _enforce_speaker_continuity(utterances_list):
        # Only merge consecutive utterances from the same speaker.
        # Aggressive short-utterance merging (the old passes 1 & 2) was
        # causing genuine 1-2 word customer responses ("Yes", "Okay", "No",
        # "Right") to be absorbed into the wrong speaker's turn. Trust the
        # diarization model and only consolidate same-speaker fragments.
        if not utterances_list:
            return utterances_list
        final = []
        for u in utterances_list:
            if final and final[-1]['speaker'] == u['speaker']:
                prev = final[-1]
                prev['end'] = max(prev['end'], u['end'])
                if prev.get('text') and not prev['text'].endswith(' '):
                    prev['text'] += ' '
                prev['text'] += u.get('text', '')
            else:
                final.append(u)
        return final

    utterances = _enforce_speaker_continuity(utterances)

    speakers = sorted(set(u["speaker"] for u in utterances))
    if agent_hint and agent_hint in speakers:
        agent_speaker = agent_hint
    else:
        # The agent ALWAYS speaks first in a call-centre recording (greeting).
        # We use "first speaker whose utterance has 3+ words" so that a stray
        # single mis-assigned boundary word from pyannote cannot flip the roles.
        agent_speaker = None
        for u in sorted(utterances, key=lambda x: x["start"]):
            if len(u.get("text", "").split()) >= 3:
                agent_speaker = u["speaker"]
                break
        # Fallback: truly first speaker if no 3-word utterance found
        if agent_speaker is None:
            if utterances:
                agent_speaker = min(utterances, key=lambda u: u["start"])["speaker"]
            else:
                agent_speaker = "SPEAKER_00"
    
    role_map = {agent_speaker: "Agent"}
    non_agent_speakers = [spk for spk in speakers if spk != agent_speaker]
    if len(non_agent_speakers) == 1:
        role_map[non_agent_speakers[0]] = "Customer"
    else:
        for i, spk in enumerate(non_agent_speakers, 1):
            role_map[spk] = f"Customer {i}"

    turns = []
    for u in utterances:
        turns.append({
            "role": role_map[u["speaker"]],
            "start": round(float(u["start"]), 2),
            "end": round(float(u["end"]), 2),
            "text": u["text"]
        })
    turns.sort(key=lambda x: x["start"])

    # Merge any consecutive turns with the same role that may remain after sorting
    merged_turns = []
    for turn in turns:
        if merged_turns and merged_turns[-1]["role"] == turn["role"]:
            prev = merged_turns[-1]
            prev["end"] = max(prev["end"], turn["end"])
            if prev["text"] and not prev["text"].endswith(" "):
                prev["text"] += " "
            prev["text"] += turn["text"]
        else:
            merged_turns.append(turn)
    turns = merged_turns

    duration = 0.0
    if turns:
        duration = max(duration, max(t["end"] for t in turns))

    return {
        "call_id": os.path.basename(audio_path),
        "duration_sec": round(duration, 2),
        "utterances": turns
    }


def _create_synthetic_diarization(aligned):
    # 0.5 s covers typical inter-turn pauses in phone conversations (0.2–0.6 s).
    # The old 1.5 s threshold was too conservative and caused entire conversation
    # blocks to be merged into a single speaker.
    MIN_SILENCE_GAP = 0.5

    segments = aligned.get("segments", [])
    if not segments:
        return aligned

    # Collect all words with timing info in order
    all_words = []
    for seg in segments:
        if "words" in seg and seg["words"]:
            all_words.extend(seg["words"])
        else:
            # Segment has no word-level timing – synthesise a single word entry
            text = seg.get("text", "").strip()
            if text:
                all_words.append({
                    "word": text,
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                })

    if not all_words:
        return aligned

    # Assign speakers based on silence gaps
    current_speaker = "SPEAKER_00"
    for i, word in enumerate(all_words):
        word["speaker"] = current_speaker
        if i < len(all_words) - 1:
            gap = all_words[i + 1].get("start", 0) - word.get("end", 0)
            if gap > MIN_SILENCE_GAP:
                current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"

    # Re-pack words back into their original segments so the downstream
    # `for segment in diarized["segments"]: for word in segment["words"]` loop works.
    result = aligned.copy()
    new_segments = []
    word_idx = 0
    for seg in segments:
        new_seg = dict(seg)
        if "words" in seg and seg["words"]:
            count = len(seg["words"])
            new_seg["words"] = all_words[word_idx:word_idx + count]
            word_idx += count
        else:
            # Segments that had no words keep their synthetic single word
            if word_idx < len(all_words):
                new_seg["words"] = [all_words[word_idx]]
                word_idx += 1
        new_segments.append(new_seg)
    result["segments"] = new_segments
    return result


def transcribe_stereo_channels(
    audio_path: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    agent_hint: str | None = None,
):
    """
    Transcribe a stereo recording by splitting channels first.

    In call-centre recordings each channel typically carries a single speaker
    (e.g. left = agent, right = customer).  Transcribing the channels
    independently avoids diarization entirely, giving perfect speaker
    separation and better word accuracy (no mixed-speaker audio fed to ASR).

    The channel whose first utterance starts earliest is treated as the Agent,
    unless agent_hint is supplied.
    """
    import whisperx
    import tempfile
    import numpy as np

    data, sr = sf.read(audio_path)
    left  = data[:, 0]
    right = data[:, 1]

    results = {}
    for label, channel_data in [("ch0", left), ("ch1", right)]:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, channel_data, sr)
        tmp.close()
        try:
            asr_model = whisperx.load_model(model_size, device, compute_type=compute_type)
            batch_size = 16 if device != "cpu" else 4
            asr_result = asr_model.transcribe(tmp.name, batch_size=batch_size)
            audio = whisperx.load_audio(tmp.name)
            align_model, metadata = whisperx.load_align_model(asr_result["language"], device)
            aligned = whisperx.align(asr_result["segments"], align_model, metadata, audio, device)
            results[label] = aligned
        finally:
            try:
                os.remove(tmp.name)
            except Exception:
                pass

    def _extract_utterances(aligned, speaker_label):
        # Each WhisperX segment is a natural phrase/sentence boundary.
        # Create one utterance per segment so the transcript is readable,
        # not one giant utterance for the whole call.
        utterances = []
        for segment in aligned.get("segments", []):
            words = segment.get("words", [])
            if words:
                # Build text from word list (more accurate timings)
                word_items = [w for w in words if w.get("word", "").strip()]
                if not word_items:
                    continue
                seg_start = word_items[0].get("start", segment.get("start", 0))
                seg_end   = word_items[-1].get("end",   segment.get("end",   0))
                text = " ".join(w["word"].strip() for w in word_items)
            else:
                text = segment.get("text", "").strip()
                if not text:
                    continue
                seg_start = segment.get("start", 0)
                seg_end   = segment.get("end",   0)
            utterances.append({
                "speaker": speaker_label,
                "start": seg_start,
                "end":   seg_end,
                "text":  text,
            })
        return utterances

    ch0_utterances = _extract_utterances(results.get("ch0", {}), "ch0")
    ch1_utterances = _extract_utterances(results.get("ch1", {}), "ch1")

    # Determine which channel is the Agent.
    # Use the same rule as mono: the Agent is whoever makes the FIRST
    # substantial utterance (3+ words).  A single "Hi." from the customer
    # must not flip the roles — the agent greeting is always multi-word.
    if agent_hint in ("ch0", "ch1"):
        agent_ch = agent_hint
    else:
        all_sorted = sorted(ch0_utterances + ch1_utterances, key=lambda u: u["start"])
        agent_ch = None
        for u in all_sorted:
            if len(u.get("text", "").split()) >= 3:
                agent_ch = u["speaker"]  # "ch0" or "ch1"
                break
        # Fallback: truly first utterance if no 3-word utterance found
        if agent_ch is None:
            ch0_first = min((u["start"] for u in ch0_utterances), default=float("inf"))
            ch1_first = min((u["start"] for u in ch1_utterances), default=float("inf"))
            agent_ch = "ch0" if ch0_first <= ch1_first else "ch1"

    role_map = {agent_ch: "Agent", ("ch1" if agent_ch == "ch0" else "ch0"): "Customer"}

    all_utterances = ch0_utterances + ch1_utterances
    all_utterances.sort(key=lambda u: u["start"])

    turns = [
        {
            "role": role_map[u["speaker"]],
            "start": round(float(u["start"]), 2),
            "end": round(float(u["end"]), 2),
            "text": u["text"],
        }
        for u in all_utterances
    ]

    # Consolidate consecutive same-role turns
    merged_turns = []
    for turn in turns:
        if merged_turns and merged_turns[-1]["role"] == turn["role"]:
            prev = merged_turns[-1]
            prev["end"] = max(prev["end"], turn["end"])
            if not prev["text"].endswith(" "):
                prev["text"] += " "
            prev["text"] += turn["text"]
        else:
            merged_turns.append(turn)

    duration = max((t["end"] for t in merged_turns), default=0.0)
    return {
        "call_id": os.path.basename(audio_path),
        "duration_sec": round(duration, 2),
        "utterances": merged_turns,
    }


def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    agent_hint: str | None = None
):
    if device == "mps":
        device = "cpu"

    is_stereo, estimated_speakers = _detect_audio_properties(audio_path)

    if is_stereo:
        # Stereo recordings have one speaker per channel — transcribe each
        # channel independently for perfect speaker separation.
        try:
            return transcribe_stereo_channels(
                audio_path=audio_path,
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                agent_hint=agent_hint,
            )
        except Exception:
            # Fall through to mono+diarization if stereo transcription fails
            pass

    num_speakers = estimated_speakers if estimated_speakers is not None else 2
    return transcribe_mono_with_diarization(
        audio_path=audio_path,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        num_speakers=num_speakers,
        agent_hint=agent_hint
    )