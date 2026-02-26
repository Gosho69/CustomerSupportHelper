import os, json, sys
import warnings
import soundfile as sf
import dotenv

warnings.filterwarnings("ignore", category=UserWarning, message=".*torchaudio.*deprecated.*")

# ---------------------------------------------------------------------------
# Agent-detection helpers
# ---------------------------------------------------------------------------

# Phrases that are strongly diagnostic of the *agent* speaking.  These are
# the standard call-centre greetings that appear at the start of a call.
_AGENT_GREETING_PHRASES = [
    "thank you for calling",
    "thanks for calling",
    "thank you for contacting",
    "thanks for contacting",
    "thank you for reaching",
    "thanks for reaching",
    "how may i help you",
    "how can i help you",
    "how may i assist you",
    "how can i assist you",
    "how can i be of assistance",
    "how may i be of assistance",
    "you've reached",
    "you have reached",
    "welcome to",
    "speaking, how",        # "this is [name] speaking, how can I…"
    "my name is",
    "good morning",
    "good afternoon",
    "good evening",
]

_ANNOUNCEMENT_KWS = frozenset({"record", "recorded", "recording",
                                "monitor", "monitored", "monitoring"})


def _detect_agent_by_greeting(utterances: list) -> str | None:
    """
    Return the speaker ID of the first utterance that contains one of the
    canonical agent greeting phrases.  Returns None if no greeting is found.
    Examines the first 8 utterances maximum (the greeting is always near the
    start of the call).
    """
    for u in sorted(utterances, key=lambda x: x["start"])[:8]:
        text_lower = u.get("text", "").lower()
        for phrase in _AGENT_GREETING_PHRASES:
            if phrase in text_lower:
                return u["speaker"]
    return None


def _detect_agent_by_word_count(utterances: list) -> str | None:
    """
    In a customer-support call the agent typically speaks more words in total
    than the customer.  Use this as a tie-breaker / last-resort heuristic.
    """
    word_counts: dict[str, int] = {}
    for u in utterances:
        spk = u["speaker"]
        word_counts[spk] = word_counts.get(spk, 0) + len(u.get("text", "").split())
    if not word_counts:
        return None
    return max(word_counts, key=word_counts.get)


def _pick_agent_speaker(utterances: list, agent_hint: str | None = None) -> str:
    """
    Unified agent-speaker selection used by both mono-diarization and stereo
    transcription paths.

    Priority order
    1. Explicit hint (already confirmed by caller to be the agent channel/ID).
    2. First utterance whose text matches a known agent greeting phrase.
    3. First utterance with 3+ words that is NOT a recording announcement.
    4. Speaker with the most total words (agents talk more than customers).
    5. Earliest speaker (last-resort absolute fallback).
    """
    if agent_hint and any(u["speaker"] == agent_hint for u in utterances):
        return agent_hint

    # 2 — greeting phrase detection
    by_greeting = _detect_agent_by_greeting(utterances)
    if by_greeting is not None:
        return by_greeting

    # 3 — first substantial non-announcement utterance
    for u in sorted(utterances, key=lambda x: x["start"]):
        words = [w.strip(".,!?;:\"'") for w in u.get("text", "").lower().split()]
        if _ANNOUNCEMENT_KWS.intersection(words):
            continue
        if len(words) >= 3:
            return u["speaker"]

    # 4 — most words
    by_words = _detect_agent_by_word_count(utterances)
    if by_words is not None:
        return by_words

    # 5 — absolute fallback
    if utterances:
        return min(utterances, key=lambda u: u["start"])["speaker"]
    return "SPEAKER_00"

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Module-level caches — models are loaded once per gunicorn worker process and
# reused across all requests.  This eliminates the 20–30 s per-call overhead
# and prevents memory pressure that causes pyannote to fail silently.
_whisper_model_cache: dict = {}
_pyannote_pipeline_cache = None

if not HF_TOKEN:
    print(
        "[Whisperer] CRITICAL: HF_TOKEN is not set — "
        "PyAnnote speaker diarization is DISABLED. "
        "All transcripts will use the low-quality silence-gap fallback. "
        "Set HF_TOKEN in /opt/agentsights/.env and restart the container.",
        file=sys.stderr, flush=True
    )
elif not HF_TOKEN.strip('"').strip("'").startswith("hf_"):
    print(
        f"[Whisperer] CRITICAL: HF_TOKEN value does not start with 'hf_' — "
        "PyAnnote authentication will fail.",
        file=sys.stderr, flush=True
    )
else:
    print(
        f"[Whisperer] HF_TOKEN loaded (hf_...{HF_TOKEN.strip(chr(34)).strip(chr(39))[-4:]})",
        file=sys.stderr, flush=True
    )


def _get_whisper_model(model_size: str, device: str, compute_type: str):
    """Return a cached WhisperX ASR model, loading it on first use."""
    import whisperx
    key = (model_size, device, compute_type)
    if key not in _whisper_model_cache:
        print(f"[Whisperer] Loading WhisperX '{model_size}' model (first use) …",
              file=sys.stderr, flush=True)
        _whisper_model_cache[key] = whisperx.load_model(model_size, device, compute_type=compute_type)
        print(f"[Whisperer] ✓ WhisperX '{model_size}' model cached",
              file=sys.stderr, flush=True)
    return _whisper_model_cache[key]


def _get_pyannote_pipeline(token: str):
    """Return a cached pyannote diarization pipeline, loading it on first use."""
    global _pyannote_pipeline_cache
    if _pyannote_pipeline_cache is None:
        from pyannote.audio import Pipeline
        print("[Whisperer] Loading pyannote/speaker-diarization-3.1 (first use) …",
              file=sys.stderr, flush=True)
        try:
            _pyannote_pipeline_cache = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
        except TypeError:
            _pyannote_pipeline_cache = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token,
            )
        print("[Whisperer] ✓ PyAnnote pipeline cached",
              file=sys.stderr, flush=True)
    return _pyannote_pipeline_cache


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

    if device not in ["cpu", "cuda"]:
        device = "cpu"

    asr_model = _get_whisper_model(model_size, device, compute_type)
    batch_size = 16 if device != "cpu" else 4
    asr_result = asr_model.transcribe(audio_path, batch_size=batch_size)
    audio = whisperx.load_audio(audio_path)
    align_model, metadata = whisperx.load_align_model(asr_result["language"], device)
    aligned = whisperx.align(asr_result["segments"], align_model, metadata, audio, device)
    
    hf_token = HF_TOKEN
    clean_token = hf_token.strip('"').strip("'") if hf_token else None
    
    diarized = None
    diarization_failed = False

    # Pass the already-loaded waveform to pyannote as a dict instead of a file
    # path. This bypasses pyannote's internal file-loading code which uses
    # torchcodec.AudioDecoder — unavailable on CPU-only PyTorch builds.
    import torch
    waveform = torch.from_numpy(audio).float().unsqueeze(0)  # (1, samples)
    pyannote_input = {"waveform": waveform, "sample_rate": 16000}

    try:
        if not clean_token:
            raise RuntimeError("HF_TOKEN is not set — cannot load pyannote model")

        diarize_pipeline = _get_pyannote_pipeline(clean_token)

        print("[Whisperer] Running speaker diarization …",
              file=sys.stderr, flush=True)

        if num_speakers is not None:
            diarize_result = diarize_pipeline(pyannote_input, num_speakers=num_speakers)
        else:
            diarize_result = diarize_pipeline(pyannote_input)
        diarize_segments = _convert_pyannote_to_whisperx(diarize_result)
        diarized = _assign_speakers_to_words(diarize_segments, aligned)

        print("[Whisperer] ✓ PyAnnote diarization complete",
              file=sys.stderr, flush=True)

    except Exception as e:
        import traceback
        print(
            f"[Whisperer] DIARIZATION FAILED — falling back to low-quality synthetic method.\n"
            f"  Error type : {type(e).__name__}\n"
            f"  Error      : {e}\n"
            f"  Traceback  :\n{traceback.format_exc()}\n"
            f"\n"
            f"  ACTIONS TO FIX:\n"
            f"  1. Ensure HF_TOKEN is set in /opt/agentsights/.env (starts with hf_)\n"
            f"  2. Accept model licence at: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            f"  3. Also accept: https://huggingface.co/pyannote/segmentation-3.0\n"
            f"  The HuggingFace account the token belongs to MUST be the account\n"
            f"  that accepted the licence.",
            file=sys.stderr, flush=True
        )
        diarization_failed = True

    diarization_method = "pyannote"
    if diarization_failed or diarized is None:
        print("[Whisperer] CRITICAL WARNING: Falling back to synthetic diarization "
              "(silence-gap method) — transcript speaker labels will be WRONG. "
              "See error above for how to fix PyAnnote authentication.",
              file=sys.stderr, flush=True)
        diarized = _create_synthetic_diarization(aligned)
        diarization_method = "synthetic_fallback"

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
    agent_speaker = _pick_agent_speaker(utterances, agent_hint)

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
        "utterances": turns,
        "diarization_method": diarization_method,
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
            asr_model = _get_whisper_model(model_size, device, compute_type)
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

    # Determine which channel is the Agent using the unified multi-heuristic
    # picker (greeting phrases → first 3+ word non-announcement → most words).
    all_channel_utterances = ch0_utterances + ch1_utterances
    agent_ch = _pick_agent_speaker(all_channel_utterances, agent_hint if agent_hint in ("ch0", "ch1") else None)

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
        "diarization_method": "stereo_channels",
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

    print(f"[Whisperer] transcribe_audio called: model={model_size}, device={device}, "
          f"compute={compute_type}, file={os.path.basename(audio_path)}",
          file=sys.stderr, flush=True)

    is_stereo, estimated_speakers = _detect_audio_properties(audio_path)

    if is_stereo:
        print("[Whisperer] Stereo audio detected — using per-channel transcription",
              file=sys.stderr, flush=True)
        # Stereo recordings have one speaker per channel — transcribe each
        # channel independently for perfect speaker separation.
        try:
            result = transcribe_stereo_channels(
                audio_path=audio_path,
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                agent_hint=agent_hint,
            )
            print(f"[Whisperer] ✓ Stereo transcription complete: "
                  f"{len(result.get('utterances', []))} utterances, "
                  f"method={result.get('diarization_method')}",
                  file=sys.stderr, flush=True)
            return result
        except Exception as exc:
            print(f"[Whisperer] Stereo transcription failed ({type(exc).__name__}: {exc}) — "
                  f"falling through to mono+diarization",
                  file=sys.stderr, flush=True)

    print("[Whisperer] Mono audio — using pyannote diarization path",
          file=sys.stderr, flush=True)

    num_speakers = estimated_speakers if estimated_speakers is not None else 2
    result = transcribe_mono_with_diarization(
        audio_path=audio_path,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        num_speakers=num_speakers,
        agent_hint=agent_hint
    )
    print(f"[Whisperer] ✓ Mono transcription complete: "
          f"{len(result.get('utterances', []))} utterances, "
          f"method={result.get('diarization_method')}",
          file=sys.stderr, flush=True)
    return result