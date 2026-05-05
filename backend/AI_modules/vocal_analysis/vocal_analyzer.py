"""
Vocal stress and prosody analyzer.

Computes energy, zero-crossing rate (ZCR), spectral centroid, and speech rate
per utterance from the raw audio waveform. Derives a per-utterance stress score
using within-call z-score normalization so scores are relative to the call's own
baseline — not fixed thresholds that break on different microphone gains.

No model loading. No external API calls. Pure signal processing.
Dependencies: soundfile, scipy, numpy — all already in requirements.txt.
"""

import sys
from typing import Optional

import numpy as np
import scipy.signal as sp_signal
import soundfile as sf


def _load_audio_mono(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Load audio as mono float32 at its native sample rate.
    If stereo, average both channels — we want aggregate stress, not per-channel.
    Returns (waveform_1d_float32, sample_rate).
    """
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _extract_segment_features(
    segment: np.ndarray, sr: int, text: str
) -> Optional[dict]:
    """
    Compute four acoustic features for a single utterance segment.

      energy_db         — RMS energy in dB (louder / more forceful speech = higher)
      zcr               — zero crossing rate per sample (high = fricatives, tension)
      spectral_centroid — power-weighted mean frequency in Hz (higher = brighter/sharper voice)
      speech_rate_wpm   — words per minute derived from transcript text and segment duration

    Returns None if the segment is shorter than 80 ms (not enough signal).
    """
    MIN_SAMPLES = int(sr * 0.08)
    if len(segment) < MIN_SAMPLES:
        return None

    # RMS energy → dB
    rms = float(np.sqrt(np.mean(segment ** 2)))
    energy_db = float(20.0 * np.log10(rms + 1e-10))

    # Zero crossing rate
    signs = np.sign(segment)
    zcr = float(np.sum(np.diff(signs) != 0) / len(segment))

    # Spectral centroid via scipy.signal.spectrogram
    nperseg = min(512, len(segment))
    freqs, _, Sxx = sp_signal.spectrogram(segment, fs=sr, nperseg=nperseg)
    power = Sxx.sum(axis=1)        # collapse time axis → shape (n_freqs,)
    total_power = float(power.sum())
    if total_power > 0:
        spectral_centroid = float(np.dot(freqs, power) / total_power)
    else:
        spectral_centroid = 0.0

    # Speech rate from transcript text
    duration_s = len(segment) / sr
    word_count = len(text.split()) if text.strip() else 0
    speech_rate_wpm = float(word_count / duration_s * 60.0) if duration_s > 0 else 0.0

    return {
        "energy_db":            round(energy_db, 2),
        "zcr":                  round(zcr, 6),
        "spectral_centroid_hz": round(spectral_centroid, 1),
        "speech_rate_wpm":      round(speech_rate_wpm, 1),
    }


def _normalize_within_call(values: list[float]) -> list[float]:
    """
    Z-score normalize a list of floats using the list's own mean and std.
    Returns all zeros when std ≈ 0 (all values identical — no variation to score).
    """
    arr = np.array(values, dtype=float)
    mean, std = float(arr.mean()), float(arr.std())
    if std < 1e-9:
        return [0.0] * len(values)
    return ((arr - mean) / std).tolist()


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _compute_stress_scores(
    raw_features: list[Optional[dict]],
) -> list[float]:
    """
    Derive a stress score in [0, 1] per utterance.

    Each of the four features is z-score normalized across all utterances in the
    call (separately), then combined with fixed weights and mapped through a
    sigmoid to produce a score that sits around 0.5 for typical utterances and
    rises toward 1.0 for outlier-stress moments.

    Weights:
      energy (35%) + ZCR (30%) + spectral_centroid (20%) + speech_rate (15%)

    Utterances where feature extraction returned None (too short) keep score 0.0.
    """
    valid_indices = [i for i, f in enumerate(raw_features) if f is not None]
    if not valid_indices:
        return [0.0] * len(raw_features)

    energy_vals = [raw_features[i]["energy_db"]           for i in valid_indices]
    zcr_vals    = [raw_features[i]["zcr"]                  for i in valid_indices]
    sc_vals     = [raw_features[i]["spectral_centroid_hz"] for i in valid_indices]
    wpm_vals    = [raw_features[i]["speech_rate_wpm"]      for i in valid_indices]

    energy_norm = _normalize_within_call(energy_vals)
    zcr_norm    = _normalize_within_call(zcr_vals)
    sc_norm     = _normalize_within_call(sc_vals)
    wpm_norm    = _normalize_within_call(wpm_vals)

    scores_for_valid = [
        float(_sigmoid(0.35 * e + 0.30 * z + 0.20 * s + 0.15 * w))
        for e, z, s, w in zip(energy_norm, zcr_norm, sc_norm, wpm_norm)
    ]

    result = [0.0] * len(raw_features)
    for idx, score in zip(valid_indices, scores_for_valid):
        result[idx] = round(score, 4)
    return result


def _empty_result(reason: str) -> dict:
    return {
        "utterance_stress":            [],
        "agent_stress_avg":            0.0,
        "customer_stress_avg":         0.0,
        "peak_stress":                 None,
        "stress_escalation_detected":  False,
        "calm_recovery_detected":      False,
        "analysis_error":              reason,
    }


def analyze_vocal_stress(audio_path: str, transcript: dict) -> dict:
    """
    Main entry point. Called from the orchestrator alongside emotion/behavioral analysis.

    Args:
        audio_path: Path to the audio file (same path passed to transcribe_audio).
        transcript:  WhisperX transcript dict — must have the shape produced by
                     whisperer.py:
                     {
                       "utterances": [
                         {"role": "Agent"|"Customer", "start": float, "end": float, "text": str},
                         ...
                       ]
                     }

    Returns:
        {
          "utterance_stress": [
            {
              "role": str, "start": float, "end": float,
              "energy_db": float, "zcr": float,
              "spectral_centroid_hz": float, "speech_rate_wpm": float,
              "stress_score": float   # 0-1, call-relative
            },
            ...
          ],
          "agent_stress_avg":           float,
          "customer_stress_avg":        float,
          "peak_stress":                {"role": str, "start": float, "end": float,
                                         "stress_score": float} | None,
          "stress_escalation_detected": bool,
          "calm_recovery_detected":     bool,
          "analysis_error":             str | None
        }
    """
    utterances = transcript.get("utterances", [])
    if not utterances:
        return _empty_result("No utterances in transcript")

    try:
        waveform, sr = _load_audio_mono(audio_path)
    except Exception as exc:
        print(
            f"[VocalAnalyzer] Could not load audio '{audio_path}': {exc}",
            file=sys.stderr,
            flush=True,
        )
        return _empty_result(str(exc))

    # --- Per-utterance feature extraction -----------------------------------
    raw_features: list[Optional[dict]] = []
    for utt in utterances:
        start_i = int(utt.get("start", 0.0) * sr)
        end_i   = int(utt.get("end",   0.0) * sr)
        # Guard against timestamps that exceed waveform length
        end_i   = min(end_i, len(waveform))
        segment = waveform[start_i:end_i]
        raw_features.append(
            _extract_segment_features(segment, sr, utt.get("text", ""))
        )

    # --- Stress scores (call-relative z-score normalization) ----------------
    stress_scores = _compute_stress_scores(raw_features)

    # --- Build per-utterance output -----------------------------------------
    utterance_stress: list[dict] = []
    for utt, raw, score in zip(utterances, raw_features, stress_scores):
        entry: dict = {
            "role":         utt["role"],
            "start":        utt["start"],
            "end":          utt["end"],
            "stress_score": score,
        }
        if raw is not None:
            entry.update(raw)
        utterance_stress.append(entry)

    # --- Aggregate stats by role --------------------------------------------
    agent_scores    = [e["stress_score"] for e in utterance_stress
                       if e["role"] == "Agent"]
    customer_scores = [e["stress_score"] for e in utterance_stress
                       if e["role"] == "Customer"]

    agent_stress_avg    = round(float(np.mean(agent_scores)),    4) if agent_scores    else 0.0
    customer_stress_avg = round(float(np.mean(customer_scores)), 4) if customer_scores else 0.0

    # --- Peak stress moment --------------------------------------------------
    if utterance_stress:
        peak_entry = max(utterance_stress, key=lambda e: e["stress_score"])
        peak_stress: Optional[dict] = {
            "role":         peak_entry["role"],
            "start":        peak_entry["start"],
            "end":          peak_entry["end"],
            "stress_score": peak_entry["stress_score"],
        }
    else:
        peak_stress = None

    # --- Stress escalation: last-third customer avg > first-third customer avg
    stress_escalation_detected = False
    if len(customer_scores) >= 4:
        third = max(1, len(customer_scores) // 3)
        early = float(np.mean(customer_scores[:third]))
        late  = float(np.mean(customer_scores[-third:]))
        stress_escalation_detected = bool(late > early + 0.1)

    # --- Calm recovery: post-peak customer stress drops ≥ 0.25 --------------
    calm_recovery_detected = False
    if peak_stress is not None and customer_scores:
        peak_idx = next(
            (
                i for i, e in enumerate(utterance_stress)
                if e["role"] == "Customer" and e["start"] == peak_stress["start"]
            ),
            None,
        )
        if peak_idx is not None:
            post_peak_scores = [
                e["stress_score"]
                for e in utterance_stress[peak_idx + 1:]
                if e["role"] == "Customer"
            ]
            if post_peak_scores:
                calm_recovery_detected = bool(
                    float(np.mean(post_peak_scores))
                    <= peak_stress["stress_score"] - 0.25
                )

    print(
        f"[VocalAnalyzer] Done — {len(utterance_stress)} utterances, "
        f"agent_avg={agent_stress_avg:.3f}, customer_avg={customer_stress_avg:.3f}, "
        f"escalation={stress_escalation_detected}",
        file=sys.stderr,
        flush=True,
    )

    return {
        "utterance_stress":            utterance_stress,
        "agent_stress_avg":            agent_stress_avg,
        "customer_stress_avg":         customer_stress_avg,
        "peak_stress":                 peak_stress,
        "stress_escalation_detected":  stress_escalation_detected,
        "calm_recovery_detected":      calm_recovery_detected,
        "analysis_error":              None,
    }
