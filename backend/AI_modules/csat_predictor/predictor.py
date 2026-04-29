"""
CSAT (Customer Satisfaction) predictor.

Uses a GradientBoostingClassifier trained on historical call data to predict
customer satisfaction (1–5 scale) from 10 features already computed by the
existing pipeline. Falls back to a deterministic rule-based formula when no
trained model exists (< 30 training examples).

Model persistence: joblib files stored alongside this module.
"""

import json
import os
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
CSAT_MODEL_PATH    = os.path.join(_HERE, "csat_model.pkl")
CSAT_METADATA_PATH = os.path.join(_HERE, "csat_metadata.json")

MIN_TRAINING_SAMPLES = 30
RETRAIN_EVERY_N_CALLS = 10   # Trigger incremental retrain after this many new calls

# ── Label encoding ──────────────────────────────────────────────────────────

SATISFACTION_TO_SCORE = {
    "very_satisfied":    5,
    "satisfied":         4,
    "neutral":           3,
    "dissatisfied":      2,
    "very_dissatisfied": 1,
}

SCORE_TO_LABEL = {v: k for k, v in SATISFACTION_TO_SCORE.items()}

RESOLUTION_ENCODE = {"resolved": 2, "pending": 1, "unresolved": 0}

TRAJECTORY_ENCODE = {
    "positive_throughout": 3,
    "resolved":            2,
    "improving":           2,
    "stable_positive":     2,
    "stable_negative":     1,
    "escalated":           0,
    "negative_throughout": 0,
    "deteriorating":       0,
}


# ── Feature extraction ──────────────────────────────────────────────────────

def extract_features(call) -> Optional[np.ndarray]:
    """
    Extract a 10-element float32 feature vector from a completed Call object.

    Feature order (fixed — never reorder, model trained against this):
      0  agent_empathy_score
      1  customer_frustration_level
      2  resolution_status_encoded     (0=unresolved, 1=pending, 2=resolved)
      3  emotional_trajectory_encoded  (0=deteriorating … 3=positive_throughout)
      4  behavioral_score
      5  agent_interruptions
      6  talk_ratio                    (agent_words / customer_words)
      7  silence_percentage
      8  avg_agent_response_time       (seconds)
      9  acknowledgment_rate           (0–1)

    Returns None if essential data is missing.
    """
    es = call.emotional_summary   or {}
    ba = call.behavioral_analysis or {}

    if not es:
        return None

    empathy         = float(es.get("agent_empathy_score",         0.3))
    frustration     = float(es.get("customer_frustration_level",  0.5))
    resolution_raw  = es.get("resolution_status", "pending")
    resolution_enc  = float(RESOLUTION_ENCODE.get(resolution_raw, 1))

    trajectory_raw  = (es.get("emotional_journey") or {}).get("trajectory", "")
    trajectory_enc  = float(TRAJECTORY_ENCODE.get(trajectory_raw, 1))

    beh_score     = float(ba.get("behavioral_score") or 50.0)
    interrupts    = float((ba.get("interruption_analysis") or {}).get("agent_interruptions", 0))
    talk_ratio    = float((ba.get("words_per_minute") or {}).get("talk_ratio", 1.0))
    silence_pct   = float((ba.get("silence_analysis") or {}).get("silence_percentage", 0.0))
    avg_resp_time = float((ba.get("response_time_analysis") or {}).get("avg_agent_response_time", 1.5))
    ack_rate      = float((ba.get("active_listening") or {}).get("acknowledgment_rate", 0.0))

    return np.array([
        empathy, frustration, resolution_enc, trajectory_enc,
        beh_score, interrupts, talk_ratio, silence_pct,
        avg_resp_time, ack_rate,
    ], dtype=np.float32)


def extract_label(call) -> Optional[int]:
    """Extract the 1–5 training label from a completed Call. Returns None if unavailable."""
    es = call.emotional_summary or {}
    return SATISFACTION_TO_SCORE.get(es.get("customer_satisfaction"))


# ── Rule-based fallback ─────────────────────────────────────────────────────

def _rule_based_predict(features: np.ndarray) -> tuple[float, str]:
    """
    Deterministic prediction used when no trained model exists.
    Returns (score_float_1–5, label_string).
    """
    score = 3.0
    score += (features[0] - 0.5) * 2.0        # empathy:       0→-1, 1→+1
    score -= features[1] * 2.5                 # frustration:   1.0→-2.5
    score += (features[2] - 1.0) * 0.6         # resolution:    resolved→+0.6
    score += (features[3] - 1.0) * 0.4         # trajectory:    improving→+0.4
    score -= min(features[5], 6.0) * 0.1       # interruptions: -0.1 each, max -0.6
    score  = float(np.clip(score, 1.0, 5.0))
    return round(score, 1), SCORE_TO_LABEL.get(round(score), "neutral")


# ── Model load / save ───────────────────────────────────────────────────────

def _load_model():
    """Returns (model, metadata_dict) or (None, None) if no model saved."""
    if not os.path.exists(CSAT_MODEL_PATH):
        return None, None
    try:
        import joblib
        model = joblib.load(CSAT_MODEL_PATH)
        meta  = {}
        if os.path.exists(CSAT_METADATA_PATH):
            with open(CSAT_METADATA_PATH) as f:
                meta = json.load(f)
        return model, meta
    except Exception as exc:
        logger.warning(f"[CSATPredictor] Could not load model: {exc}")
        return None, None


def _save_model(model, metadata: dict) -> None:
    import joblib
    joblib.dump(model, CSAT_MODEL_PATH)
    with open(CSAT_METADATA_PATH, "w") as f:
        json.dump(metadata, f)
    logger.info(f"[CSATPredictor] Model saved — metadata={metadata}")


# ── Public API ──────────────────────────────────────────────────────────────

def should_auto_retrain(current_call_count: int) -> bool:
    """
    Returns True when enough new calls have accumulated since the last training
    run to warrant kicking off an incremental retrain.

    Rules:
      - Never triggers below MIN_TRAINING_SAMPLES (rule-based fallback is used there).
      - After the threshold is reached, retrain fires every RETRAIN_EVERY_N_CALLS
        new calls, measured against `last_trained_at_count` stored in metadata.
    """
    if current_call_count < MIN_TRAINING_SAMPLES:
        return False
    _, meta = _load_model()
    last_count = (meta or {}).get("last_trained_at_count", 0)
    return (current_call_count - last_count) >= RETRAIN_EVERY_N_CALLS


def predict(call) -> tuple[float, str]:
    """
    Predict CSAT for a single Call object.
    Returns (predicted_score_1.0–5.0, predicted_label).
    Falls back to rule-based if no trained model exists.
    """
    features = extract_features(call)
    if features is None:
        return 3.0, "neutral"

    model, _ = _load_model()
    if model is None:
        return _rule_based_predict(features)

    try:
        proba   = model.predict_proba(features.reshape(1, -1))[0]
        classes = model.classes_
        score   = float(np.dot(classes, proba))
        score   = round(float(np.clip(score, 1.0, 5.0)), 1)
        return score, SCORE_TO_LABEL.get(round(score), "neutral")
    except Exception as exc:
        logger.warning(f"[CSATPredictor] Prediction failed, using rule-based: {exc}")
        return _rule_based_predict(features)


def train(calls, total_call_count: int = None) -> dict:
    """
    Train a new GradientBoostingClassifier on the provided Call queryset/list.
    Returns metadata dict or raises ValueError if < MIN_TRAINING_SAMPLES labeled calls.

    `total_call_count` — pass the current count of eligible calls so that
    should_auto_retrain() can correctly calculate how many new calls have
    arrived since this training run. Defaults to len(labeled calls) if omitted.

    Called from the Celery retrain task only — never from the request cycle.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    X, y = [], []
    for call in calls:
        features = extract_features(call)
        label    = extract_label(call)
        if features is not None and label is not None:
            X.append(features)
            y.append(label)

    if len(X) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Only {len(X)} labeled calls — need {MIN_TRAINING_SAMPLES} to train."
        )

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)

    clf = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
    )

    cv_scores   = cross_val_score(clf, X_arr, y_arr, cv=5, scoring="accuracy")
    cv_accuracy = float(cv_scores.mean())

    clf.fit(X_arr, y_arr)

    feature_names = [
        "agent_empathy_score", "customer_frustration_level",
        "resolution_status_encoded", "emotional_trajectory_encoded",
        "behavioral_score", "agent_interruptions", "talk_ratio",
        "silence_percentage", "avg_agent_response_time", "acknowledgment_rate",
    ]
    importances = dict(zip(feature_names, clf.feature_importances_.tolist()))

    _, old_meta = _load_model()
    old_accuracy = (old_meta or {}).get("cv_accuracy", 0.0)

    metadata = {
        "n_training_samples":    len(X),
        "cv_accuracy":           round(cv_accuracy, 4),
        "feature_importances":   importances,
        "previous_accuracy":     old_accuracy,
        "improved":              cv_accuracy >= old_accuracy - 0.02,
        # Track how many calls existed when this model was trained so
        # should_auto_retrain() can compute new-calls-since-last-train.
        "last_trained_at_count": total_call_count if total_call_count is not None else len(X),
    }

    # Only replace existing model if new one is not significantly worse (allow 2% regression)
    if cv_accuracy >= old_accuracy - 0.02:
        _save_model(clf, metadata)
        logger.info(
            f"[CSATPredictor] Retrained. CV accuracy: {cv_accuracy:.4f} (was {old_accuracy:.4f})"
        )
    else:
        logger.warning(
            f"[CSATPredictor] New accuracy {cv_accuracy:.4f} significantly worse than "
            f"{old_accuracy:.4f} — keeping old model."
        )

    return metadata
