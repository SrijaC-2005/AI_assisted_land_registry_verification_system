"""Explain the captured consensus-selector decision with LIME.

Rules this module enforces:
1. The selector and LIME use the same model, scaler, label encoder and the
   exact feature vector captured when the transaction was queued.
2. LIME re-runs the model only to verify it reproduces the captured class.
   If it does not, the explanation is flagged instead of silently explaining
   a different class.
3. Training-range checking lives here (check_training_range) so app.py and
   the explanation use one definition.
4. LIME weights are local surrogate weights. They are not probabilities and
   do not prove a protocol is optimal.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

SUPPORTED = ("PoW", "PoS", "PBFT", "Raft", "HotStuff")
FEATURES = [
    "node_count",
    "network_latency_ms",
    "tx_throughput_tps",
    "security_risk_score",
    "fault_tolerance_requirement",
]
FEATURE_LABELS = [
    "Registered node count",
    "In-process consensus round latency (ms; proxy, not P2P RTT)",
    "Committed transactions per second (application)",
    "Transaction GNN fraud-risk score (0-1)",
    "Configured fault-tolerance requirement",
]
LABEL_BY_FEATURE = dict(zip(FEATURES, FEATURE_LABELS))


# --------------------------------------------------------------- ranges
def compute_training_ranges(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    ranges: Dict[str, Dict[str, float]] = {}
    for feature in FEATURES:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").dropna()
        if values.empty:
            continue
        ranges[feature] = {"min": float(values.min()), "max": float(values.max())}
    return ranges


def load_training_ranges(evaluation_path: str, fallback_frame: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """Prefer ranges saved by train_consensus_selector.py; else compute from the background frame."""
    try:
        if os.path.exists(evaluation_path):
            with open(evaluation_path, "r", encoding="utf-8") as handle:
                saved = json.load(handle).get("feature_ranges")
            if saved and all(f in saved for f in FEATURES):
                return {f: {"min": float(saved[f]["min"]), "max": float(saved[f]["max"])} for f in FEATURES}
    except Exception:
        pass
    return compute_training_ranges(fallback_frame) if fallback_frame is not None else {}


def check_training_range(features: Dict[str, Any], ranges: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """Return one entry per feature whose runtime value lies outside the training range."""
    out: List[Dict[str, Any]] = []
    for feature in FEATURES:
        if feature not in ranges or features.get(feature) is None:
            continue
        value = float(features[feature])
        low, high = ranges[feature]["min"], ranges[feature]["max"]
        if value < low or value > high:
            out.append({
                "feature": feature,
                "label": LABEL_BY_FEATURE[feature],
                "value": value,
                "training_min": low,
                "training_max": high,
                "direction": "below" if value < low else "above",
            })
    return out


# --------------------------------------------------------------- model helpers
def _probabilities(model, scaler, rows) -> np.ndarray:
    scaled = scaler.transform(np.asarray(rows, dtype=float))
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(scaled), dtype=float)
    scores = np.asarray(model.decision_function(scaled), dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def predict_consensus(model, scaler, encoder, features: Dict[str, Any]) -> Dict[str, Any]:
    """Single prediction path shared by app.py and the explainer."""
    vector = np.asarray([[float(features[name]) for name in FEATURES]], dtype=float)
    code = int(model.predict(scaler.transform(vector))[0])
    probs = _probabilities(model, scaler, vector)[0]
    classes = [str(encoder.inverse_transform([int(c)])[0]) for c in model.classes_]
    return {
        "predicted_consensus": str(encoder.inverse_transform([code])[0]),
        "class_probabilities": {name: round(float(p), 5) for name, p in zip(classes, probs)},
    }


def _local_lime(model, scaler, encoder, background: pd.DataFrame, current: np.ndarray, expected: Optional[str]):
    missing_bg = [name for name in FEATURES if name not in background.columns]
    if missing_bg:
        raise ValueError(f"LIME background is missing columns: {missing_bg}")
    bg = (
        background[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(bg) < 10:
        raise ValueError("Need at least 10 complete background rows for LIME.")

    class_codes = list(model.classes_)
    class_names = [str(encoder.inverse_transform([int(code)])[0]) for code in class_codes]
    pred_code = int(model.predict(scaler.transform(current.reshape(1, -1)))[0])
    predicted_name = str(encoder.inverse_transform([pred_code])[0])

    # Explain the captured class. If the model now disagrees, say so.
    target_name = expected if expected in class_names else predicted_name
    class_index = class_names.index(target_name)

    explainer = LimeTabularExplainer(
        training_data=bg.to_numpy(dtype=float),
        feature_names=FEATURE_LABELS,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    exp = explainer.explain_instance(
        current,
        lambda rows: _probabilities(model, scaler, rows),
        labels=[class_index],
        num_features=len(FEATURES),
        num_samples=1800,
    )
    weights = exp.as_list(label=class_index)
    return predicted_name, {
        "method": "LIME (local surrogate) on the same trained consensus-selection classifier",
        "target_consensus": target_name,
        "model_reproduced_prediction": predicted_name,
        "matches_captured_prediction": expected is None or predicted_name == expected,
        "scope": (
            "Local approximation around the captured runtime input. "
            "Background rows are the selector's training data, used only as LIME's sampling distribution."
        ),
        "features": [
            {
                "feature": name,
                "weight": round(float(weight), 5),
                "direction": (
                    "supports this class in the local surrogate"
                    if weight > 0 else "opposes this class in the local surrogate"
                ),
            }
            for name, weight in weights
        ],
        "warning": (
            "LIME weights describe local influence in a surrogate model. They are not probabilities, "
            "causal effects, or evidence that the protocol is optimal. When inputs are outside the "
            "training range, both the prediction and this explanation are extrapolations."
        ),
    }


# --------------------------------------------------------------- public API
def _error(message: str, current_mode: str, sources: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "module": "Module 4 — Explainable Dynamic Consensus",
        "error": message,
        "selected_consensus": None,
        "current_consensus": current_mode,
        "input_sources": sources or {},
        "lime_explanation": {"method": "LIME not run", "features": [], "warning": message},
    }


def explain_consensus(
    risk_assessment: Optional[Dict[str, Any]] = None,
    transaction: Optional[Dict[str, Any]] = None,
    network_metrics: Optional[Dict[str, Any]] = None,
    active_nodes: int = 1,
    current_mode: str = "PoW",
    model: Any = None,
    scaler: Any = None,
    label_encoder: Any = None,
    training_frame: Any = None,
    current_features: Optional[Dict[str, Any]] = None,
    expected_consensus: Optional[str] = None,
    training_ranges: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    risk = risk_assessment or {}
    tx = transaction or {}
    metrics = network_metrics or {}
    sources = dict(metrics.get("input_sources") or {})
    source_types = dict(metrics.get("source_types") or {})
    measurements = dict(metrics.get("measurements") or {})
    limitations = list(metrics.get("limitations") or [])

    if model is None or scaler is None or label_encoder is None or training_frame is None:
        return _error("Trained selector is unavailable. No fallback decision was used.", current_mode)

    row = dict(current_features or {})
    missing = [name for name in FEATURES if row.get(name) in (None, "")]
    if missing:
        return _error(f"Runtime features are missing: {missing}", current_mode, sources)
    try:
        current = np.asarray([float(row[name]) for name in FEATURES], dtype=float)
    except (TypeError, ValueError) as exc:
        return _error(f"Runtime feature values are not numeric: {exc}", current_mode, sources)
    if not np.isfinite(current).all():
        return _error("Runtime feature vector contains non-finite values.", current_mode, sources)

    reproduced, lime = _local_lime(model, scaler, label_encoder, training_frame, current, expected_consensus)
    prediction = predict_consensus(model, scaler, label_encoder, row)
    selected = expected_consensus or reproduced

    ranges = training_ranges if training_ranges else compute_training_ranges(training_frame)
    out_of_range = check_training_range(row, ranges)
    range_table = {
        f: {"training_min": ranges[f]["min"], "training_max": ranges[f]["max"], "runtime_value": float(row[f])}
        for f in FEATURES if f in ranges
    }

    feature_display = {name: round(float(row[name]), 6) for name in FEATURES}
    risk_value = risk.get("risk_score")
    try:
        risk_value = round(float(risk_value), 2) if risk_value is not None else None
    except (TypeError, ValueError):
        risk_value = None
    trust_value = tx.get("trust_score", tx.get("owner_trust_score"))
    try:
        trust_value = round(float(trust_value), 2) if trust_value is not None else None
    except (TypeError, ValueError):
        trust_value = None

    return {
        "module": "Module 4 — Explainable Dynamic Consensus",
        "selected_consensus": selected,
        "current_consensus": current_mode,
        "selection_matches_current": selected == current_mode,
        "model_prediction_matches_captured": lime["matches_captured_prediction"],
        "decision_method": "Trained consensus selector on runtime-measured inputs",
        "algorithms": list(SUPPORTED),
        "risk_score": risk_value,
        "risk_level": str(risk.get("risk_level", "Unavailable")),
        "fraud_probability": risk.get("fraud_probability"),
        "risk_recommendation": risk.get("recommendation"),
        "trust_score": trust_value,
        "network": {
            "active_nodes": int(float(row["node_count"])),
            "latency_ms": round(float(row["network_latency_ms"]), 3),
            "throughput_tps": round(float(row["tx_throughput_tps"]), 6),
        },
        "input_features": feature_display,
        "input_sources": sources,
        "source_types": source_types,
        "runtime_measurements": measurements,
        "limitations": limitations,
        "class_probabilities": prediction["class_probabilities"],
        "lime_explanation": lime,
        "factors": [
            {
                "factor": FEATURE_LABELS[i],
                "value": str(feature_display[f]),
                "source_type": source_types.get(f, "unknown"),
                "effect": sources.get(f, "Source not recorded"),
                "out_of_range": any(o["feature"] == f for o in out_of_range),
            }
            for i, f in enumerate(FEATURES)
        ],
        "algorithm_options": {
            name: ("ML prediction" if name == selected else "Not predicted for this input")
            for name in SUPPORTED
        },
        "training_feature_ranges": range_table,
        "out_of_training_range": out_of_range,
    }
