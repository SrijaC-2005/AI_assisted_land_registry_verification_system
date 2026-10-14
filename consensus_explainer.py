"""Model-based Module 4 explanation. LIME explains the trained selector, not hand-written rules."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

SUPPORTED = ("PoW", "PoS", "PBFT", "Raft", "HotStuff")
FEATURES = [
    "node_count", "network_latency_ms", "tx_throughput_tps",
    "energy_joules_per_min", "security_risk_score", "vehicle_count",
    "vehicle_count_variability", "fault_tolerance_requirement",
]
FEATURE_LABELS = [
    "Node count", "Network latency (ms)", "Transaction throughput (TPS)",
    "Energy (joules/min)", "Security risk score", "Vehicle count",
    "Vehicle-count variability", "Fault-tolerance requirement",
]


def _probabilities(model, scaler, rows):
    scaled = scaler.transform(np.asarray(rows, dtype=float))
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(scaled), dtype=float)
    scores = np.asarray(model.decision_function(scaled), dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _local_lime(model, scaler, encoder, background: pd.DataFrame, current: np.ndarray):
    bg = background[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(bg) < 10:
        raise ValueError("Need at least 10 complete representative background rows for LIME.")
    class_codes = list(model.classes_)
    class_names = [str(encoder.inverse_transform([int(code)])[0]) for code in class_codes]
    pred_code = int(model.predict(scaler.transform(current.reshape(1, -1)))[0])
    class_index = class_codes.index(pred_code)
    predicted_name = str(encoder.inverse_transform([pred_code])[0])
    explainer = LimeTabularExplainer(
        training_data=bg.to_numpy(dtype=float), feature_names=FEATURE_LABELS,
        class_names=class_names, mode="classification", discretize_continuous=True,
        random_state=42,
    )
    exp = explainer.explain_instance(
        current, lambda rows: _probabilities(model, scaler, rows),
        labels=[class_index], num_features=len(FEATURES), num_samples=1800,
    )
    weights = exp.as_list(label=class_index)
    return predicted_name, {
        "method": "LIME for the trained consensus-selection classifier",
        "target_consensus": predicted_name,
        "scope": "Local surrogate approximation of the trained selector for this input; background distribution comes from labelled project network-metrics rows.",
        "features": [
            {"feature": name, "weight": round(float(weight), 5),
             "direction": "supports this model prediction locally" if weight > 0 else "opposes this model prediction locally"}
            for name, weight in weights
        ],
        "warning": "LIME weights are local surrogate approximations, not causal effects. The model learns the consensus labels in the training CSV; if those labels came from rules, it may learn to imitate those rules.",
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
) -> Dict[str, Any]:
    risk = risk_assessment or {}
    tx = transaction or {}
    metrics = network_metrics or {}
    sources = dict(metrics.get("input_sources") or {})

    if model is None or scaler is None or label_encoder is None or training_frame is None:
        return {
            "module": "Module 4 — Explainable Dynamic Consensus",
            "error": "Trained consensus selector is unavailable. Train it first with train_consensus_selector.py; no rule-based result is presented as an AI prediction.",
            "selected_consensus": None, "current_consensus": current_mode,
            "lime_explanation": {"method": "LIME not run", "features": [], "warning": "No trained selector loaded."},
        }

    row = dict(current_features or {})
    # For fields not supplied by runtime instrumentation, use the latest complete dataset row,
    # and label their source as simulated/project data rather than live-measured.
    for feature in FEATURES:
        if row.get(feature) is None or row.get(feature) == "":
            if training_frame is not None and feature in training_frame.columns and not training_frame.empty:
                valid = pd.to_numeric(training_frame[feature], errors="coerce").dropna()
                if not valid.empty:
                    row[feature] = float(valid.iloc[-1])
                    sources.setdefault(feature, "Simulated/project dataset (latest valid row; not live telemetry)")
    missing = [f for f in FEATURES if row.get(f) is None or row.get(f) == ""]
    if missing:
        return {"module": "Module 4 — Explainable Dynamic Consensus", "error": f"Cannot predict: missing required features {missing}", "selected_consensus": None, "current_consensus": current_mode, "input_sources": sources, "lime_explanation": {"method": "LIME not run", "features": [], "warning": "Prediction skipped because required inputs are missing."}}

    current = np.asarray([float(row[f]) for f in FEATURES], dtype=float)
    if not np.isfinite(current).all():
        return {"module": "Module 4 — Explainable Dynamic Consensus", "error": "Input features contain non-finite values; prediction skipped.", "selected_consensus": None, "current_consensus": current_mode, "input_sources": sources}

    selected, lime = _local_lime(model, scaler, label_encoder, training_frame, current)
    scaled = scaler.transform(current.reshape(1, -1))
    probs = model.predict_proba(scaled)[0] if hasattr(model, "predict_proba") else None
    class_codes = list(model.classes_)
    probabilities = {}
    if probs is not None:
        probabilities = {str(label_encoder.inverse_transform([int(code)])[0]): round(float(p), 5) for code, p in zip(class_codes, probs)}
    current_label = current_mode if current_mode in SUPPORTED else "PoW"
    feature_display = {name: round(float(row[name]), 5) for name in FEATURES}
    return {
        "module": "Module 4 — Explainable Dynamic Consensus",
        "selected_consensus": selected,
        "current_consensus": current_label,
        "selection_matches_current": selected == current_label,
        "reason": f"The existing dynamic consensus selector selected {selected} for the captured input row. LIME explains that selection; it does not recommend a different protocol.",
        "decision_method": "Trained SGD consensus-selection classifier",
        "algorithm_explanation": f"Selected by the existing dynamic consensus selector: {selected}. Active blockchain mode: {current_label}.",
        "algorithms": list(SUPPORTED),
        # Transaction fraud risk is separate from the selector's security_risk_score input.
        # Never substitute the sampled network-metrics feature for the transaction's AI score.
        "risk_score": (
            round(float(risk.get("risk_score")), 2)
            if risk.get("risk_score") is not None and str(risk.get("risk_score")).strip() != ""
            else None
        ),
        "risk_level": str(risk.get("risk_level", "Unavailable")),
        # The uploaded application has no owner-trust calculation or persisted owner-trust field.
        # Missing trust must remain unavailable, not be fabricated as 0.
        "trust_score": (
            tx.get("trust_score", tx.get("owner_trust_score"))
            if tx.get("trust_score", tx.get("owner_trust_score")) is not None
            else None
        ),
        "selector_security_risk_input": round(float(row["security_risk_score"]), 5),
        "network": {
            "active_nodes": int(float(row["node_count"])),
            "latency_ms": float(row["network_latency_ms"]),
            "throughput_tps": float(row["tx_throughput_tps"]),
            "energy_joules_per_min": float(row["energy_joules_per_min"]),
            "source": "Per-feature provenance is shown separately.",
        },
        "input_features": feature_display,
        "input_sources": sources,
        "class_probabilities": probabilities,
        "lime_explanation": lime,
        "factors": [{"factor": FEATURE_LABELS[i], "value": str(feature_display[f]), "effect": sources.get(f, "Source not recorded"), "impact": "model input"} for i, f in enumerate(FEATURES)],
        "algorithm_options": {name: ("Predicted class" if name == selected else "Alternative class; not selected by this input") for name in SUPPORTED},
        "method_note": "LIME explains the existing selector locally for the captured input row. Values inherited from the project CSV are dataset values, not live measurements. Module 4 observes the selection and does not change consensus.",
    }
