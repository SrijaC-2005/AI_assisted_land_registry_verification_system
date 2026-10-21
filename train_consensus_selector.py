"""Train the consensus selector and save everything the app needs to use it honestly.

Usage
-----
  python train_consensus_selector.py --source historical
      Train on Dataset/Current_Network_Metrics.csv (simulator output).
      Reproduces the current model and, unlike before, saves feature ranges.

  python train_consensus_selector.py --source benchmark       (default)
      Train on Dataset/Runtime_Consensus_Benchmarks.csv: runtime-measured
      scenarios whose 'consensus' label was assigned from documented
      benchmark results. Never fill labels randomly.

Outputs (FL_SGD/consensus_selector/)
  model.pkl, scaler.pkl, label_encoder.pkl
  lime_background.csv  - training rows; app.py uses them for LIME and ranges
  evaluation.json      - metrics, feature_ranges, label audit, data source
  backup_<timestamp>/  - previous artifacts, so nothing is lost
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset"
SOURCES = {
    "historical": DATA_DIR / "Current_Network_Metrics.csv",
    "benchmark": DATA_DIR / "Runtime_Consensus_Benchmarks.csv",
}
OBSERVATIONS_PATH = DATA_DIR / "Runtime_Measured_Metrics.csv"
INDEPENDENT_PATH = DATA_DIR / "Runtime_Consensus_Benchmarks_Independent.csv"
OUT_DIR = BASE_DIR / "FL_SGD" / "consensus_selector"
ARTIFACTS = ["model.pkl", "scaler.pkl", "label_encoder.pkl", "lime_background.csv", "evaluation.json"]
FEATURES = [
    "node_count",
    "network_latency_ms",
    "tx_throughput_tps",
    "security_risk_score",
    "fault_tolerance_requirement",
]
SUPPORTED = {"PoW", "PoS", "PBFT", "Raft", "HotStuff"}
MIN_ROWS = {"historical": 100, "benchmark": 25}


def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + ["consensus"] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns: {missing}. Found: {list(df.columns)}")
    clean = df[FEATURES + ["consensus"]].copy()
    for f in FEATURES:
        clean[f] = pd.to_numeric(clean[f], errors="coerce")
    clean["consensus"] = clean["consensus"].astype(str).str.strip()
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[clean["consensus"].isin(SUPPORTED)]
    counts = clean["consensus"].value_counts()
    return clean[clean["consensus"].isin(counts[counts >= 2].index)]


def label_audit(clean: pd.DataFrame) -> dict:
    """Show how well a single feature separates the labels.

    If one feature alone almost determines the class, the labels probably encode
    how the data was generated (which protocol the simulator ran), not which
    protocol is best for the scenario.
    """
    per_feature = {}
    for f in FEATURES:
        X = clean[[f]].to_numpy(dtype=float)
        y = clean["consensus"].to_numpy()
        clf = make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", max_iter=2000, random_state=0))
        try:
            score = float(cross_val_score(clf, X, y, cv=3).mean())
        except ValueError:
            score = None
        per_feature[f] = {
            "single_feature_cv_accuracy": None if score is None else round(score, 4),
            "range_by_class": {
                c: [float(g[f].min()), float(g[f].max())] for c, g in clean.groupby("consensus")
            },
        }
    chance = 1.0 / clean["consensus"].nunique()
    leaks = [f for f, v in per_feature.items()
             if v["single_feature_cv_accuracy"] and v["single_feature_cv_accuracy"] > max(0.6, 2.5 * chance)]
    return {
        "chance_accuracy": round(chance, 4),
        "per_feature": per_feature,
        "features_that_alone_predict_label": leaks,
        "note": (
            "A feature that alone predicts the label well suggests the label reflects how rows were generated. "
            "For the historical simulator, 'consensus' is the protocol that was being simulated, not a "
            "benchmark judgement of the best protocol."
        ),
    }


def duplication_report(df_raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    exact = int(clean.duplicated(FEATURES + ["consensus"]).sum())
    conflicting = int(clean.groupby(FEATURES)["consensus"].nunique().gt(1).sum())
    policy_cells = int(clean.groupby(["security_risk_score", "fault_tolerance_requirement"]).ngroups)
    return {
        "raw_rows": int(len(df_raw)),
        "usable_rows": int(len(clean)),
        "exact_duplicate_rows_removed": exact,
        "feature_vectors_with_conflicting_labels": conflicting,
        "distinct_risk_x_fault_tolerance_cells": policy_cells,
    }


def fit(X, y):
    scaler = StandardScaler().fit(X)
    model = SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=3000, tol=1e-4, random_state=42)
    model.fit(scaler.transform(X), y)
    return model, scaler


def grouped_evaluation(clean: pd.DataFrame, encoder: LabelEncoder) -> dict:
    """Hold out whole groups so the test set contains values never seen in training."""
    X = clean[FEATURES].to_numpy(dtype=float)
    y = encoder.transform(clean["consensus"])
    results = {}
    for name, column in [("unseen_security_risk_values", "security_risk_score"),
                         ("unseen_node_counts", "node_count"),
                         ("unseen_load_levels", "tx_throughput_tps")]:
        groups = clean[column].to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            results[name] = {"skipped": "fewer than 2 distinct values"}
            continue
        folds = min(5, n_groups)
        accs = []
        for train_idx, test_idx in GroupKFold(n_splits=folds).split(X, y, groups):
            if len(np.unique(y[train_idx])) < 2:
                continue
            m, sc = fit(X[train_idx], y[train_idx])
            accs.append(float(accuracy_score(y[test_idx], m.predict(sc.transform(X[test_idx])))))
        results[name] = {
            "folds": len(accs),
            "mean_accuracy": round(float(np.mean(accs)), 4) if accs else None,
            "min_fold_accuracy": round(float(np.min(accs)), 4) if accs else None,
        }
    return results


def independent_evaluation(model, scaler, encoder) -> dict:
    if not INDEPENDENT_PATH.exists():
        return {"skipped": f"{INDEPENDENT_PATH.name} not found. Run: python benchmark_consensus.py --independent"}
    ind = load_clean(INDEPENDENT_PATH)
    ind = ind[ind["consensus"].isin(encoder.classes_)]
    if ind.empty:
        return {"skipped": "no rows with known classes"}
    X = ind[FEATURES].to_numpy(dtype=float)
    y = encoder.transform(ind["consensus"])
    pred = model.predict(scaler.transform(X))
    raw = pd.read_csv(INDEPENDENT_PATH)
    ft_band = None
    if "fault_tolerance_requirement" in raw:
        mask = (ind["fault_tolerance_requirement"] > 0.5) & (ind["fault_tolerance_requirement"] < 0.75)
        if mask.any():
            ft_band = round(float(accuracy_score(y[mask.to_numpy()], pred[mask.to_numpy()])), 4)
    return {
        "rows": int(len(ind)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "accuracy_on_unseen_fault_tolerance_values": ft_band,
        "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(len(encoder.classes_))).tolist(),
        "classes": encoder.classes_.tolist(),
    }


def backup_existing() -> Path | None:
    existing = [OUT_DIR / a for a in ARTIFACTS if (OUT_DIR / a).exists()]
    if not existing:
        return None
    dest = OUT_DIR / f"backup_{datetime.now():%Y%m%d_%H%M%S}"
    dest.mkdir(parents=True)
    for p in existing:
        shutil.copy2(p, dest / p.name)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(SOURCES), default="benchmark")
    args = parser.parse_args()

    data_path = SOURCES[args.source]
    if not data_path.exists():
        msg = f"Training file not found: {data_path}"
        if args.source == "benchmark":
            msg += ("\nCreate it from benchmark runs: copy rows from Runtime_Measured_Metrics.csv and set "
                    "'consensus' only from documented benchmark results.\n"
                    "To rebuild the current model with saved ranges instead, run: "
                    "python train_consensus_selector.py --source historical")
        raise SystemExit(msg)

    raw = pd.read_csv(data_path)
    clean = load_clean(data_path)
    dup = duplication_report(raw, clean)
    clean = clean.drop_duplicates(FEATURES + ["consensus"]).reset_index(drop=True)
    if len(clean) < MIN_ROWS[args.source] or clean["consensus"].nunique() < 2:
        raise SystemExit(f"Only {len(clean)} usable rows / {clean['consensus'].nunique()} labels in {data_path.name}.")

    X = clean[FEATURES].to_numpy(dtype=float)
    encoder = LabelEncoder()
    y = encoder.fit_transform(clean["consensus"].to_numpy())
    test_count = max(len(encoder.classes_), int(round(len(y) * 0.2)))
    test_count = min(test_count, len(y) - len(encoder.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_count, random_state=42, stratify=y
    )
    model, scaler = fit(X_train, y_train)
    pred = model.predict(scaler.transform(X_test))
    accuracy = float(accuracy_score(y_test, pred))
    labels = np.arange(len(encoder.classes_))

    feature_ranges = {
        f: {"min": float(clean[f].min()), "max": float(clean[f].max()),
            "p01": float(clean[f].quantile(0.01)), "p99": float(clean[f].quantile(0.99))}
        for f in FEATURES
    }
    audit = label_audit(clean)
    grouped = grouped_evaluation(clean, encoder)
    independent = independent_evaluation(model, scaler, encoder)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backup = backup_existing()
    joblib.dump(model, OUT_DIR / "model.pkl")
    joblib.dump(scaler, OUT_DIR / "scaler.pkl")
    joblib.dump(encoder, OUT_DIR / "label_encoder.pkl")
    clean[FEATURES].to_csv(OUT_DIR / "lime_background.csv", index=False)

    warning = (
        "Trained on simulator output. Labels are the protocol each simulated row was generated for, "
        "not benchmark judgements. Feature scales (network latency, TPS) differ from this app's "
        "in-process measurements, so the app will reject most live predictions as out of range. "
        "Holdout accuracy measures agreement with the simulator, not real-world optimality."
        if args.source == "historical" else
        "Holdout accuracy on benchmark-labelled runtime scenarios. Not proof of optimality; "
        "evaluate on independent loads before claiming the selector picks the best protocol."
    )
    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training_source": args.source,
        "training_data_source": str(data_path),
        "features": FEATURES,
        "classes": encoder.classes_.tolist(),
        "rows_used": int(len(clean)),
        "class_counts": {str(k): int(v) for k, v in clean["consensus"].value_counts().items()},
        "test_accuracy": accuracy,
        "test_accuracy_note": (
            "Random stratified split. Rows sharing a policy cell appear in both train and test, so this "
            "number is optimistic; see grouped_evaluation and independent_evaluation."
        ),
        "duplication_report": dup,
        "grouped_evaluation": grouped,
        "independent_evaluation": independent,
        "classification_report": classification_report(
            y_test, pred, labels=labels, target_names=encoder.classes_, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "feature_ranges": feature_ranges,
        "label_audit": audit,
        "warning": warning,
    }
    (OUT_DIR / "evaluation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Source: {args.source} ({data_path.name}), rows={len(clean)}, classes={list(encoder.classes_)}")
    print(f"Data: {dup}")
    print(f"Random-split holdout accuracy: {accuracy:.4f}  (optimistic; same policy cells in train and test)")
    for name, res in grouped.items():
        print(f"Grouped CV, {name}: {res}")
    print(f"Independent scenarios: {independent}")
    print("Training ranges saved (the app rejects predictions outside these):")
    for f, r in feature_ranges.items():
        print(f"  {f:28s} {r['min']:>10.4g} .. {r['max']:<10.4g}")
    if audit["features_that_alone_predict_label"]:
        print("Label audit: these features alone predict the label well ->",
              ", ".join(audit["features_that_alone_predict_label"]))
    if backup:
        print(f"Previous artifacts backed up to: {backup}")
    print(f"Saved artifacts to: {OUT_DIR}")
    print("Note:", warning)


if __name__ == "__main__":
    main()