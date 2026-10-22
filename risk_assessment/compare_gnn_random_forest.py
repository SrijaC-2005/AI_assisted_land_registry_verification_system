
"""
Compare the existing LandFraudGNN with a Random Forest baseline.

Run from the project root:
    python -m risk_assessment.compare_gnn_random_forest

Existing project files and GNN weights are not modified.
Outputs are saved in risk_assessment/model_comparison_results/.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

from risk_assessment.gnn_data import prepare_gnn_data
from risk_assessment.model import LandFraudGNN


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "land_fraud_gnn.pth"
OUTPUT_DIR = ROOT / "model_comparison_results"
RANDOM_STATE = 42


# ---------------------------------------------------------
# Metric and threshold functions
# ---------------------------------------------------------

def find_threshold(y_true, probabilities):
    """Select the threshold with the best MCC on validation data."""
    best_threshold = 0.50
    best_mcc = -1.0

    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        score = matthews_corrcoef(y_true, predictions)

        if score > best_mcc:
            best_mcc = score
            best_threshold = float(threshold)

    return best_threshold


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, probabilities)
        pr_auc = average_precision_score(y_true, probabilities)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "specificity": float(specificity),
        "g_mean": float(np.sqrt(recall * specificity)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------
# Graph 1: Presentation-ready grouped bar chart
# ---------------------------------------------------------

def save_comparison_graph(comparison):
    chart_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "mcc",
        "specificity",
        "g_mean",
    ]

    metric_labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "PR-AUC",
        "MCC",
        "Specificity",
        "G-Mean",
    ]

    gnn_scores = (
        comparison.loc[comparison["Model"] == "GNN", chart_metrics]
        .iloc[0]
        .to_numpy(dtype=float)
    )

    rf_scores = (
        comparison.loc[
            comparison["Model"] == "Random Forest", chart_metrics
        ]
        .iloc[0]
        .to_numpy(dtype=float)
    )

    x = np.arange(len(metric_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(15, 8))

    bars1 = ax.bar(
        x - width / 2,
        gnn_scores,
        width,
        label="GNN",
        color="#2878B5",
        edgecolor="white",
        linewidth=0.7,
    )

    bars2 = ax.bar(
        x + width / 2,
        rf_scores,
        width,
        label="Random Forest",
        color="#F28E2B",
        edgecolor="white",
        linewidth=0.7,
    )

    # Exact value labels above every bar.
    ax.bar_label(
        bars1,
        fmt="%.3f",
        padding=4,
        fontsize=9,
        fontweight="bold",
    )

    ax.bar_label(
        bars2,
        fmt="%.3f",
        padding=4,
        fontsize=9,
        fontweight="bold",
    )

    # Use a meaningful 0–1 scale with a little room for labels.
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_xlabel("Evaluation Metric", fontsize=12, labelpad=10)

    ax.set_title(
        "GNN vs Random Forest — Test Metrics",
        fontsize=17,
        fontweight="bold",
        pad=20,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.30,
    )
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    png_path = OUTPUT_DIR / "comparison_metrics.png"
    svg_path = OUTPUT_DIR / "comparison_metrics.svg"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("\nGraph saved:")
    print(png_path)
    print(svg_path)


# ---------------------------------------------------------
# Graph 2: Save the complete comparison table as an image
# ---------------------------------------------------------

def save_comparison_table(comparison):
    columns = [
        "Model",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "mcc",
        "specificity",
        "g_mean",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    table_df = comparison[columns].copy()

    # Use readable presentation labels.
    table_df.columns = [
        "Model",
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "PR-AUC",
        "MCC",
        "Specificity",
        "G-Mean",
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    # Format numeric values consistently.
    for column in table_df.columns:
        if column == "Model":
            continue
        if column in ["TN", "FP", "FN", "TP"]:
            table_df[column] = table_df[column].map(
                lambda value: f"{int(value)}"
            )
        else:
            table_df[column] = table_df[column].map(
                lambda value: f"{float(value):.4f}"
            )

    fig, ax = plt.subplots(figsize=(20, 3.5))
    ax.axis("off")

    ax.set_title(
        "GNN vs Random Forest — Test Set Results",
        fontsize=18,
        fontweight="bold",
        pad=22,
    )

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    # Style the header and alternate row backgrounds.
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#C9D3DF")

        if row == 0:
            cell.set_facecolor("#17324D")
            cell.set_text_props(
                color="white",
                weight="bold",
            )
        elif row == 1:
            cell.set_facecolor("#E8F1FB")
        else:
            cell.set_facecolor("#FFF0E1")

        if col == 0 and row > 0:
            cell.set_text_props(weight="bold")

    fig.tight_layout()

    png_path = OUTPUT_DIR / "comparison_table.png"
    svg_path = OUTPUT_DIR / "comparison_table.svg"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("\nComparison table saved:")
    print(png_path)
    print(svg_path)


# ---------------------------------------------------------
# Main comparison
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained GNN weights not found: {MODEL_PATH}\n"
            "Train the existing GNN first. This script does not "
            "retrain or replace it."
        )

    print("\nLoading the existing graph dataset...")
    data = prepare_gnn_data()

    transaction_indices = torch.where(data.transaction_mask)[0]

    X_all = (
        data.x[transaction_indices]
        .detach()
        .cpu()
        .numpy()
    )

    y_all = (
        data.y[transaction_indices]
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    if not set(np.unique(y_all)).issubset({0, 1}):
        raise ValueError("Expected labels 0=normal and 1=fraud.")

    if len(np.unique(y_all)) != 2:
        raise ValueError(
            "Dataset must contain both normal and fraud transactions."
        )

    # Stratified 60/20/20 train-validation-test split.
    indices = np.arange(len(y_all))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_all[train_val_idx],
    )

    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]

    print(f"\nTransactions: {len(y_all)}")
    print(
        f"Train: {len(train_idx)} | "
        f"Validation: {len(val_idx)} | "
        f"Test: {len(test_idx)}"
    )
    print(
        f"Test normal: {(y_test == 0).sum()} | "
        f"Test fraud: {(y_test == 1).sum()}"
    )

    # Load existing GNN weights without retraining.
    gnn = LandFraudGNN(
        input_features=data.x.shape[1],
        hidden_channels=64,
    )

    state = torch.load(MODEL_PATH, map_location="cpu")
    gnn.load_state_dict(state)
    gnn.eval()

    with torch.no_grad():
        logits = gnn(data.x, data.edge_index)
        all_probabilities = torch.softmax(logits, dim=1)[:, 1]
        gnn_probabilities = (
            all_probabilities[transaction_indices]
            .cpu()
            .numpy()
        )

    gnn_threshold = find_threshold(
        y_val,
        gnn_probabilities[val_idx],
    )

    # Train Random Forest only on the training subset.
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=1,
    )

    start = time.perf_counter()
    rf.fit(X_all[train_idx], y_train)
    rf_train_seconds = time.perf_counter() - start

    fraud_column = list(rf.classes_).index(1)

    rf_val_probabilities = rf.predict_proba(
        X_all[val_idx]
    )[:, fraud_column]

    rf_threshold = find_threshold(
        y_val,
        rf_val_probabilities,
    )

    start = time.perf_counter()
    rf_test_probabilities = rf.predict_proba(
        X_all[test_idx]
    )[:, fraud_column]
    rf_inference_seconds = time.perf_counter() - start

    # Evaluate both models on the same final test set.
    results = {
        "GNN": calculate_metrics(
            y_test,
            gnn_probabilities[test_idx],
            gnn_threshold,
        ),
        "Random Forest": calculate_metrics(
            y_test,
            rf_test_probabilities,
            rf_threshold,
        ),
    }

    results["Random Forest"]["training_seconds"] = float(
        rf_train_seconds
    )
    results["Random Forest"]["test_inference_seconds"] = float(
        rf_inference_seconds
    )
    results["GNN"]["training_seconds"] = None
    results["GNN"]["test_inference_seconds"] = None

    # Convert NaN values to JSON null if necessary.
    json_results = json.loads(
        json.dumps(results, allow_nan=True)
    )

    def clean_json(value):
        if isinstance(value, dict):
            return {key: clean_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean_json(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    with open(
        OUTPUT_DIR / "comparison_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(clean_json(json_results), file, indent=2)

    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "mcc",
        "specificity",
        "g_mean",
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            **{
                key: metrics[key]
                for key in metric_keys
            },
        })

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        OUTPUT_DIR / "comparison_metrics.csv",
        index=False,
    )

    print("\n========== TEST SET COMPARISON ==========")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        "\nConfusion matrices "
        "(rows=true [normal, fraud], columns=predicted [normal, fraud]):"
    )

    for model_name, metrics in results.items():
        print(
            f"\n{model_name}:\n"
            f"{np.array(metrics['confusion_matrix'])}"
        )

    # Create both presentation-ready images.
    save_comparison_graph(comparison)
    save_comparison_table(comparison)

    print("\n========== INTERPRETATION ==========")
    for metric in (
        "f1_score",
        "recall",
        "pr_auc",
        "roc_auc",
        "mcc",
    ):
        g = results["GNN"][metric]
        r = results["Random Forest"][metric]

        if np.isfinite(g) and np.isfinite(r):
            winner = (
                "GNN" if g > r
                else "Random Forest" if r > g
                else "Tie"
            )
            print(
                f"{metric}: {winner} "
                f"(GNN={g:.4f}, Random Forest={r:.4f})"
            )

    print("\nAll files saved in:")
    print(OUTPUT_DIR.resolve())
    print("- comparison_metrics.csv")
    print("- comparison_results.json")
    print("- comparison_metrics.png")
    print("- comparison_metrics.svg")
    print("- comparison_table.png")
    print("- comparison_table.svg")


if __name__ == "__main__":
    main()
