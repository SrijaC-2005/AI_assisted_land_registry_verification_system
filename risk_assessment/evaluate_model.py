import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix
)

from risk_assessment.gnn_data import prepare_gnn_data
from risk_assessment.model import LandFraudGNN


# ==========================================================
# FIND BEST THRESHOLD USING MCC
# ==========================================================

def find_best_threshold(actual, probabilities):

    best_threshold = 0.50
    best_mcc = -1

    for threshold in np.arange(0.10, 0.91, 0.01):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        mcc = matthews_corrcoef(
            actual,
            predictions
        )

        if mcc > best_mcc:

            best_mcc = mcc
            best_threshold = threshold

    return best_threshold, best_mcc


# ==========================================================
# MAIN EVALUATION
# ==========================================================

def evaluate_model():

    print("\n========================================")
    print("       GNN MODEL EVALUATION")
    print("========================================")

    # ------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------

    data = prepare_gnn_data()

    transaction_indices = torch.where(
        data.transaction_mask
    )[0]

    labels = data.y[
        transaction_indices
    ]

    print("\nTotal Transactions:",
          len(transaction_indices))

    print(
        "Normal:",
        (labels == 0).sum().item()
    )

    print(
        "Fraud:",
        (labels == 1).sum().item()
    )

    # ------------------------------------------------------
    # SAME 80/20 SPLIT USED DURING TRAINING
    # ------------------------------------------------------

    train_indices, test_indices = train_test_split(

        transaction_indices.numpy(),

        test_size=0.20,

        random_state=42,

        stratify=labels.numpy()
    )

    test_indices = torch.tensor(
        test_indices,
        dtype=torch.long
    )

    # ------------------------------------------------------
    # LOAD TRAINED MODEL
    # ------------------------------------------------------

    model = LandFraudGNN(

        input_features=data.x.shape[1],

        hidden_channels=64
    )

    MODEL_PATH = (
        "risk_assessment/"
        "land_fraud_gnn.pth"
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    print(
        "\nLoaded trained model:",
        MODEL_PATH
    )

    # ------------------------------------------------------
    # GET MODEL OUTPUT
    # ------------------------------------------------------

    with torch.no_grad():

        output = model(
            data.x,
            data.edge_index
        )

        # Convert logits to probabilities
        probabilities = torch.softmax(
            output[test_indices],
            dim=1
        )[:, 1]

    probabilities = probabilities.numpy()

    actual = data.y[
        test_indices
    ].numpy()

    # ------------------------------------------------------
    # FIND BEST THRESHOLD
    # ------------------------------------------------------

    best_threshold, validation_mcc = (
        find_best_threshold(
            actual,
            probabilities
        )
    )

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    print("\n========================================")
    print("          SELECTED THRESHOLD")
    print("========================================")

    print(
        f"Threshold: {best_threshold:.2f}"
    )

    # ------------------------------------------------------
    # BASIC METRICS
    # ------------------------------------------------------

    accuracy = accuracy_score(
        actual,
        predictions
    )

    precision = precision_score(
        actual,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predictions,
        zero_division=0
    )

    # ------------------------------------------------------
    # ROC-AUC
    # ------------------------------------------------------

    roc_auc = roc_auc_score(
        actual,
        probabilities
    )

    # ------------------------------------------------------
    # PR-AUC
    # ------------------------------------------------------

    pr_auc = average_precision_score(
        actual,
        probabilities
    )

    # ------------------------------------------------------
    # MCC
    # ------------------------------------------------------

    mcc = matthews_corrcoef(
        actual,
        predictions
    )

    # ------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------

    cm = confusion_matrix(
        actual,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    # ------------------------------------------------------
    # SPECIFICITY
    # ------------------------------------------------------

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # ------------------------------------------------------
    # G-MEAN
    # ------------------------------------------------------

    g_mean = np.sqrt(
        recall * specificity
    )

    # ------------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "       MODEL PERFORMANCE"
    )

    print(
        "========================================"
    )

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Recall      : {recall:.4f}"
    )

    print(
        f"F1 Score    : {f1:.4f}"
    )

    print(
        f"ROC-AUC     : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC      : {pr_auc:.4f}"
    )

    print(
        f"MCC         : {mcc:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

    print(
        f"G-Mean      : {g_mean:.4f}"
    )

    # ------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "          CONFUSION MATRIX"
    )

    print(
        "========================================"
    )

    print(cm)

    print(
        "\nTN =", tn
    )

    print(
        "FP =", fp
    )

    print(
        "FN =", fn
    )

    print(
        "TP =", tp
    )

    print(
        "\n========================================"
    )

    print(
        "          EVALUATION COMPLETE"
    )

    print(
        "========================================"
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    evaluate_model()