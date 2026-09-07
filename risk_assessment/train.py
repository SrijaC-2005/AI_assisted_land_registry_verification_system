import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from gnn_data import prepare_gnn_data
from model import LandFraudGNN


def train_model():

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    data = prepare_gnn_data()

    # --------------------------------------------------
    # GET TRANSACTION NODES
    # --------------------------------------------------

    transaction_indices = torch.where(
        data.transaction_mask
    )[0]

    labels = data.y[
        transaction_indices
    ]

    # --------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------

    train_indices, test_indices = train_test_split(

        transaction_indices.numpy(),

        test_size=0.20,

        random_state=42,

        stratify=labels.numpy()
    )

    train_indices = torch.tensor(
        train_indices,
        dtype=torch.long
    )

    test_indices = torch.tensor(
        test_indices,
        dtype=torch.long
    )

    print("\n========== DATA SPLIT ==========")

    print(
        "Training Transactions:",
        len(train_indices)
    )

    print(
        "Testing Transactions:",
        len(test_indices)
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = LandFraudGNN(

        input_features=data.x.shape[1],

        hidden_channels=64
    )

    # --------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=0.005,

        weight_decay=1e-4
    )

    # --------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------

    train_labels = data.y[
        train_indices
    ]

    normal_count = (
        train_labels == 0
    ).sum().item()

    fraud_count = (
        train_labels == 1
    ).sum().item()

    total = normal_count + fraud_count

    # Higher importance for minority fraud class
    normal_weight = total / (
        2 * normal_count
    )

    fraud_weight = total / (
        2 * fraud_count
    )

    class_weights = torch.tensor(
        [
            normal_weight,
            fraud_weight
        ],
        dtype=torch.float
    )

    print("\n========== CLASS WEIGHTS ==========")

    print(
        "Normal Weight:",
        normal_weight
    )

    print(
        "Fraud Weight:",
        fraud_weight
    )

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    epochs = 150

    print("\n========== TRAINING ==========")

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        # Forward pass
        output = model(
            data.x,
            data.edge_index
        )

        # Calculate loss only for training transactions
        loss = F.cross_entropy(

            output[train_indices],

            data.y[train_indices],

            weight=class_weights
        )

        # Backpropagation
        loss.backward()

        optimizer.step()

        # Print every 10 epochs
        if (
            (epoch + 1) % 10 == 0
            or epoch == 0
        ):

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| Loss: {loss.item():.4f}"
            )

    # --------------------------------------------------
    # TESTING
    # --------------------------------------------------

    model.eval()

    with torch.no_grad():

        output = model(
            data.x,
            data.edge_index
        )

        predictions = (
            output[test_indices]
            .argmax(dim=1)
        )

        actual = data.y[
            test_indices
        ]

    predictions = predictions.numpy()

    actual = actual.numpy()

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

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

    cm = confusion_matrix(
        actual,
        predictions
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print(
        "\n========== MODEL PERFORMANCE =========="
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        "\n========== CONFUSION MATRIX =========="
    )

    print(cm)

    print(
        "\n========== CLASSIFICATION REPORT =========="
    )

    print(
        classification_report(
            actual,
            predictions,
            target_names=[
                "Normal",
                "Fraud"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    MODEL_PATH = (
        "risk_assessment/"
        "land_fraud_gnn.pth"
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print(
        f"\nModel saved successfully: "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":

    train_model()