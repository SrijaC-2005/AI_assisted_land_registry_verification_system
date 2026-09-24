import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import shap

from risk_assessment.model import LandFraudGNN
from risk_assessment.gnn_data import prepare_gnn_data


# --------------------------------------------------
# MODEL PATH
# --------------------------------------------------

MODEL_PATH = "risk_assessment/land_fraud_gnn.pth"


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

def load_model(data):

    model = LandFraudGNN(
        input_features=data.x.shape[1],
        hidden_channels=64
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


# --------------------------------------------------
# GET RISK LEVEL
# --------------------------------------------------

def get_risk_level(risk_score):

    if risk_score < 30:
        return "LOW"

    elif risk_score < 70:
        return "MEDIUM"

    else:
        return "HIGH"


# --------------------------------------------------
# GET RECOMMENDATION
# --------------------------------------------------

def get_recommendation(risk_level):

    if risk_level == "LOW":
        return "APPROVE"

    elif risk_level == "MEDIUM":
        return "REVIEW"

    else:
        return "REJECT"


# --------------------------------------------------
# CONVERT TIMESTAMP
#
# Supports:
# 1. ISO timestamp
#    2027-08-15T12:00:00
#
# 2. Unix timestamp
#    1786627242.6035454
# --------------------------------------------------

def convert_timestamp(value):

    if pd.isna(value):
        return pd.NaT

    try:
        # Try Unix timestamp first only for numeric values
        if isinstance(value, (int, float)):

            return pd.to_datetime(
                float(value),
                unit="s"
            )

        # Check if string contains only a number
        value_string = str(value).strip()

        try:
            numeric_value = float(value_string)

            return pd.to_datetime(
                numeric_value,
                unit="s"
            )

        except ValueError:
            pass

        # Otherwise treat as normal datetime / ISO timestamp
        return pd.to_datetime(
            value,
            errors="coerce"
        )

    except Exception:
        return pd.NaT


# --------------------------------------------------
# CALCULATE FEATURES FOR NEW TRANSACTION
# --------------------------------------------------

def calculate_transaction_features(transaction, historical_transactions):

    land_id = str(transaction.get("land_id", ""))
    seller = transaction.get("old_owner", "")
    buyer = transaction.get("new_owner", "")
    current_price = float(transaction.get("land_price", 0))

    current_time = convert_timestamp(
    transaction.get("timestamp")
)

    # -----------------------------------------
    # FILTER PREVIOUS TRANSACTIONS FOR SAME LAND
    # -----------------------------------------

    same_land_transactions = []

    for tx in historical_transactions:

        if str(tx.get("land_id", "")) == land_id:
            same_land_transactions.append(tx)

    # -----------------------------------------
    # LAND TRANSACTION COUNT
    # -----------------------------------------

    land_transaction_count = len(same_land_transactions)

    # -----------------------------------------
    # SELLER TRANSACTION COUNT
    # -----------------------------------------

    seller_transaction_count = sum(
        1
        for tx in historical_transactions
        if tx.get("old_owner") == seller
    )

    # -----------------------------------------
    # BUYER TRANSACTION COUNT
    # -----------------------------------------

    buyer_transaction_count = sum(
        1
        for tx in historical_transactions
        if tx.get("new_owner") == buyer
    )

    # -----------------------------------------
    # DOCUMENT DUPLICATE COUNT
    # -----------------------------------------

    current_document = transaction.get("document_hash", "")

    document_duplicate_count = sum(
        1
        for tx in historical_transactions
        if tx.get("document_hash") == current_document
    )

    # -----------------------------------------
    # PRICE DEVIATION
    # Compare with previous transaction of SAME LAND
    # -----------------------------------------

    price_deviation = 0.0

    if len(same_land_transactions) > 0:

        previous_tx = same_land_transactions[-1]

        previous_price = float(
            previous_tx.get("land_price", 0)
        )

        if previous_price > 0:

            price_deviation = abs(
                current_price - previous_price
            ) / previous_price

    # -----------------------------------------
    # RAPID TRANSFER FEATURE
    # -----------------------------------------

    rapid_transfer_feature = 0

    if len(same_land_transactions) > 0:

        previous_tx = same_land_transactions[-1]

        previous_timestamp = convert_timestamp(
    previous_tx.get("timestamp")
)

        if (
            pd.notna(current_time)
            and pd.notna(previous_timestamp)
        ):

            time_difference = (
                current_time - previous_timestamp
            ).total_seconds()

            # Transfer again within 24 hours
            if 0 <= time_difference <= 86400:
                rapid_transfer_feature = 1

    # -----------------------------------------
    # CREATE FEATURE DICTIONARY
    # -----------------------------------------

    features = {
        "land_price": current_price,
        "land_transaction_count": land_transaction_count,
        "document_duplicate_count": document_duplicate_count,
        "seller_transaction_count": seller_transaction_count,
        "buyer_transaction_count": buyer_transaction_count,
        "price_deviation": price_deviation,
        "rapid_transfer_feature": rapid_transfer_feature
    }

    

    return features


# --------------------------------------------------
# SHAP EXPLANATION
# --------------------------------------------------

def calculate_shap_explanation(
    model,
    updated_x,
    updated_edge_index,
    transaction_index,
    data,
    current_features
):

    feature_names = [
        "land_price",
        "land_transaction_count",
        "document_duplicate_count",
        "seller_transaction_count",
        "buyer_transaction_count",
        "price_deviation",
        "rapid_transfer_feature"
    ]

    # ----------------------------------------------
    # CREATE BACKGROUND DATA FROM EXISTING
    # TRANSACTION NODES
    # ----------------------------------------------

    transaction_nodes = data.x[data.transaction_mask]

    background_scaled = transaction_nodes[
        :, 3:9
    ].numpy()

    background_raw = data.scaler.inverse_transform(
        background_scaled
    )

    rapid_transfer = transaction_nodes[
        :, 9
    ].numpy().reshape(-1, 1)

    background = np.concatenate(
        [
            background_raw,
            rapid_transfer
        ],
        axis=1
    )

    # Use a small background set for faster SHAP
    background = background[:10]

    # ----------------------------------------------
    # PREDICTION FUNCTION FOR SHAP
    # ----------------------------------------------

    def shap_predict(X):

        predictions = []

        for row in X:

            x_copy = updated_x.clone()

            raw_features = np.array(
                [
                    0,
                    0,
                    1,
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6]
                ],
                dtype=float
            ).reshape(1, -1)

            # Apply the same scaling used by GNN
            raw_features[
                :,
                data.numerical_columns
            ] = data.scaler.transform(
                raw_features[
                    :,
                    data.numerical_columns
                ]
            )

            feature_tensor = torch.tensor(
                raw_features,
                dtype=torch.float
            )

            x_copy[
                transaction_index
            ] = feature_tensor[0]

            with torch.no_grad():

                output = model(
                    x_copy,
                    updated_edge_index
                )

                probability = F.softmax(
                    output,
                    dim=1
                )[transaction_index][1]

            predictions.append(
                probability.item()
            )

        return np.array(predictions)

    # ----------------------------------------------
    # CREATE SHAP EXPLAINER
    # ----------------------------------------------

    explainer = shap.KernelExplainer(
        shap_predict,
        background
    )

    # ----------------------------------------------
    # CALCULATE SHAP VALUES
    # ----------------------------------------------

    shap_values = explainer.shap_values(
        np.array(
            [[
                current_features["land_price"],
                current_features["land_transaction_count"],
                current_features["document_duplicate_count"],
                current_features["seller_transaction_count"],
                current_features["buyer_transaction_count"],
                current_features["price_deviation"],
                current_features["rapid_transfer_feature"]
            ]]
        ),
        nsamples=128
    )

    shap_values = np.array(shap_values)

    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    # ----------------------------------------------
    # CREATE RESULT
    # ----------------------------------------------

    shap_explanation = {}

    for name, value in zip(
        feature_names,
        shap_values
    ):

        shap_explanation[name] = round(
            float(value),
            6
        )

    return shap_explanation
# --------------------------------------------------
# ASSESS A NEW TRANSACTION
# --------------------------------------------------


def assess_transaction(
    transaction,
    historical_transactions=None
):

    # ----------------------------------------------
    # LOAD EXISTING GNN DATA
    # ----------------------------------------------

    data = prepare_gnn_data()

    print(
        "Number of features per node:",
        data.x.shape[1]
    )

    # ----------------------------------------------
    # LOAD TRAINED MODEL
    # ----------------------------------------------

    model = load_model(data)

    # ----------------------------------------------
    # CALCULATE TRANSACTION FEATURES
    # ----------------------------------------------

    features = calculate_transaction_features(
        transaction,
        historical_transactions
    )
    

    # ----------------------------------------------
    # GET CURRENT GRAPH FEATURES AND EDGES
    # ----------------------------------------------

    updated_x = data.x.clone()

    updated_edge_index = data.edge_index.clone()

    # Copy existing mapping
    node_to_index = data.node_to_index.copy()

    # ----------------------------------------------
    # HELPER FUNCTION:
    # ADD A NEW NODE
    # ----------------------------------------------

    def add_node(feature_vector):

        nonlocal updated_x

        new_index = updated_x.shape[0]

        new_feature_tensor = torch.tensor(
            [feature_vector],
            dtype=torch.float
        )

        updated_x = torch.cat(
            [
                updated_x,
                new_feature_tensor
            ],
            dim=0
        )

        return new_index

    # ----------------------------------------------
    # SELLER NODE
    # ----------------------------------------------

    seller_node = transaction["old_owner"]

    if seller_node in node_to_index:

        seller_index = node_to_index[
            seller_node
        ]

    else:

        seller_index = add_node([

            1,  # is_user
            0,  # is_land
            0,  # is_transaction

            0,
            0,
            0,
            0,
            0,
            0,
            0

        ])

        node_to_index[
            seller_node
        ] = seller_index

    # ----------------------------------------------
    # BUYER NODE
    # ----------------------------------------------

    buyer_node = transaction["new_owner"]

    if buyer_node in node_to_index:

        buyer_index = node_to_index[
            buyer_node
        ]

    else:

        buyer_index = add_node([

            1,  # is_user
            0,  # is_land
            0,  # is_transaction

            0,
            0,
            0,
            0,
            0,
            0,
            0

        ])

        node_to_index[
            buyer_node
        ] = buyer_index

    # ----------------------------------------------
    # LAND NODE
    # ----------------------------------------------

    land_node = (
        f"LAND_{transaction['land_id']}"
    )

    if land_node in node_to_index:

        land_index = node_to_index[
            land_node
        ]

    else:

        land_index = add_node([

            0,  # is_user
            1,  # is_land
            0,  # is_transaction

            0,
            0,
            0,
            0,
            0,
            0,
            0

        ])

        node_to_index[
            land_node
        ] = land_index

    # ----------------------------------------------
    # CREATE NEW TRANSACTION FEATURE VECTOR
    # ----------------------------------------------

    new_transaction_features = torch.tensor(
        [[

            0,  # is_user
            0,  # is_land
            1,  # is_transaction

            features["land_price"],

            features[
                "land_transaction_count"
            ],

            features[
                "document_duplicate_count"
            ],

            features[
                "seller_transaction_count"
            ],

            features[
                "buyer_transaction_count"
            ],

            features[
                "price_deviation"
            ],

            features[
                "rapid_transfer_feature"
            ]

        ]],
        dtype=torch.float
    )

    # ----------------------------------------------
    # APPLY SAME SCALING
    # ----------------------------------------------

    new_features_numpy = (
        new_transaction_features.numpy()
    )

    new_features_numpy[
        :,
        data.numerical_columns
    ] = data.scaler.transform(

        new_features_numpy[
            :,
            data.numerical_columns
        ]
    )

    new_transaction_features = torch.tensor(
        new_features_numpy,
        dtype=torch.float
    )

    # ----------------------------------------------
    # ADD TRANSACTION NODE
    # ----------------------------------------------

    transaction_index = updated_x.shape[0]

    updated_x = torch.cat(
        [
            updated_x,
            new_transaction_features
        ],
        dim=0
    )

    # ----------------------------------------------
    # CREATE GRAPH EDGES
    #
    # Seller ↔ Transaction
    # Buyer  ↔ Transaction
    # Land   ↔ Transaction
    # ----------------------------------------------

    new_edges = torch.tensor(
        [

            [
                seller_index,
                transaction_index
            ],

            [
                transaction_index,
                seller_index
            ],

            [
                buyer_index,
                transaction_index
            ],

            [
                transaction_index,
                buyer_index
            ],

            [
                land_index,
                transaction_index
            ],

            [
                transaction_index,
                land_index
            ]

        ],
        dtype=torch.long

    ).t().contiguous()

    # ----------------------------------------------
    # ADD EDGES TO GRAPH
    # ----------------------------------------------

    updated_edge_index = torch.cat(
        [
            updated_edge_index,
            new_edges
        ],
        dim=1
    )

    # ----------------------------------------------
    # MODEL PREDICTION
    # ----------------------------------------------

    model.eval()

    with torch.no_grad():

        output = model(
            updated_x,
            updated_edge_index
        )

        probabilities = F.softmax(
            output,
            dim=1
        )

    # ----------------------------------------------
    # GET FRAUD PROBABILITY
    # ----------------------------------------------

    fraud_probability = (

        probabilities[
            transaction_index
        ][1].item()
    )

    # ----------------------------------------------
    # RISK SCORE
    # ----------------------------------------------

    risk_score = round(
        fraud_probability * 100,
        2
    )

    # ----------------------------------------------
    # RISK LEVEL
    # ----------------------------------------------

    risk_level = get_risk_level(
        risk_score
    )

    # ----------------------------------------------
    # RECOMMENDATION
    # ----------------------------------------------

    recommendation = get_recommendation(
        risk_level
    )

    
    shap_explanation = calculate_shap_explanation(
    model,
    updated_x,
    updated_edge_index,
    transaction_index,
    data,
    features
)
    # ----------------------------------------------
    # RETURN RESULT
    # ----------------------------------------------

    return {

        "fraud_probability": round(
            fraud_probability,
            4
        ),

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "recommendation":
            recommendation,

        "features":
            features,


        "shap_explanation":
            shap_explanation
    }


# --------------------------------------------------
# TEST NEW TRANSACTION
# --------------------------------------------------

if __name__ == "__main__":

    dataset_path = "Dataset/land_transactions.csv"

    df = pd.read_csv(dataset_path)

    # Convert dataset rows into transaction dictionaries
    historical_transactions = df.to_dict("records")

    print(f"\nHistorical transactions available: {len(historical_transactions)}")


    test_transaction = {
    "land_id": "1844",
    "old_owner": "User_469",
    "new_owner": "User_999",
    "land_price": 20000000,
    "document_hash": "e9453c54eacf126ddaf57c571c000c8daf04ce251980dbcb16a2aaa8d2a451ac",
    "timestamp": "2026-10-02T05:10:00"
}

    result = assess_transaction(
        test_transaction,
        historical_transactions
    )

    print(
        "\n========== NEW TRANSACTION RISK ASSESSMENT =========="
    )

    print(
        "\nFraud Probability:",
        result["fraud_probability"]
    )

    print(
        "Risk Score:",
        f"{result['risk_score']}/100"
    )

    print(
        "Risk Level:",
        result["risk_level"]
    )

    print(
        "Recommendation:",
        result["recommendation"]
    )

    print(
        "\n========== FEATURES USED =========="
    )

    for key, value in result[
        "features"
    ].items():

        print(
            f"{key}: {value}"
        )
    print("\n========== SHAP EXPLANATION ==========")

    for feature, shap_value in result["shap_explanation"].items():
        print(f"{feature}: {shap_value:+.6f}")

    print("======================================")
        