import torch
from torch_geometric.data import Data

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from risk_assessment.graph_builder import build_training_graph


def prepare_gnn_data():

    # --------------------------------------------------
    # BUILD GRAPH AND LOAD DATASET
    # --------------------------------------------------

    graph, df = build_training_graph()

    print("\nPreparing improved graph features...")

    # Make a copy
    df = df.copy()

    # --------------------------------------------------
    # FEATURE 1: LAND TRANSACTION FREQUENCY
    # --------------------------------------------------

    df["land_transaction_count"] = (
        df.groupby("land_id")["land_id"]
        .transform("count")
    )

    # --------------------------------------------------
    # FEATURE 2: DOCUMENT DUPLICATE COUNT
    # --------------------------------------------------

    df["document_duplicate_count"] = (
        df.groupby("document_hash")["document_hash"]
        .transform("count")
    )

    # --------------------------------------------------
    # FEATURE 3: SELLER TRANSACTION COUNT
    # --------------------------------------------------

    df["seller_transaction_count"] = (
        df.groupby("old_owner")["old_owner"]
        .transform("count")
    )

    # --------------------------------------------------
    # FEATURE 4: BUYER TRANSACTION COUNT
    # --------------------------------------------------

    df["buyer_transaction_count"] = (
        df.groupby("new_owner")["new_owner"]
        .transform("count")
    )

    # --------------------------------------------------
    # FEATURE 5: PRICE DEVIATION
    #
    # Compare transaction price with the average price
    # for that particular land.
    # --------------------------------------------------

    df["land_average_price"] = (
        df.groupby("land_id")["land_price"]
        .transform("mean")
    )

    df["price_deviation"] = (
        abs(
            df["land_price"] -
            df["land_average_price"]
        )
        /
        (df["land_average_price"] + 1)
    )

    # --------------------------------------------------
    # FEATURE 6: RAPID TRANSFER FEATURE
    #
    # Sort transactions based on land and time.
    # Calculate time gap between consecutive
    # transactions of the same land.
    # --------------------------------------------------

    df["timestamp_dt"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        by=["land_id", "timestamp_dt"]
    ).reset_index(drop=True)

    df["previous_transaction_time"] = (
        df.groupby("land_id")["timestamp_dt"]
        .shift(1)
    )

    df["time_difference_hours"] = (
        (
            df["timestamp_dt"] -
            df["previous_transaction_time"]
        )
        .dt.total_seconds()
        / 3600
    )

    # If no previous transaction exists,
    # give a large value
    df["time_difference_hours"] = (
        df["time_difference_hours"]
        .fillna(999999)
    )

    # Rapid transfer if same land is transferred
    # within 24 hours
    df["rapid_transfer_feature"] = (
        df["time_difference_hours"] < 24
    ).astype(int)

    # --------------------------------------------------
    # CREATE LOOKUP USING TRANSACTION ID
    #
    # graph_builder uses transaction_id to create
    # transaction nodes.
    # --------------------------------------------------

    transaction_features = {}

    for _, row in df.iterrows():

        transaction_features[
            row["transaction_id"]
        ] = {
            "land_price": row["land_price"],
            "land_transaction_count":
                row["land_transaction_count"],

            "document_duplicate_count":
                row["document_duplicate_count"],

            "seller_transaction_count":
                row["seller_transaction_count"],

            "buyer_transaction_count":
                row["buyer_transaction_count"],

            "price_deviation":
                row["price_deviation"],

            "rapid_transfer_feature":
                row["rapid_transfer_feature"],

            "is_fraud":
                row["is_fraud"]
        }

    # --------------------------------------------------
    # CREATE NODE → INDEX MAPPING
    # --------------------------------------------------

    node_list = list(graph.nodes())

    node_to_index = {
        node: index
        for index, node in enumerate(node_list)
    }

    # --------------------------------------------------
    # CREATE NODE FEATURES
    #
    # Feature format:
    #
    # [is_user,
    #  is_land,
    #  is_transaction,
    #  land_price,
    #  land_transaction_count,
    #  document_duplicate_count,
    #  seller_transaction_count,
    #  buyer_transaction_count,
    #  price_deviation,
    #  rapid_transfer_feature]
    # --------------------------------------------------

    features = []

    labels = []

    for node in node_list:

        node_data = graph.nodes[node]

        node_type = node_data["node_type"]

        # ----------------------------------------------
        # USER NODE
        # ----------------------------------------------

        if node_type == "user":

            features.append([
                1,  # is_user
                0,  # is_land
                0,  # is_transaction
                0,  # land_price
                0,  # land_transaction_count
                0,  # document_duplicate_count
                0,  # seller_transaction_count
                0,  # buyer_transaction_count
                0,  # price_deviation
                0   # rapid_transfer
            ])

            labels.append(-1)

        # ----------------------------------------------
        # LAND NODE
        # ----------------------------------------------

        elif node_type == "land":

            features.append([
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ])

            labels.append(-1)

        # ----------------------------------------------
        # TRANSACTION NODE
        # ----------------------------------------------

        elif node_type == "transaction":

            # Remove TX_ prefix
            transaction_id = node.replace(
                "TX_",
                "",
                1
            )

            tx = transaction_features[
                transaction_id
            ]

            features.append([
                0,  # is_user
                0,  # is_land
                1,  # is_transaction

                tx["land_price"],

                tx["land_transaction_count"],

                tx["document_duplicate_count"],

                tx["seller_transaction_count"],

                tx["buyer_transaction_count"],

                tx["price_deviation"],

                tx["rapid_transfer_feature"]
            ])

            labels.append(
                tx["is_fraud"]
            )

    # --------------------------------------------------
    # NORMALIZE NUMERICAL FEATURES
    # --------------------------------------------------

    features = np.array(
        features,
        dtype=float
    )

    numerical_columns = [
        3,  # land_price
        4,  # land_transaction_count
        5,  # document_duplicate_count
        6,  # seller_transaction_count
        7,  # buyer_transaction_count
        8   # price_deviation
    ]

    scaler = StandardScaler()

    features[:, numerical_columns] = (
        scaler.fit_transform(
            features[:, numerical_columns]
        )
    )

    # --------------------------------------------------
    # CONVERT TO PYTORCH TENSORS
    # --------------------------------------------------

    x = torch.tensor(
        features,
        dtype=torch.float
    )

    y = torch.tensor(
        labels,
        dtype=torch.long
    )

    # --------------------------------------------------
    # CREATE BIDIRECTIONAL EDGES
    # --------------------------------------------------

    edge_list = []

    for source, target in graph.edges():

        source_index = node_to_index[source]

        target_index = node_to_index[target]

        # Forward edge
        edge_list.append([
            source_index,
            target_index
        ])

        # Reverse edge
        edge_list.append([
            target_index,
            source_index
        ])

    edge_index = torch.tensor(
        edge_list,
        dtype=torch.long
    ).t().contiguous()

    # --------------------------------------------------
    # CREATE PYTORCH GEOMETRIC DATA
    # --------------------------------------------------

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    # --------------------------------------------------
    # MASK FOR TRANSACTION NODES
    # --------------------------------------------------

    data.transaction_mask = (
        data.y != -1
    )

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------

    print("\n========== IMPROVED GNN DATA ==========")

    print(
        "Node Feature Matrix:",
        data.x.shape
    )

    print(
        "Edge Index:",
        data.edge_index.shape
    )

    print(
        "Transaction Nodes:",
        data.transaction_mask.sum().item()
    )

    print(
        "Normal:",
        (
            data.y[data.transaction_mask] == 0
        ).sum().item()
    )

    print(
        "Fraud:",
        (
            data.y[data.transaction_mask] == 1
        ).sum().item()
    )

    print("\nNumber of features per node: 10")
    data.scaler = scaler
    data.numerical_columns = numerical_columns
    data.node_list = node_list
    data.node_to_index = node_to_index
    return data


if __name__ == "__main__":

    data = prepare_gnn_data()