import pandas as pd
import networkx as nx
import os


# ---------------------------------------------
# DATASET PATH
# ---------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_FILE = os.path.join(
    BASE_DIR,
    "Dataset",
    "land_transactions.csv"
)


def build_training_graph():

    # ---------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------

    df = pd.read_csv(DATASET_FILE)

    print(f"\nTotal transactions loaded: {len(df)}")

    # Create graph
    graph = nx.Graph()

    # ---------------------------------------------
    # CREATE GRAPH
    # ---------------------------------------------

    for _, tx in df.iterrows():

        # Prefixes prevent name conflicts
        seller = f"USER_{tx['old_owner']}"
        buyer = f"USER_{tx['new_owner']}"
        land = f"LAND_{tx['land_id']}"
        transaction = f"TX_{tx['transaction_id']}"

        # -----------------------------------------
        # ADD USER NODES
        # -----------------------------------------

        graph.add_node(
            seller,
            node_type="user"
        )

        graph.add_node(
            buyer,
            node_type="user"
        )

        # -----------------------------------------
        # ADD LAND NODE
        # -----------------------------------------

        graph.add_node(
            land,
            node_type="land"
        )

        # -----------------------------------------
        # ADD TRANSACTION NODE
        # -----------------------------------------

        graph.add_node(
            transaction,
            node_type="transaction",
            land_price=float(tx["land_price"]),
            timestamp=tx["timestamp"],
            document_hash=tx["document_hash"],
            is_fraud=int(tx["is_fraud"]),
            fraud_type=tx["fraud_type"]
        )

        # -----------------------------------------
        # CREATE RELATIONSHIPS
        # -----------------------------------------

        # Seller -> Transaction
        graph.add_edge(
            seller,
            transaction,
            relationship="SELLS"
        )

        # Buyer -> Transaction
        graph.add_edge(
            buyer,
            transaction,
            relationship="BUYS"
        )

        # Land -> Transaction
        graph.add_edge(
            land,
            transaction,
            relationship="INVOLVES_LAND"
        )

    return graph, df


# ---------------------------------------------
# TEST GRAPH
# ---------------------------------------------

if __name__ == "__main__":

    graph, df = build_training_graph()

    print("\n========== TRAINING GRAPH ==========")

    print("\nTotal Nodes:", graph.number_of_nodes())
    print("Total Edges:", graph.number_of_edges())

    # Count node types
    user_count = 0
    land_count = 0
    transaction_count = 0

    for _, data in graph.nodes(data=True):

        if data["node_type"] == "user":
            user_count += 1

        elif data["node_type"] == "land":
            land_count += 1

        elif data["node_type"] == "transaction":
            transaction_count += 1

    print("\nNode Distribution:")

    print("Users:", user_count)
    print("Lands:", land_count)
    print("Transactions:", transaction_count)

    print("\nFraud Distribution:")

    print(df["is_fraud"].value_counts())

    print("\n========== SAMPLE TRANSACTION NODES ==========")

    count = 0

    for node, data in graph.nodes(data=True):

        if data["node_type"] == "transaction":

            print(node)
            print(data)
            print()

            count += 1

            if count == 3:
                break