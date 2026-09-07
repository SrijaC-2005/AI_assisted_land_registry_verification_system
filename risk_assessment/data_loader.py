import json
import os


# Path to your existing blockchain ledger
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEDGER_FILE = os.path.join(
    BASE_DIR,
    "Blockchain_Ledger.json"
)


def load_transactions():
    """
    Read all valid land transactions
    from the blockchain ledger.
    """

    if not os.path.exists(LEDGER_FILE):
        print("❌ Ledger file not found:", LEDGER_FILE)
        return []

    with open(LEDGER_FILE, "r") as f:
        data = json.load(f)

    transactions = []

    # Go through every block
    for block in data.get("chain", []):

        # Get transactions inside the block
        for tx in block.get("transactions", []):

            # Ignore Genesis Block
            if "land_id" not in tx:
                continue

            transactions.append({
                "land_id": tx.get("land_id"),
                "old_owner": tx.get("old_owner"),
                "new_owner": tx.get("new_owner"),
                "land_price": tx.get("land_price"),
                "document_hash": tx.get("document_hash"),
                "timestamp": tx.get("timestamp"),
                "transaction_id": tx.get("id"),
                "block_index": block.get("index")
            })

    return transactions


if __name__ == "__main__":

    transactions = load_transactions()

    print("\nTotal transactions found:", len(transactions))

    for tx in transactions:
        print(tx)