import csv
import random
import hashlib
import uuid
from datetime import datetime, timedelta


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

NUM_TRANSACTIONS = 5000
OUTPUT_FILE = "Dataset/land_transactions.csv"

NUM_USERS = 500
NUM_LANDS = 2000

users = [f"User_{i}" for i in range(1, NUM_USERS + 1)]
lands = [str(1000 + i) for i in range(NUM_LANDS)]


# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------

def generate_hash():
    """Generate a random document hash."""
    random_value = str(uuid.uuid4())
    return hashlib.sha256(random_value.encode()).hexdigest()


def generate_price():
    """Generate a normal land price."""
    return random.randint(300000, 5000000)


def generate_timestamp(base_time):
    """Generate a timestamp."""
    seconds = random.randint(60, 86400 * 365)
    return base_time + timedelta(seconds=seconds)


# -------------------------------------------------
# STORE EXISTING INFORMATION
# -------------------------------------------------

land_owner = {}
land_document = {}
land_price_history = {}

for land in lands:
    land_owner[land] = random.choice(users)
    land_document[land] = generate_hash()
    land_price_history[land] = generate_price()


# -------------------------------------------------
# DATA GENERATION
# -------------------------------------------------

transactions = []

base_time = datetime.now()


for i in range(NUM_TRANSACTIONS):

    transaction_id = str(uuid.uuid4())

    # Approximately 80% normal
    # Approximately 20% suspicious
    is_fraud = 1 if random.random() < 0.20 else 0

    fraud_type = "normal"

    # =================================================
    # NORMAL TRANSACTION
    # =================================================

    if is_fraud == 0:

        land_id = random.choice(lands)

        old_owner = land_owner[land_id]

        possible_buyers = [u for u in users if u != old_owner]
        new_owner = random.choice(possible_buyers)

        previous_price = land_price_history[land_id]

        # Normal price variation
        land_price = int(
            previous_price * random.uniform(0.8, 1.2)
        )

        document_hash = land_document[land_id]

        # Update ownership
        land_owner[land_id] = new_owner
        land_price_history[land_id] = land_price

    # =================================================
    # FRAUD PATTERN 1
    # DUPLICATE LAND SALE
    # =================================================

    elif random.random() < 0.25:

        fraud_type = "duplicate_land"

        land_id = random.choice(lands)

        # Use a random seller instead of actual owner
        old_owner = random.choice(users)

        new_owner = random.choice(
            [u for u in users if u != old_owner]
        )

        land_price = generate_price()

        document_hash = generate_hash()


    # =================================================
    # FRAUD PATTERN 2
    # DUPLICATE DOCUMENT
    # =================================================

    elif random.random() < 0.50:

        fraud_type = "duplicate_document"

        land_id = random.choice(lands)

        old_owner = land_owner[land_id]

        new_owner = random.choice(
            [u for u in users if u != old_owner]
        )

        land_price = generate_price()

        # Take document from another land
        other_land = random.choice(
            [l for l in lands if l != land_id]
        )

        document_hash = land_document[other_land]


    # =================================================
    # FRAUD PATTERN 3
    # ABNORMAL PRICE
    # =================================================

    elif random.random() < 0.75:

        fraud_type = "abnormal_price"

        land_id = random.choice(lands)

        old_owner = land_owner[land_id]

        new_owner = random.choice(
            [u for u in users if u != old_owner]
        )

        previous_price = land_price_history[land_id]

        # Extremely low or high price
        if random.random() < 0.5:
            land_price = int(
                previous_price * random.uniform(0.01, 0.20)
            )
        else:
            land_price = int(
                previous_price * random.uniform(3, 10)
            )

        document_hash = land_document[land_id]


    # =================================================
    # FRAUD PATTERN 4
    # RAPID TRANSFER
    # =================================================

    else:

        fraud_type = "rapid_transfer"

        land_id = random.choice(lands)

        old_owner = land_owner[land_id]

        new_owner = random.choice(
            [u for u in users if u != old_owner]
        )

        land_price = land_price_history[land_id]

        document_hash = land_document[land_id]

        # Update ownership
        land_owner[land_id] = new_owner


    timestamp = generate_timestamp(base_time)

    transaction = {
        "transaction_id": transaction_id,
        "land_id": land_id,
        "old_owner": old_owner,
        "new_owner": new_owner,
        "land_price": land_price,
        "document_hash": document_hash,
        "timestamp": timestamp.isoformat(),
        "is_fraud": is_fraud,
        "fraud_type": fraud_type
    }

    transactions.append(transaction)


# -------------------------------------------------
# SAVE DATASET
# -------------------------------------------------

fieldnames = [
    "transaction_id",
    "land_id",
    "old_owner",
    "new_owner",
    "land_price",
    "document_hash",
    "timestamp",
    "is_fraud",
    "fraud_type"
]

with open(
    OUTPUT_FILE,
    mode="w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for transaction in transactions:
        writer.writerow(transaction)


# -------------------------------------------------
# SUMMARY
# -------------------------------------------------

fraud_count = sum(
    tx["is_fraud"]
    for tx in transactions
)

normal_count = NUM_TRANSACTIONS - fraud_count


print("\n========== DATASET GENERATED ==========")

print(f"Total Transactions : {NUM_TRANSACTIONS}")
print(f"Normal Transactions: {normal_count}")
print(f"Fraud Transactions : {fraud_count}")

print("\nFraud Type Distribution:")

fraud_types = {}

for tx in transactions:
    fraud_type = tx["fraud_type"]

    fraud_types[fraud_type] = (
        fraud_types.get(fraud_type, 0) + 1
    )

for fraud_type, count in fraud_types.items():
    print(f"{fraud_type}: {count}")

print(f"\nDataset saved to: {OUTPUT_FILE}")