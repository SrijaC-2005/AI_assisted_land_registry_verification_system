import pandas as pd


FILE_PATH = "Dataset/land_transactions.csv"


# Load dataset
df = pd.read_csv(FILE_PATH)


print("\n========== DATASET INFORMATION ==========")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


print("\n========== FIRST 10 ROWS ==========")
print(df.head(10))


print("\n========== FRAUD DISTRIBUTION ==========")
print(df["is_fraud"].value_counts())


print("\n========== FRAUD TYPE DISTRIBUTION ==========")
print(df["fraud_type"].value_counts())


print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


print("\n========== SAMPLE FRAUD TRANSACTIONS ==========")

fraud_transactions = df[df["is_fraud"] == 1]

print(
    fraud_transactions[
        [
            "land_id",
            "old_owner",
            "new_owner",
            "land_price",
            "document_hash",
            "fraud_type"
        ]
    ].head(10)
)