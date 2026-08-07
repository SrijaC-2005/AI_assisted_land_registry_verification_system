import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Dataset/Network_Metrics_Dataset.csv")
print(df.columns)
print()
print(df.dtypes)
print("Dataset Shape:", df.shape)
print(df.head())

# -----------------------------
# Remove unnecessary columns
# -----------------------------
X = df.drop(columns=["row_id", "timestamp", "consensus","notes"])

print(X.dtypes)
print("\nUnique values in notes:")
print(df["notes"].unique()[:20])

y = df["consensus"]

# -----------------------------
# Encode target labels
# -----------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = SGDClassifier(random_state=42)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nAccuracy:", accuracy)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model,
            "FL_SGD/federated_sgd_global_model.pkl")

joblib.dump(encoder,
            "FL_SGD/fed_label_encoder.pkl")

joblib.dump(scaler,
            "FL_SGD/fed_scaler.pkl")

print("\n✅ Model Saved Successfully")