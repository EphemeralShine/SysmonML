import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

def encode_categoricals(df, exclude=["label"], one_hot=["EventType"]):
    df_encoded = df.copy()
    label_encoders = {}

    for col in df.columns:
        if col in exclude:
            continue
        if col in one_hot:
            # Use one-hot encoding for specified column
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
        elif df[col].dtype == "object":
            # Label encode other categorical columns
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df_encoded, label_encoders


# Load
print("Loading and preprocessing data...")
df = pd.read_csv("sysmon_normalized_full.csv", low_memory=False)
df = df[df["label"].isin([0, 1])]

# Drop volatile fields
drop_columns = [
    "ProcessId", "ParentProcessId", "ProcessGuid", "ParentProcessGuid",
    "LogonGuid", "LogonId", "EventRecordID",
    "SourceProcessId", "TargetProcessId",
    "SourceProcessGuid", "TargetProcessGuid",
    "NewThreadId", "UtcTime", "CreationUtcTime", "TerminalSessionId" # "Image", "DestinationHostname", "ParentImage", "DestinationIp", "TargetFilename", "SourcePort", "QueryName", "QueryResults"
]

df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

# Encode Categoricals
df_encoded, encoders = encode_categoricals(df)
X = df_encoded.drop(columns=["label"])
y = df_encoded["label"]

# Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\nEvaluation Results:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Feature Importance
print("Computing feature importances...")
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop Features by Importance:")
print(importance_df.to_string(index=False))

# Retrain on Top-k Features 
top_k = 6
top_features = importance_df["Feature"].head(top_k).tolist()
print(f"Retraining using top {top_k} features: {top_features}")

# Subset the feature matrix
X_top = df_encoded[top_features]
X_top_train, X_top_test, y_top_train, y_top_test = train_test_split(
    X_top, y, test_size=0.3, stratify=y, random_state=42
)

model_top = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model_top.fit(X_top_train, y_top_train)

y_top_pred = model_top.predict(X_top_test)

# Re-evaluate
acc_top = accuracy_score(y_top_test, y_top_pred)
prec_top = precision_score(y_top_test, y_top_pred, zero_division=0)
rec_top = recall_score(y_top_test, y_top_pred, zero_division=0)
f1_top = f1_score(y_top_test, y_top_pred, zero_division=0)

print(f"\nTop-{top_k} Features Evaluation:")
print(f"Accuracy : {acc_top:.4f}")
print(f"Precision: {prec_top:.4f}")
print(f"Recall   : {rec_top:.4f}")
print(f"F1 Score : {f1_top:.4f}")
# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.gca().invert_yaxis()

# Summary Table
summary = pd.DataFrame({
    "Model": ["Random Forest (All Features)", "Random Forest (Top 6 Features)"],
    "Accuracy": [accuracy, acc_top],
    "Precision": [precision, prec_top],
    "Recall": [recall, rec_top],
    "F1 Score": [f1, f1_top]
})

print("\nEvaluation Summary:")
print(summary.to_string(index=False))

summary.set_index("Model")[["Precision", "Recall", "F1 Score"]].plot(
    kind="bar", figsize=(8, 5), ylim=(0, 1), legend=True, title="Model Comparison"
)
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
