import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


CSV_PATH = "sysmon_normalized_full.csv"
SEQ_LEN = 7
STRIDE = 1
OUTPUT_DIR = "./"

def encode_categoricals(df, exclude=["label"]):
    df_encoded = df.copy()
    for col in df.columns:
        if df[col].dtype == object and col not in exclude:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
    return df_encoded

# LOAD
print("[*] Reading CSV...")
df = pd.read_csv(CSV_PATH, low_memory=False)
df = df[df["label"].isin([0, 1])]  # Keep binary

print(f"[+] Loaded {len(df)} events")
print("[+] Individual event label distribution:")
print(df["label"].value_counts())

# ENCODE
df_encoded = encode_categoricals(df)
features = df_encoded.drop(columns=["label"]).values
labels = df_encoded["label"].values

# SLIDING WINDOW
print("[*] Generating sequences...")
X = []
y = []

for i in range(0, len(features) - SEQ_LEN, STRIDE):
    seq = features[i:i+SEQ_LEN]
    seq_labels = labels[i:i+SEQ_LEN]
    label = int(seq_labels.mean() > 0.1)
    X.append(seq)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Handle NaNs
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Check class balance
unique, counts = np.unique(y, return_counts=True)
print(f"[+] Sequence class distribution: {dict(zip(unique, counts))}")

# Split
print("[*] Stratified train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Save
np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

print(f"[✓] Saved {len(X_train)} train sequences and {len(X_test)} test sequences")