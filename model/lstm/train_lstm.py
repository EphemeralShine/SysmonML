import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from lstm_model import SysmonLSTM

# HYPERPARAMETERS
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD DATA FROM FILES (Already Split)
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print(f"[*] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# CONVERT TO TENSORS
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# MODEL SETUP
input_size = X_train.shape[2]
model = SysmonLSTM(input_size).to(DEVICE)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# TRAIN
print(f"\n[*] Training on {DEVICE}...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(train_dataset)
    print(f"[Epoch {epoch+1:2d}/{EPOCHS}] Loss: {avg_loss:.4f}")

# SAVE MODEL
torch.save(model.state_dict(), "lstm_sysmon_model.pt")
print("[✓] Model saved to lstm_sysmon_model.pt")

# EVALUATE
print("\n[*] Evaluating on test set...")
model.eval()
y_preds, y_trues = [], []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(DEVICE)
        probs = model(batch_x).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        y_preds.extend(preds)
        y_trues.extend(batch_y.numpy().astype(int))

accuracy = accuracy_score(y_trues, y_preds)
precision = precision_score(y_trues, y_preds, zero_division=0)
recall = recall_score(y_trues, y_preds, zero_division=0)
f1 = f1_score(y_trues, y_preds, zero_division=0)

print("\n[✓] LSTM Evaluation Results:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n[✓] Classification Report:")
print(classification_report(y_trues, y_preds, zero_division=0))
