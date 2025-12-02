import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# ============================================================
# 1. LOAD DATA (IMPORTANT: TAB-SEPARATED)
# ============================================================
print("Loading Dataset...")
df = pd.read_csv("asd_dataset_ready.csv", sep="\t")
print("Columns Loaded:", df.columns.tolist())

# ============================================================
# 2. FIX COLUMN NAMES IF NEEDED
# ============================================================
expected_cols = [
    "age", "gender", "ethnicity", "jaundice", "family_ASD",
    "screening_score",
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
    "ASD"
]

# If merged column exists, split it
if len(df.columns) == 1:
    print("Fixing merged header...")
    df = df[df.columns[0]].str.split("\t", expand=True)
    df.columns = expected_cols

# ============================================================
# 3. ENCODE CATEGORICAL VALUES
# ============================================================
enc = LabelEncoder()

cat_cols = ["gender", "ethnicity", "jaundice", "family_ASD"]
for col in cat_cols:
    df[col] = enc.fit_transform(df[col])

# ============================================================
# 4. FEATURES + LABEL
# ============================================================
feature_cols = [
    "age", "gender", "ethnicity", "jaundice", "family_ASD", "screening_score",
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"
]

X = df[feature_cols].values
y = df["ASD"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 5. PYTORCH DATASET
# ============================================================
class ASDDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # LSTM expects 3D: (seq_len, features)
        return self.X[idx].unsqueeze(0), self.y[idx]

train_ds = ASDDataset(X_train, y_train)
test_ds = ASDDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# ============================================================
# 6. LSTM MODEL
# ============================================================
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

lstm_model = LSTMClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

# ============================================================
# 7. TRAIN LSTM
# ============================================================
print("Training LSTM...")
for epoch in range(10):
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        preds = lstm_model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# LSTM Accuracy
lstm_model.eval()
preds = []
true = []
with torch.no_grad():
    for Xb, yb in test_loader:
        out = lstm_model(Xb)
        preds.extend(torch.argmax(out, 1).numpy())
        true.extend(yb.numpy())

print("\nLSTM Accuracy:", accuracy_score(true, preds))

# ============================================================
# 8. SIMPLE RNN MODEL
# ============================================================
class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1])

rnn_model = RNNClassifier()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)

# Train RNN
print("\nTraining RNN...")
for epoch in range(10):
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        preds = rnn_model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# RNN Accuracy
rnn_model.eval()
preds = []
true = []
with torch.no_grad():
    for Xb, yb in test_loader:
        out = rnn_model(Xb)
        preds.extend(torch.argmax(out, 1).numpy())
        true.extend(yb.numpy())

print("RNN Accuracy:", accuracy_score(true, preds))

# ============================================================
# 9. NAIVE BAYES + DECISION TREE
# ============================================================
nb = GaussianNB()
dt = DecisionTreeClassifier()

nb.fit(X_train, y_train)
dt.fit(X_train, y_train)

nb_pred = nb.predict(X_test)
dt_pred = dt.predict(X_test)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
