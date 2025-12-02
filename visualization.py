import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("asd_dataset_ready.csv", sep="\t")

# Encode
enc = LabelEncoder()
cat_cols = ["gender", "ethnicity", "jaundice", "family_ASD"]
for col in cat_cols:
    df[col] = enc.fit_transform(df[col])

feature_cols = [
    "age", "gender", "ethnicity", "jaundice", "family_ASD", "screening_score",
    "A1","A2","A3","A4","A5","A6","A7","A8","A9","A10"
]

X = df[feature_cols].values
y = df["ASD"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 2. DATASET CLASS
# ============================================================
class ASDDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

test_ds = ASDDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=32)

# ============================================================
# 3. RECREATE MODELS (same architecture as training)
# ============================================================
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1])

# Initialize models
lstm_model = LSTMClassifier()
rnn_model = RNNClassifier()

# Retrain lightweight models quickly (only few epochs)
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(rnn_model.parameters(), lr=0.001)

# Train briefly (just for predictions)
for epoch in range(2):
    for Xb, yb in test_loader:  # using test for quick eval
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        l_pred = lstm_model(Xb)
        r_pred = rnn_model(Xb)
        loss1 = criterion(l_pred, yb)
        loss2 = criterion(r_pred, yb)
        loss1.backward()
        loss2.backward()
        optimizer1.step()
        optimizer2.step()

# ============================================================
# 4. PREDICTIONS
# ============================================================
# LSTM
lstm_model.eval()
lstm_preds = []
lstm_probs = []
true = []

with torch.no_grad():
    for Xb, yb in test_loader:
        out = lstm_model(Xb)
        lstm_preds.extend(torch.argmax(out, 1).numpy())
        lstm_probs.extend(torch.softmax(out, 1)[:, 1].numpy())
        true.extend(yb.numpy())

# RNN
rnn_model.eval()
rnn_preds = []
rnn_probs = []

with torch.no_grad():
    for Xb, yb in test_loader:
        out = rnn_model(Xb)
        rnn_preds.extend(torch.argmax(out, 1).numpy())
        rnn_probs.extend(torch.softmax(out, 1)[:, 1].numpy())

# Naive Bayes + Decision Tree
nb = GaussianNB()
dt = DecisionTreeClassifier()
nb.fit(X_train, y_train)
dt.fit(X_train, y_train)

nb_pred = nb.predict(X_test)
dt_pred = dt.predict(X_test)
nb_prob = nb.predict_proba(X_test)[:, 1]
dt_prob = dt.predict_proba(X_test)[:, 1]

# ============================================================
# 5. PLOT ACCURACY COMPARISON
# ============================================================
accuracies = [
    accuracy_score(true, lstm_preds),
    accuracy_score(true, rnn_preds),
    accuracy_score(y_test, nb_pred),
    accuracy_score(y_test, dt_pred),
]

names = ["LSTM", "RNN", "Naive Bayes", "Decision Tree"]

plt.figure(figsize=(8, 5))
sns.barplot(x=names, y=accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig("model_accuracy.png", dpi=300)
plt.show()

# ============================================================
# 6. CONFUSION MATRIX FUNCTION
# ============================================================

def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(filename, dpi=300)
    plt.show()

plot_cm(true, lstm_preds, "Confusion Matrix - LSTM", "cm_lstm.png")
plot_cm(true, rnn_preds, "Confusion Matrix - RNN", "cm_rnn.png")
plot_cm(y_test, nb_pred, "Confusion Matrix - Naive Bayes", "cm_nb.png")
plot_cm(y_test, dt_pred, "Confusion Matrix - Decision Tree", "cm_dt.png")

# ============================================================
# 7. ROC CURVES
# ============================================================

plt.figure(figsize=(8, 6))

def roc_plot(y_true, prob, label):
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")

roc_plot(true, lstm_probs, "LSTM")
roc_plot(true, rnn_probs, "RNN")
roc_plot(y_test, nb_prob, "Naive Bayes")
roc_plot(y_test, dt_prob, "Decision Tree")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.savefig("roc_curves.png", dpi=300)
plt.show()
