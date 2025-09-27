# Internship Logistic Regression - Complete Program

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\Admin\Downloads\data (1).csv")  # <-- replace with your CSV path

print("First 5 rows of your dataset:")
print(df.head())

# -----------------------------
# 2. Drop useless columns
# -----------------------------
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# -----------------------------
# 3. Define features and target
# -----------------------------
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"].map({"M": 1, "B": 0})  # M=1, B=0

# -----------------------------
# 4. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Standardize features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 6. Train Logistic Regression model
# -----------------------------
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# -----------------------------
# 7. Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probabilities for class 1

# -----------------------------
# 8. Evaluation
# -----------------------------
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# -----------------------------
# 9. ROC Curve Plot
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc_score:.3f})')
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# -----------------------------
# 10. Sigmoid Function Plot
# -----------------------------
z_vals = np.linspace(-10, 10, 200)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

plt.figure(figsize=(5,4))
plt.plot(z_vals, sigmoid(z_vals))
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.grid(True)
plt.show()