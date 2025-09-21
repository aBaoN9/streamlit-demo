"""
Evaluate saved logistic regression model on processed test set
and export metrics + figures for the dashboard.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORT_FIG = ROOT / "report" / "figures"
REPORT_FIG.mkdir(parents=True, exist_ok=True)

# Load processed test
X_test = pd.read_csv(PROC / "X_test.csv", index_col=0)
y_test = pd.read_csv(PROC / "y_test.csv", index_col=0)["status"]

# Load model trained on processed (from train.py step 2)
model_path = MODELS / "model_latest.pkl"
if not model_path.exists():
    raise FileNotFoundError(f"Không thấy {model_path}. Chạy: python src/train.py trước.")

model = joblib.load(model_path)
print(f"✅ Loaded {model_path.name}")

# Predict
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]

# Metrics
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
roc  = roc_auc_score(y_test, y_score)
pra  = average_precision_score(y_test, y_score)

print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-score : {f1:.3f}")
print(f"ROC AUC  : {roc:.3f}")
print(f"PR  AUC  : {pra:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4.6,4.1))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix — Logistic Regression")
plt.tight_layout()
plt.savefig(REPORT_FIG / "confusion_matrix.png", dpi=150)
plt.close()

# ROC
plt.figure(figsize=(4.8,3.8))
RocCurveDisplay.from_predictions(y_test, y_score)
plt.title(f"ROC Curve — AUC={roc:.3f}")
plt.tight_layout()
plt.savefig(REPORT_FIG / "roc_curve.png", dpi=150)
plt.close()

# PR
plt.figure(figsize=(4.8,3.8))
PrecisionRecallDisplay.from_predictions(y_test, y_score)
plt.title(f"Precision–Recall — AP={pra:.3f}")
plt.tight_layout()
plt.savefig(REPORT_FIG / "pr_curve.png", dpi=150)
plt.close()

print(f"📊 Figures saved to: {REPORT_FIG}")
