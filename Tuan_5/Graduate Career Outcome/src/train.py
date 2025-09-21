"""
Train Logistic Regression in two forms:
1) A full inference pipeline (preprocess + model) trained on RAW data
   -> saved to models/pipeline_latest.pkl  (for Streamlit app: accepts raw inputs)

2) A plain model trained on PROCESSED training set (if available)
   -> saved to models/model_latest.pkl     (for evaluate.py that expects processed X_test)
"""

from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# =========================
# Paths & basic config
# =========================
SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW_DIR / "Placement_Data_Full_Class.csv"

print("=== TRAIN START ===")
print(f"ROOT         : {ROOT}")
print(f"RAW CSV      : {RAW_CSV}")
print(f"PROCESSED DIR: {PROC_DIR}")
print(f"MODELS DIR   : {MODELS_DIR}")
print()


# ====================================================
# Helper: build preprocess (cat: impute+OHE, num: impute+scale)
# Note: use OneHotEncoder(..., sparse=False) for wider sklearn compatibility.
# ====================================================
def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Xử lý tương thích version sklearn
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)        # sklearn cũ

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocess

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocess


# ====================================================
# 1) Train FULL PIPELINE on RAW data  -> pipeline_latest.pkl
#    This is what the Streamlit app will load (accepts raw inputs).
# ====================================================
if not RAW_CSV.exists():
    print("❌ Không tìm thấy file RAW:", RAW_CSV)
    print("➡️  Đảm bảo đã đặt Placement_Data_Full_Class.csv vào data/raw/")
    sys.exit(1)

print("Step 1) Load RAW data...")
df_raw = pd.read_csv(RAW_CSV)
print("RAW shape:", df_raw.shape)

# Build target and features for RAW
if "status" not in df_raw.columns:
    raise ValueError("Thiếu cột target 'status' trong RAW CSV.")

y_raw = df_raw["status"].map({"Placed": 1, "Not Placed": 0})
if y_raw.isnull().any():
    bad_vals = df_raw.loc[y_raw.isnull(), "status"].unique()
    raise ValueError(f"Giá trị 'status' không hợp lệ: {bad_vals}. Cần map về {{Placed, Not Placed}}.")

drop_cols = [c for c in ["sl_no", "salary", "status"] if c in df_raw.columns]
X_raw = df_raw.drop(columns=drop_cols)

print("X_raw columns:", list(X_raw.columns))
print("Class balance (RAW):")
print(y_raw.value_counts(normalize=True).round(3))
print()

print("Step 2) Build preprocess & pipeline...")
preprocess = build_preprocess(X_raw)
logreg = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=SEED
)
pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", logreg)
])

print("Step 3) Fit pipeline on RAW (full dataset for inference pipeline)...")
pipeline.fit(X_raw, y_raw)
pipe_path = MODELS_DIR / "pipeline_latest.pkl"
joblib.dump(pipeline, pipe_path)
print(f"✅ Saved inference pipeline to: {pipe_path}")
print()


# ====================================================
# 2) Train PLAIN MODEL on PROCESSED train set -> model_latest.pkl
#    This is optional but useful for evaluate.py that expects processed X_test.
# ====================================================
x_train_path = PROC_DIR / "X_train.csv"
y_train_path = PROC_DIR / "y_train.csv"

if x_train_path.exists() and y_train_path.exists():
    print("Step 4) Found processed train set. Train plain model for evaluation pipeline...")
    X_train = pd.read_csv(x_train_path, index_col=0)
    y_train = pd.read_csv(y_train_path, index_col=0)["status"]

    print("Processed TRAIN shape:", X_train.shape)
    print("Class balance (TRAIN):")
    print(y_train.value_counts(normalize=True).round(3))

    logreg_proc = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED
    )
    logreg_proc.fit(X_train, y_train)

    model_path = MODELS_DIR / "model_latest.pkl"
    joblib.dump(logreg_proc, model_path)
    print(f"✅ Saved processed-model to: {model_path}")
else:
    print("ℹ️ Không thấy processed train set (data/processed/X_train.csv, y_train.csv).")
    print("   Bỏ qua bước lưu model_latest.pkl (chỉ lưu pipeline_latest.pkl cho app).")

print("\n=== TRAIN FINISH ===")
