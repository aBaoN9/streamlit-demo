from pathlib import Path
import json
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw" / "Placement_Data_Full_Class.csv"
PROC = DATA / "processed"
MODELS = ROOT / "models"
REPORT = ROOT / "report"
APP_DIR = ROOT / "app"
HIST_CSV = APP_DIR / "pred_history.csv"

def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW)

def load_processed():
    X_train = pd.read_csv(PROC / "X_train.csv", index_col=0)
    X_test  = pd.read_csv(PROC / "X_test.csv", index_col=0)
    y_train = pd.read_csv(PROC / "y_train.csv", index_col=0)["status"]
    y_test  = pd.read_csv(PROC / "y_test.csv", index_col=0)["status"]
    return X_train, X_test, y_train, y_test

def load_pipeline():
    path = MODELS / "pipeline_latest.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Không thấy {path}. Chạy: python src/train.py")
    return joblib.load(path)

def ensure_history():
    APP_DIR.mkdir(parents=True, exist_ok=True)
    if not HIST_CSV.exists():
        pd.DataFrame(columns=["timestamp","source","inputs","pred","proba","threshold","model"]).to_csv(HIST_CSV, index=False)

def add_history(record: dict):
    ensure_history()
    df = pd.read_csv(HIST_CSV)
    df.loc[len(df)] = [
        record.get("timestamp"),
        record.get("source"),
        json.dumps(record.get("inputs"), ensure_ascii=False),
        record.get("pred"),
        record.get("proba"),
        record.get("threshold"),
        record.get("model"),
    ]
    df.to_csv(HIST_CSV, index=False)

def read_history() -> pd.DataFrame:
    ensure_history()
    return pd.read_csv(HIST_CSV)

def clear_history():
    ensure_history()
    pd.DataFrame(columns=["timestamp","source","inputs","pred","proba","threshold","model"]).to_csv(HIST_CSV, index=False)
