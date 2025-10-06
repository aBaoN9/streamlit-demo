import joblib, pandas as pd
from pathlib import Path
from .config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "decision_tree_rating_regressor.pkl"
_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_one(sample_dict):
    model = load_model()
    X = pd.DataFrame([sample_dict])
    return float(model.predict(X)[0])
