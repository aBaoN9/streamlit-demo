import joblib, pandas as pd
from pathlib import Path
from .config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "naive_bayes_genre_from_description.pkl"
_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_text(text: str):
    model = load_model()
    return model.predict(pd.Series([text]))[0]
