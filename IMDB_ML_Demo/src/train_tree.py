import joblib
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from .config import MODELS_DIR, COL_RATING, COL_CERT, COL_GENRE
from .data_utils import load_raw
from .features import parse_year, parse_duration, clean_votes, count_stars, desc_len, primary_genre

def build_df():
    df = load_raw().copy()
    df["year_num"] = df["year"].map(parse_year)
    df["duration_min"] = df["duration"].map(parse_duration)
    df["votes_num"] = df["votes"].map(clean_votes)
    df["stars_count"] = df["stars"].map(count_stars)
    df["desc_len"] = df["description"].map(desc_len)
    df["genre_primary"] = df["genre"].map(primary_genre)
    return df.dropna(subset=[COL_RATING])

def train_and_save():
    df = build_df()
    num = ["year_num","duration_min","votes_num","stars_count","desc_len"]
    cat = [COL_CERT, "genre_primary"]
    X = df[num + cat]
    y = df[COL_RATING].astype(float)

    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    pipe = Pipeline([("prep", prep),
                     ("model", DecisionTreeRegressor(max_depth=6, random_state=42))])
    pipe.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "decision_tree_rating_regressor.pkl"
    joblib.dump(pipe, out)
    return out

if __name__ == "__main__":
    print(train_and_save())
