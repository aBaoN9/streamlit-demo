import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from .config import MODELS_DIR
from .data_utils import load_raw
from .features import primary_genre

def build_df():
    df = load_raw().copy()
    df["genre_primary"] = df["genre"].map(primary_genre)
    df = df.dropna(subset=["description","genre_primary"])
    df = df[df["description"].str.strip().astype(bool)]
    # lọc genre hiếm
    counts = df["genre_primary"].value_counts()
    keeps = counts[counts>=5].index
    return df[df["genre_primary"].isin(keeps)].copy()

def train_and_save():
    df = build_df()
    X, y = df["description"], df["genre_primary"]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2))),
        ("clf", MultinomialNB())
    ])
    pipe.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "naive_bayes_genre_from_description.pkl"
    joblib.dump(pipe, out)
    return out

if __name__ == "__main__":
    print(train_and_save())
