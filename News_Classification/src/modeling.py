# src/modeling.py
from functools import lru_cache
from typing import List, Dict, Any, Tuple
import joblib, numpy as np, pandas as pd
from src.paths import artifact_path, data_path
from src.preprocess import clean_text

@lru_cache(maxsize=1)
def load_classifier():
    return joblib.load(artifact_path("knn_clf.pkl"))

def _top_tfidf_terms(vec, text: str, topk: int = 12) -> List[Tuple[str, float]]:
    Xr = vec.transform([text])
    row = Xr.getrow(0)
    inds, vals = row.indices, row.data
    terms = vec.get_feature_names_out()
    order = np.argsort(-vals)[:topk]
    return [(terms[inds[i]], float(vals[i])) for i in order]

def predict_label(text: str) -> int:
    pipe = load_classifier()
    return int(pipe.predict([clean_text(text)])[0])

def predict_with_neighbors(text: str, k: int = 5) -> Dict[str, Any]:
    pipe = load_classifier()
    txt = clean_text(text)
    vec = pipe.named_steps["tfidfvectorizer"]
    knn = pipe.named_steps["kneighborsclassifier"]

    qv = vec.transform([txt])
    dist, idx = knn.kneighbors(qv, n_neighbors=k, return_distance=True)
    sim = (1.0 - dist[0]).astype(float)
    pred = int(pipe.predict([txt])[0])

    # phiếu bầu theo lớp
    df_corpus = pd.read_csv(data_path("df_file.csv"))
    neigh_labels = [int(df_corpus.iloc[i]["Label"]) for i in idx[0]]
    unique, counts = np.unique(neigh_labels, return_counts=True)
    votes = {int(c): int(n) for c, n in zip(unique, counts)}

    return {
        "pred": pred,
        "neighbors_idx": idx[0].tolist(),
        "neighbors_sim": sim.tolist(),
        "neighbors_labels": neigh_labels,
        "votes": votes,
    }

def explain_prediction(text: str, k: int = 5, topk_terms: int = 12) -> Dict[str, Any]:
    """
    Trả về: pred, votes, top_terms_input, neighbors (idx,label,sim,overlap_terms)
    -> dùng để 'giải thích' tại sao ra label.
    """
    pipe = load_classifier()
    txt = clean_text(text)
    vec = pipe.named_steps["tfidfvectorizer"]
    knn = pipe.named_steps["kneighborsclassifier"]

    top_terms_input = _top_tfidf_terms(vec, txt, topk=topk_terms)

    qv = vec.transform([txt])
    dist, idx = knn.kneighbors(qv, n_neighbors=k, return_distance=True)
    sim = (1.0 - dist[0]).astype(float)

    df_corpus = pd.read_csv(data_path("df_file.csv"))
    neighbors = []
    for s, i in zip(sim, idx[0]):
        t = clean_text(str(df_corpus.iloc[i]["Text"]))
        lab = int(df_corpus.iloc[i]["Label"])
        # tìm overlap theo token exact
        input_words = set(w for w,_ in top_terms_input)
        doc_top = _top_tfidf_terms(vec, t, topk=topk_terms)
        doc_words = set(w for w,_ in doc_top)
        overlap = sorted(list(input_words & doc_words))
        neighbors.append({
            "idx": int(i),
            "label": lab,
            "similarity": float(s),
            "snippet": t[:200] + ("..." if len(t)>200 else ""),
            "overlap_terms": overlap,
        })

    pred = int(pipe.predict([txt])[0])
    neigh_labels = [n["label"] for n in neighbors]
    unique, counts = np.unique(neigh_labels, return_counts=True)
    votes = {int(c): int(n) for c, n in zip(unique, counts)}

    return {
        "pred": pred,
        "votes": votes,
        "top_terms_input": top_terms_input,
        "neighbors": neighbors
    }

def neighbors_dataframe(df_corpus: pd.DataFrame, idx_list: List[int], sim_list: List[float]) -> pd.DataFrame:
    sub = df_corpus.iloc[idx_list].copy()
    sub.insert(0, "Similarity", sim_list)
    return sub[["Similarity", "Label", "Text"]]
