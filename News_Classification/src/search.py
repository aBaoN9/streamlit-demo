# src/search.py
from functools import lru_cache
from typing import List, Dict, Any
import joblib, pandas as pd, numpy as np
from src.paths import artifact_path, data_path
from src.preprocess import clean_text

@lru_cache(maxsize=1)
def load_vectorizer():
    return joblib.load(artifact_path("tfidf.pkl"))

@lru_cache(maxsize=1)
def load_index():
    return joblib.load(artifact_path("knn_index.pkl"))

@lru_cache(maxsize=1)
def load_corpus_df():
    return pd.read_csv(data_path("df_file.csv"))

def search_top_k(query: str, k: int = 5) -> pd.DataFrame:
    tfidf, index, df = load_vectorizer(), load_index(), load_corpus_df()
    q = clean_text(query)
    qv = tfidf.transform([q])
    dist, idx = index.kneighbors(qv, n_neighbors=k, return_distance=True)
    sim = (1.0 - dist[0]).astype(float)
    out = df.iloc[idx[0]].copy()
    out.insert(0, "Similarity", sim)
    return out[["Similarity", "Label", "Text"]]

def explain_search(query: str, k: int = 5, topk_terms: int = 12) -> Dict[str, Any]:
    """
    Trả về: top_terms_query (tf-idf cao), results: each has label, similarity, snippet, overlap_terms
    """
    tfidf, index, df = load_vectorizer(), load_index(), load_corpus_df()
    q = clean_text(query)
    qv = tfidf.transform([q])

    # top terms của query
    row = qv.getrow(0)
    inds, vals = row.indices, row.data
    terms = tfidf.get_feature_names_out()
    order = np.argsort(-vals)[:topk_terms]
    top_terms_query = [(terms[inds[i]], float(vals[i])) for i in order]

    dist, idx = index.kneighbors(qv, n_neighbors=k, return_distance=True)
    sim = (1.0 - dist[0]).astype(float)

    results = []
    query_tokens = set(w for w,_ in top_terms_query)
    for s, i in zip(sim, idx[0]):
        txt = clean_text(str(df.iloc[i]["Text"]))
        # top terms của document để lấy overlap
        dv = tfidf.transform([txt]).getrow(0)
        d_inds, d_vals = dv.indices, dv.data
        d_order = np.argsort(-d_vals)[:topk_terms]
        d_terms = tfidf.get_feature_names_out()
        doc_top = set(d_terms[d_inds[j]] for j in d_order)
        overlap = sorted(list(query_tokens & doc_top))
        results.append({
            "similarity": float(s),
            "label": int(df.iloc[i]["Label"]),
            "snippet": txt[:200] + ("..." if len(txt)>200 else ""),
            "overlap_terms": overlap
        })
    return {"top_terms_query": top_terms_query, "results": results}
