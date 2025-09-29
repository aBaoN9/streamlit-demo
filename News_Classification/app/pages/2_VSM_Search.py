import streamlit as st, pandas as pd, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.append(str(PROJECT_ROOT))

from src.search import explain_search
from src.history import append_history

st.set_page_config(page_title="VSM Search", layout="wide")
st.title("🔍 VSM Search")

query = st.text_input("Nhập truy vấn (tiếng Anh):", placeholder="e.g., election results and budget policy")
top_k = st.slider("Số kết quả:", 3, 10, 5)
topk_terms = st.slider("Số từ TF-IDF nổi bật hiển thị:", 6, 20, 12)

if st.button("Tìm kiếm"):
    if query.strip():
        exp = explain_search(query, k=top_k, topk_terms=topk_terms)

        st.subheader("🧩 Từ khóa TF-IDF nổi bật của query")
        st.dataframe(pd.DataFrame(exp["top_terms_query"], columns=["term","tfidf"]))

        st.subheader("📃 Kết quả top-k (có từ khóa trùng)")
        rows = []
        for r in exp["results"]:
            rows.append({
                "Similarity": r["similarity"],
                "Label": r["label"],
                "Overlap terms": ", ".join(r["overlap_terms"]),
                "Snippet": r["snippet"]
            })
        st.dataframe(pd.DataFrame(rows))

        append_history({
            "type": "vsm",
            "query": query,
            "k": top_k,
            "top_terms": [t for t,_ in exp["top_terms_query"]]
        })
    else:
        st.warning("Bạn cần nhập query trước.")

