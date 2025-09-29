import streamlit as st, pandas as pd, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.append(str(PROJECT_ROOT))

from src.modeling import explain_prediction, neighbors_dataframe
from src.paths import data_path
from src.history import append_history

st.set_page_config(page_title="KNN Classifier", layout="wide")
st.title("📌 KNN News Classifier — có giải thích")

text_input = st.text_area("Nhập nội dung tin tức:", height=200, placeholder="Paste a short news paragraph in English...")
k = st.slider("Số láng giềng (k):", 3, 15, 7, step=2)
topk_terms = st.slider("Số từ TF-IDF nổi bật hiển thị:", 6, 20, 12)

if st.button("Phân loại"):
    if text_input.strip():
        out = explain_prediction(text_input, k=k, topk_terms=topk_terms)
        st.success(f"🔎 Nhãn dự đoán: **{out['pred']}**")

        # 1) Phiếu bầu theo láng giềng
        vote_series = pd.Series(out["votes"]).sort_index()
        st.subheader("🧮 Phiếu bầu theo láng giềng")
        st.bar_chart(vote_series)

        # 2) Từ khóa TF-IDF nổi bật của văn bản đầu vào
        st.subheader("🧩 Từ khóa TF-IDF nổi bật (văn bản đầu vào)")
        df_terms = pd.DataFrame(out["top_terms_input"], columns=["term","tfidf"])
        st.dataframe(df_terms)

        # 3) Bảng láng giềng k gần nhất + overlap terms
        st.subheader("🔗 Láng giềng gần nhất (có từ khóa trùng)")
        neigh_df_full = pd.read_csv(data_path("df_file.csv"))
        idx_list = [n["idx"] for n in out["neighbors"]]
        sim_list = [n["similarity"] for n in out["neighbors"]]
        tbl = neighbors_dataframe(neigh_df_full, idx_list, sim_list)
        tbl["overlap_terms"] = [", ".join(n["overlap_terms"]) for n in out["neighbors"]]
        st.dataframe(tbl)

        # 4) Ghi lịch sử
        append_history({
            "type": "knn",
            "input": text_input[:160] + ("..." if len(text_input)>160 else ""),
            "k": k,
            "pred": out["pred"],
            "votes": out["votes"]
        })
    else:
        st.warning("Bạn cần nhập nội dung.")
