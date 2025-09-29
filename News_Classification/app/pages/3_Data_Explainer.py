import streamlit as st, pandas as pd, numpy as np, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.append(str(PROJECT_ROOT))

from src.paths import data_path

st.set_page_config(page_title="Giải thích dữ liệu", layout="wide")
st.title("📊 Giải thích dữ liệu")

df = pd.read_csv(data_path("df_file.csv"))
st.markdown(f"- Số văn bản: **{len(df)}**")

# Phân phối nhãn
st.subheader("Phân phối nhãn")
st.bar_chart(df["Label"].value_counts().sort_index())

# Độ dài văn bản
st.subheader("Độ dài văn bản (số từ)")
lens = df["Text"].astype(str).str.split().apply(len)
st.line_chart(lens.describe()[["min","25%","50%","75%","max"]])
st.caption(f"Mean={lens.mean():.1f}, Median={lens.median():.1f}")

st.markdown("""
**Nhận xét nhanh:** dữ liệu là tin tức tiếng Anh, độ dài trung bình đủ để TF-IDF bắt cụm từ; phân phối nhãn tương đối cân bằng, phù hợp cho K-NN/VSM.
""")
