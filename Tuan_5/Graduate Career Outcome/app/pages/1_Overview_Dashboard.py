import streamlit as st
import pandas as pd
from pathlib import Path
from components.utils import load_raw, REPORT

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
st.title("📊 Overview Dashboard")

raw = load_raw()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(raw))
c2.metric("Columns", raw.shape[1])
c3.metric("Placed %", f"{(raw['status'].eq('Placed').mean()*100):.1f}%")
num_cols = ["ssc_p","hsc_p","degree_p","etest_p","mba_p"]
c4.metric("# Numeric / # Categorical", f"{len(num_cols)} / {raw.shape[1]-len(num_cols)-3}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Class balance")
    st.bar_chart(raw["status"].value_counts())

with col2:
    st.subheader("Numeric preview")
    st.dataframe(raw[num_cols].describe().T)

st.divider()
st.subheader("Model figures (nếu đã chạy evaluate.py)")
fig_dir = Path(REPORT) / "figures"
colA, colB, colC = st.columns(3)
for name, col in zip(["roc_curve.png", "pr_curve.png", "confusion_matrix.png"], [colA, colB, colC]):
    f = fig_dir / name
    if f.exists():
        col.image(str(f), caption=name)
    else:
        col.info(f"Chưa có {name}. Chạy: `python src/evaluate.py`")
