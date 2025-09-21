import streamlit as st
import pandas as pd
from components.utils import load_raw, load_processed

st.set_page_config(page_title="Data Browser", page_icon="🗂️", layout="wide")
st.title("🗂️ Data Browser")

mode = st.radio("Chọn dữ liệu", ["Raw","X_train","X_test","y_train","y_test"], horizontal=True)
if mode=="Raw":
    df = load_raw()
else:
    X_train, X_test, y_train, y_test = load_processed()
    df = {"X_train":X_train, "X_test":X_test, "y_train":y_train, "y_test":y_test}[mode]

st.caption(f"Shape: {df.shape}")
st.dataframe(df if isinstance(df, pd.DataFrame) else df.to_frame())

st.download_button(
    "Download CSV",
    (df if isinstance(df, pd.DataFrame) else df.to_frame()).to_csv(index=True).encode("utf-8"),
    file_name=f"{mode}.csv", mime="text/csv"
)
