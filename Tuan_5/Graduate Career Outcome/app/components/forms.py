import random
import numpy as np
import streamlit as st
import pandas as pd
from .utils import load_raw, load_processed

RAW_COLS_CAT = ["gender","ssc_b","hsc_b","hsc_s","degree_t","workex","specialisation"]
RAW_COLS_NUM = ["ssc_p","hsc_p","degree_p","etest_p","mba_p"]

def _stats():
    raw = load_raw()
    return {
        "cat": {c: sorted(raw[c].dropna().unique().tolist()) for c in RAW_COLS_CAT},
        "num": {c: (float(raw[c].min()), float(raw[c].max())) for c in RAW_COLS_NUM}
    }

def manual_form():
    s = _stats()
    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)
        gender = col1.selectbox("gender", s["cat"]["gender"])
        ssc_b  = col2.selectbox("ssc_b", s["cat"]["ssc_b"])
        hsc_b  = col3.selectbox("hsc_b", s["cat"]["hsc_b"])
        hsc_s  = col1.selectbox("hsc_s", s["cat"]["hsc_s"])
        degree_t = col2.selectbox("degree_t", s["cat"]["degree_t"])
        workex   = col3.selectbox("workex", s["cat"]["workex"])
        special  = col1.selectbox("specialisation", s["cat"]["specialisation"])

        ssc_p = col2.slider("ssc_p (%)", *s["num"]["ssc_p"])
        hsc_p = col3.slider("hsc_p (%)", *s["num"]["hsc_p"])
        degree_p = col1.slider("degree_p (%)", *s["num"]["degree_p"])
        etest_p  = col2.slider("etest_p (%)", *s["num"]["etest_p"])
        mba_p    = col3.slider("mba_p (%)", *s["num"]["mba_p"])

        submitted = st.form_submit_button("Predict")
    x = {
      "gender": gender, "ssc_b": ssc_b, "hsc_b": hsc_b, "hsc_s": hsc_s,
      "degree_t": degree_t, "workex": workex, "specialisation": special,
      "ssc_p": ssc_p, "hsc_p": hsc_p, "degree_p": degree_p, "etest_p": etest_p, "mba_p": mba_p
    }
    return submitted, x

def random_sample():
    s = _stats()
    x = {c: random.choice(s["cat"][c]) for c in RAW_COLS_CAT}
    for c,(mn,mx) in s["num"].items():
        x[c] = float(np.round(np.random.uniform(mn, mx), 2))
    return x

def pick_from_test():
    # sử dụng processed để chứng minh độ chính xác (hiển thị y_true)
    X_train, X_test, y_train, y_test = load_processed()
    idx = st.selectbox("Chọn index từ X_test", X_test.index.tolist())
    row = X_test.loc[idx]
    y_true = y_test.loc[idx]
    return idx, row, y_true
