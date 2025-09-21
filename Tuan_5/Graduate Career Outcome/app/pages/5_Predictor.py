import streamlit as st
import pandas as pd, numpy as np
from datetime import datetime
from components.utils import load_pipeline, load_processed, add_history
from components.forms import manual_form, random_sample, pick_from_test

st.set_page_config(page_title="Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Predictor")

pipeline = load_pipeline()
model_name = type(pipeline.named_steps["model"]).__name__

tabs = st.tabs(["Manual","Random","From Test","Upload CSV"])
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

def _predict_one(xdict):
    df = pd.DataFrame([xdict])
    proba = float(pipeline.predict_proba(df)[:,1][0])
    pred = int(proba >= threshold)
    return pred, proba

with tabs[0]:
    st.subheader("Manual Input")
    submitted, x = manual_form()
    if submitted:
        pred, proba = _predict_one(x)
        st.success(f"Prediction: **{pred}** (Placed=1) — Probability: **{proba:.3f}**")
        add_history({
          "timestamp": datetime.now().isoformat(timespec="seconds"),
          "source":"manual", "inputs":x, "pred":pred, "proba":proba,
          "threshold":threshold, "model":model_name
        })

with tabs[1]:
    st.subheader("Random Sample")
    if st.button("Randomize & Predict"):
        x = random_sample()
        st.json(x)
        pred, proba = _predict_one(x)
        st.info(f"Prediction: **{pred}**, proba **{proba:.3f}**")
        add_history({
          "timestamp": datetime.now().isoformat(timespec="seconds"),
          "source":"random", "inputs":x, "pred":pred, "proba":proba,
          "threshold":threshold, "model":model_name
        })

with tabs[2]:
    st.subheader("Pick from Test (processed)")
    idx, row, y_true = pick_from_test()
    st.write("Row (processed) preview:", row.to_frame().T)
    if st.button("Predict selected"):
        # processed row → đi tắt vào model bên trong pipeline
        model = pipeline.named_steps["model"]
        proba = float(model.predict_proba(row.to_frame().T)[:,1][0])
        pred = int(proba >= threshold)
        st.write(f"True label: **{int(y_true)}**")
        st.success(f"Pred: **{pred}** — proba **{proba:.3f}** — {'✅ Đúng' if pred==int(y_true) else '❌ Sai'}")
        add_history({
          "timestamp": datetime.now().isoformat(timespec="seconds"),
          "source":"from_test_idx", "inputs":{"index": int(idx)}, "pred":pred, "proba":proba,
          "threshold":threshold, "model":model_name
        })

with tabs[3]:
    st.subheader("Upload CSV (raw schema)")
    up = st.file_uploader("Chọn file CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        probs = pipeline.predict_proba(df)[:,1]
        preds = (probs >= threshold).astype(int)
        out = df.copy()
        out["pred"] = preds
        out["proba"] = probs
        st.dataframe(out.head())
        st.download_button("Download results", out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")
        # lưu một bản tóm tắt vào history
        add_history({
          "timestamp": datetime.now().isoformat(timespec="seconds"),
          "source":"upload_csv", "inputs":{"rows": len(df)}, "pred": int(out["pred"].mean()>0.5),
          "proba": float(np.mean(probs)), "threshold":threshold, "model":model_name
        })

st.caption("Ghi chú: Log loss chỉ có ý nghĩa khi có nhãn thật (From Test / CSV có cột y).")
