import streamlit as st

st.set_page_config(page_title="Project Flow", page_icon="🔁", layout="centered")
st.title("🔁 Project Flow")

steps = [
    "📥 data/raw → tải CSV gốc",
    "🔧 notebooks/00..01 → EDA & feature engineering",
    "🧪 data/processed → X_train/X_test/y_train/y_test",
    "🧠 notebooks/02..05 → baseline, explainability, teaching train/test",
    "🏋️ src/train.py → fit pipeline + save pipeline_latest.pkl",
    "💾 models/*.pkl → artifacts",
    "🧾 src/evaluate.py → metrics + figures vào report/figures",
    "🌐 app/* → dashboard, predict, history"
]
for i, s in enumerate(steps, start=1):
    st.markdown(f"**Bước {i}.** {s}")
