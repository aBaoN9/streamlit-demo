import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="wide")

MODEL_PATH = "model.joblib"
DEFAULT_DATA = "spam_train.csv"

# -----------------------
# Utility: load data
# -----------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Chuẩn hoá tên cột phổ biến
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get("text", None)
    label_col = cols.get("label", None)
    if text_col is None or label_col is None:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].astype(str).fillna("")
    df["label"] = df["label"].astype(str).fillna("")
    return df

# -----------------------
# Build / load model
# -----------------------
def build_pipeline(ngram: int = 1, min_df: int = 1, use_idf: bool = True) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words=None,      # tuỳ ngôn ngữ; để None cho tổng quát
            ngram_range=(1, ngram),
            min_df=min_df,
            sublinear_tf=True,
            use_idf=use_idf
        )),
        ("clf", MultinomialNB(alpha=1.0))
    ])

@st.cache_resource(show_spinner=True)
def train_pipeline(df: pd.DataFrame, ngram: int, min_df: int, use_idf: bool, test_size: float, random_state: int):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values, df["label"].values, test_size=test_size, random_state=random_state, stratify=df["label"].values
    )
    pipe = build_pipeline(ngram=ngram, min_df=min_df, use_idf=use_idf)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "pipeline": pipe,
        "metrics": {"accuracy": acc, "precision": p, "recall": r, "f1": f1},
        "cm": (cm, np.unique(y_test)),
        "report": report
    }

def save_model(pipe: Pipeline, path: str = MODEL_PATH):
    joblib.dump(pipe, path)

def load_model(path: str = MODEL_PATH) -> Pipeline | None:
    if os.path.exists(path):
        return joblib.load(path)
    return None

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("⚙️ Cấu hình")
uploaded_csv = st.sidebar.file_uploader("Tải CSV train (có cột text, label)", type=["csv"])
data_source = "Uploaded" if uploaded_csv else "Default"
st.sidebar.markdown(f"**Data source**: `{data_source}`")

with st.sidebar.expander("Hyperparameters"):
    ngram = st.slider("TF-IDF n-gram max", 1, 3, 1, help="1=unigram, 2=bi-gram, 3=tri-gram")
    min_df = st.number_input("min_df (bỏ từ quá hiếm)", min_value=1, value=1, step=1)
    use_idf = st.checkbox("use_idf (TF-IDF vs TF)", value=True)
    test_size = st.slider("test size", 0.1, 0.4, 0.2, step=0.05)
    random_state = st.number_input("random_state", value=42, step=1)

with st.sidebar:
    colA, colB = st.columns(2)
    retrain_btn = colA.button("🔁 Train")
    save_btn = colB.button("💾 Save model")

    load_btn = st.button("📦 Load model từ file")
    clear_cache = st.button("🧹 Clear cache (data/model)")

if clear_cache:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Đã xoá cache.")

# -----------------------
# Data
# -----------------------
try:
    if uploaded_csv:
        df = load_data(uploaded_csv)
    else:
        df = load_data(DEFAULT_DATA)
except Exception as e:
    st.error(f"Lỗi đọc dữ liệu: {e}")
    st.stop()

st.title("📧 Spam Classifier — Naive Bayes + TF-IDF")
st.caption("Nhập email mới để dự đoán spam/ham, hoặc huấn luyện lại mô hình ngay trong app.")

tab_pred, tab_train, tab_batch, tab_data, tab_about = st.tabs(
    ["🔮 Dự đoán", "🧪 Train & Evaluate", "📦 Batch predict", "🗂️ Dữ liệu", "ℹ️ About"]
)

# -----------------------
# Train & Evaluate
# -----------------------
with tab_train:
    st.subheader("Huấn luyện & đánh giá")
    st.write(f"🔢 Số mẫu: **{len(df)}** | Nhãn: **{', '.join(sorted(df['label'].unique()))}**")

    if retrain_btn or load_btn is False:
        pass  # just to show the buttons do something

    model = None
    metrics = None
    cm = None
    report = None

    if load_btn:
        loaded = load_model()
        if loaded is not None:
            model = loaded
            st.success("Đã load model từ file.")
        else:
            st.warning("Chưa có model.joblib trong thư mục.")

    if retrain_btn or model is None:
        with st.spinner("Đang huấn luyện..."):
            out = train_pipeline(df, ngram, min_df, use_idf, test_size, random_state)
            model = out["pipeline"]
            metrics = out["metrics"]
            cm = out["cm"]
            report = out["report"]
        st.success("Huấn luyện xong.")

    # Store in session for Predict/Batch
    if "model" not in st.session_state or retrain_btn or load_btn:
        st.session_state["model"] = model

    # Show metrics if we just trained
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        c2.metric("Precision (weighted)", f"{metrics['precision']:.3f}")
        c3.metric("Recall (weighted)", f"{metrics['recall']:.3f}")
        c4.metric("F1 (weighted)", f"{metrics['f1']:.3f}")

    if cm:
        import matplotlib.pyplot as plt
        mat, labels = cm
        fig = plt.figure()
        plt.imshow(mat, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
        plt.yticks(ticks=range(len(labels)), labels=labels)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(j, i, mat[i, j], ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig)

    if report:
        st.text("Classification report")
        st.code(report, language="text")

    if save_btn and "model" in st.session_state and st.session_state["model"] is not None:
        save_model(st.session_state["model"])
        st.success("Đã lưu model → model.joblib")

# -----------------------
# Predict (single text)
# -----------------------
with tab_pred:
    st.subheader("Dự đoán 1 văn bản")
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.info("Chưa có model trong session. Vào tab **Train & Evaluate** để train hoặc **Load model**.")
    else:
        txt = st.text_area("Dán nội dung email / tin nhắn", height=160, placeholder="e.g., Congratulations! You won ...")
        if st.button("Dự đoán"):
            if not txt.strip():
                st.warning("Nhập nội dung trước.")
            else:
                pred = st.session_state["model"].predict([txt])[0]
                proba = None
                if hasattr(st.session_state["model"], "predict_proba"):
                    proba = st.session_state["model"].predict_proba([txt])[0]
                    # map classes -> prob
                    cls = st.session_state["model"].classes_
                    conf = dict(zip(cls, proba))
                st.success(f"Kết quả: **{pred}**")
                if proba is not None:
                    st.write("Xác suất:")
                    st.json({k: float(f"{v:.4f}") for k, v in conf.items()})

# -----------------------
# Batch Predict
# -----------------------
with tab_batch:
    st.subheader("Batch predict từ CSV")
    demo = pd.DataFrame({"text": ["free entry in 2 a wkly comp to win tickets", "hi mom, how are you today?"]})
    st.caption("Mẫu CSV cần cột **text**")
    st.dataframe(demo, use_container_width=True)

    csv = st.file_uploader("Tải CSV cần dự đoán", type=["csv"], key="pred_csv")
    if csv and "model" in st.session_state and st.session_state["model"] is not None:
        df_pred = pd.read_csv(csv)
        if "text" not in [c.lower() for c in df_pred.columns]:
            st.error("CSV phải có cột 'text'.")
        else:
            # Chuẩn hoá tên cột 'text'
            for c in df_pred.columns:
                if c.lower() == "text":
                    df_pred = df_pred.rename(columns={c: "text"})
            preds = st.session_state["model"].predict(df_pred["text"].astype(str).fillna(""))
            out = df_pred.copy()
            out["prediction"] = preds
            st.dataframe(out.head(30), use_container_width=True)
            # Cho phép tải về
            out_path = "batch_predictions.csv"
            out.to_csv(out_path, index=False, encoding="utf-8-sig")
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Tải kết quả CSV", data=f, file_name="batch_predictions.csv", mime="text/csv")
    elif csv and ("model" not in st.session_state or st.session_state["model"] is None):
        st.info("Chưa có model. Hãy Train hoặc Load trước.")

# -----------------------
# Data tab
# -----------------------
with tab_data:
    st.subheader("Xem dữ liệu train (mẫu)")
    st.dataframe(df.sample(min(300, len(df))), use_container_width=True)

# -----------------------
# About
# -----------------------
with tab_about:
    st.markdown("""
**Spam Classifier** dùng `TfidfVectorizer + MultinomialNB` (Naive Bayes).
- Train/test split có stratify theo nhãn.
- Tuỳ chỉnh n-gram, min_df, TF vs TF-IDF.
- Lưu/Load model bằng `joblib`.

**Schema dữ liệu**: CSV phải có cột `text` và `label`.
    """)
