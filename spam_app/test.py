import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="wide")
st.title("📩 SMS Spam Classifier — Naive Bayes vs Decision Tree (Voting & Stacking)")

# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Try to normalize common schemas
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    # Try to map typical column names
    if "label" not in df.columns:
        # Try 'category', 'target', etc.
        for cand in ["category", "target", "class"]:
            if cand in df.columns:
                df["label"] = df[cand]
                break
    if "text" not in df.columns:
        for cand in ["message", "sms", "content", "body"]:
            if cand in df.columns:
                df["text"] = df[cand]
                break
    # Keep only needed cols
    if not {"label", "text"}.issubset(df.columns):
        raise ValueError("File CSV cần có 2 cột: 'label' và 'text' (hoặc tên tương đương).")
    # Drop NA + strip
    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    # Normalize label variants
    df["label"] = df["label"].replace(
        {"spam": "spam", "ham": "ham", "1":"spam", "0":"ham", "spam msg":"spam", "normal":"ham"}
    )
    return df

def make_vectorizer(kind: str):
    if kind == "Word (1-2gram)":
        return TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)
    elif kind == "Char (3-5gram)":
        return TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_df=0.95)
    else:
        return TfidfVectorizer(ngram_range=(1,1), min_df=2, max_df=0.95)

def build_model(model_name: str, vec_kind: str):
    vec = make_vectorizer(vec_kind)

    if model_name == "Naive Bayes":
        clf = MultinomialNB()
        pipe = Pipeline([("vec", vec), ("clf", clf)])
        return pipe

    if model_name == "Decision Tree":
        clf = DecisionTreeClassifier(
            criterion="gini", max_depth=None, random_state=42, min_samples_leaf=1
        )
        pipe = Pipeline([("vec", vec), ("clf", clf)])
        return pipe

    if model_name == "Voting (NB + DT, soft)":
        nb = ("nb", Pipeline([("vec", vec), ("clf", MultinomialNB())]))
        dt = ("dt", Pipeline([("vec", vec), ("clf", DecisionTreeClassifier(random_state=42))]))
        # Soft voting needs predict_proba — cả NB và DT đều hỗ trợ
        clf = VotingClassifier(estimators=[nb, dt], voting="soft", n_jobs=None, flatten_transform=True)
        return clf

    if model_name == "Stacking (NB+DT -> LR)":
        # Stacking với vectorizer chia sẻ: ta làm "2 pipeline riêng" -> lấy xác suất, ghép cột
        # Để đơn giản trong Streamlit: ta huấn luyện 2 base model riêng và meta LR trên output probs.
        # Ta bọc bằng một class nhỏ để có fit/predict nhất quán.
        class StackingWrapper:
            def __init__(self, vec_kind):
                self.vec_kind = vec_kind
                self.nb = Pipeline([("vec", make_vectorizer(vec_kind)), ("clf", MultinomialNB())])
                self.dt = Pipeline([("vec", make_vectorizer(vec_kind)), ("clf", DecisionTreeClassifier(random_state=42))])
                self.meta = LogisticRegression(max_iter=200)

            def fit(self, X, y):
                self.lb = LabelEncoder().fit(y)
                y_enc = self.lb.transform(y)
                self.nb.fit(X, y_enc)
                self.dt.fit(X, y_enc)
                Z = self._stack_features(X)  # probs concat
                self.meta.fit(Z, y_enc)
                return self

            def _stack_features(self, X):
                p1 = self.nb.predict_proba(X)
                p2 = self.dt.predict_proba(X)
                return np.hstack([p1, p2])

            def predict(self, X):
                Z = self._stack_features(X)
                y_hat = self.meta.predict(Z)
                return self.lb.inverse_transform(y_hat)

            def predict_proba(self, X):
                Z = self._stack_features(X)
                return self.meta.predict_proba(Z)

        return StackingWrapper(vec_kind)

    raise ValueError("Unknown model.")

def evaluate(y_true, y_pred, y_proba=None, pos_label="spam"):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1  = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[pos_label, "ham"])
    roc = None
    if y_proba is not None and len(np.unique(y_true)) == 2:
        # map labels to {1,0} w.r.t pos_label
        y_bin = (np.array(y_true) == pos_label).astype(int)
        # Find column index of pos_label if proba is aligned with classes
        if isinstance(y_proba, pd.DataFrame):
            p_spam = y_proba[pos_label].values
        else:
            # assume column order [ham, spam] or [0,1]? try to infer:
            # safer: if y_proba has 2 columns, pick the second as "spam" (common for LabelEncoder=ham(0),spam(1))
            p_spam = y_proba[:, -1]
        try:
            roc = roc_auc_score(y_bin, p_spam)
        except Exception:
            roc = None
    return acc, pre, rec, f1, cm, roc


# =========================
# Sidebar controls
# =========================
st.sidebar.header("⚙️ Cấu hình")
model_name = st.sidebar.selectbox(
    "Chọn mô hình", 
    ["Naive Bayes", "Decision Tree", "Voting (NB + DT, soft)", "Stacking (NB+DT -> LR)"],
    index=0
)
vec_kind = st.sidebar.selectbox(
    "Kiểu vector hóa TF-IDF",
    ["Word (1-2gram)", "Char (3-5gram)", "Word (1-gram)"],
    index=0
)
test_size = st.sidebar.slider("Tỉ lệ test", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)
pos_label = "spam"

st.sidebar.markdown("---")
st.sidebar.caption("💡 Dữ liệu CSV cần có cột `label` (spam/ham) và `text`.")

# =========================
# Data input
# =========================
tab1, tab2 = st.tabs(["📂 Dữ liệu", "🧪 Dự đoán nhanh"])

with tab1:
    uploaded = st.file_uploader("Tải file CSV của bạn (label, text)", type=["csv"])
    use_sample = st.checkbox("Dùng file mẫu 20 dòng (song ngữ) nếu chưa có dữ liệu", value=False)

    if uploaded is None and not use_sample:
        st.info("Hãy tải file CSV hoặc chọn dùng file mẫu.")
    else:
        if use_sample:
            df = pd.read_csv("sample_sms_spam.csv") if False else None  # placeholder: not used
            # Thay vì đọc file cục bộ, ta generate nhỏ trong code cho demo:
            data = [
                ("spam","Congratulations! You have won a $1000 gift card. Click here to claim now."),
                ("ham","Hey, are we still on for lunch at 12?"),
                ("spam","URGENT: Your account will be suspended. Verify your details at http://bit.ly/xyz"),
                ("ham","I'll be late to the meeting by 10 minutes."),
                ("spam","Claim your FREE vacation now!!! Reply YES to win."),
                ("ham","Please review the attached report and send me your feedback."),
                ("spam","You have been selected for a limited-time offer. Apply now!"),
                ("ham","Thanks for the birthday wishes!"),
                ("spam","Final notice: Payment required to avoid penalty. Pay at secure link."),
                ("ham","Can you pick up milk on your way home?"),
                ("spam","Chúc mừng! Bạn trúng thưởng iPhone 15. Bấm vào link để nhận quà."),
                ("ham","Tối nay 7h lớp mình học Zoom nhé."),
                ("spam","Cảnh báo: Tài khoản của bạn sẽ bị khóa trong 24h. Xác minh ngay."),
                ("ham","Mai họp nhóm ở thư viện, nhớ mang laptop."),
                ("spam","Nhận quà miễn phí chỉ hôm nay! Trả lời YES để nhận."),
                ("ham","Mình đã chuyển khoản tiền nhà rồi nha."),
                ("spam","Khuyến mãi siêu sốc! Mua 1 tặng 1. Click ngay!"),
                ("ham","Cảm ơn bạn đã hỗ trợ mình hôm qua."),
                ("spam","Thông báo cuối: Nộp phí ngay để tránh phạt. Xem link bảo mật."),
                ("ham","Bạn gửi mình file bài tập với nhé."),
            ]
            df = pd.DataFrame(data, columns=["label","text"])
        else:
            df = load_csv(uploaded)

        st.success(f"Đã nạp {len(df)} dòng. Tỉ lệ spam: {(df['label']=='spam').mean()*100:.2f}%")
        st.dataframe(df.head(10), use_container_width=True)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"].values, df["label"].values,
            test_size=test_size, random_state=random_state, stratify=df["label"].values
        )

        # Build + fit
        model = build_model(model_name, vec_kind)
        with st.spinner("Đang huấn luyện..."):
            model.fit(X_train, y_train)

        # Predict
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        y_pred = model.predict(X_test)

        # Evaluate
        acc, pre, rec, f1, cm, roc = evaluate(y_test, y_pred, y_proba=y_proba, pos_label=pos_label)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision (spam)", f"{pre:.3f}")
        c3.metric("Recall (spam)", f"{rec:.3f}")
        c4.metric("F1 (spam)", f"{f1:.3f}")
        c5.metric("ROC-AUC", f"{roc:.3f}" if roc is not None else "—")

        st.markdown("#### Confusion Matrix (labels: [spam, ham])")
        cm_df = pd.DataFrame(cm, index=["spam_true","ham_true"], columns=["spam_pred","ham_pred"])
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("#### Classification Report")
        st.text(classification_report(y_test, y_pred, zero_division=0))

with tab2:
    st.write("Nhập một hoặc nhiều tin nhắn, mỗi dòng là một tin:")
    user_text = st.text_area("Tin nhắn", height=160, placeholder="Ví dụ:\nDear valued customer, verify your account at ...\nTối nay họp nhóm lúc 7h nhé.")
    if "last_trained_model" not in st.session_state:
        st.session_state["last_trained_model"] = None
        st.session_state["model_name"] = None

    predict_btn = st.button("Phân loại")
    if predict_btn:
        if user_text.strip() == "":
            st.warning("Nhập ít nhất 1 dòng tin nhắn.")
        else:
            # For quick demo, we re-train a tiny model on-the-fly using demo data
            # In a real session, you would keep the trained model from Tab 1 (shared state),
            # but Streamlit may re-run script; easiest is to train on minimal embedded data:
            data_demo = [
                ("spam","Congratulations! You have won a $1000 gift card. Click here to claim now."),
                ("ham","Hey, are we still on for lunch at 12?"),
                ("spam","URGENT: Your account will be suspended. Verify your details at http://bit.ly/xyz"),
                ("ham","I'll be late to the meeting by 10 minutes."),
                ("spam","Chúc mừng! Bạn trúng thưởng iPhone 15. Bấm vào link để nhận quà."),
                ("ham","Mai họp nhóm ở thư viện, nhớ mang laptop.")
            ]
            df_demo = pd.DataFrame(data_demo, columns=["label","text"])
            small_model = build_model(model_name, vec_kind)
            small_model.fit(df_demo["text"], df_demo["label"])

            lines = [ln.strip() for ln in user_text.split("\n") if ln.strip()]
            preds = small_model.predict(lines)
            probs = None
            if hasattr(small_model, "predict_proba"):
                proba = small_model.predict_proba(lines)
                # heuristic: take spam prob as last column
                probs = proba[:, -1]

            res = pd.DataFrame({"text": lines, "pred": preds})
            if probs is not None:
                res["P(spam)"] = probs.round(3)

            st.success("Kết quả dự đoán:")
            st.dataframe(res, use_container_width=True)
            st.caption("Gợi ý: P(spam) cao → khả năng là tin nhắn rác/phishing.")

st.markdown("---")
st.caption("👩‍🔬 Mẹo: Dùng **Voting** hoặc **Stacking** để tăng độ ổn định. Thử thay đổi kiểu vector TF-IDF (Word vs Char) để phù hợp dữ liệu Việt/Anh.")
