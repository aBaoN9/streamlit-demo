# --- fix import path ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.predict_nb import predict_text, load_model

# ✅ dùng deep-translator, tương thích Py 3.13
from deep_translator import GoogleTranslator

def translate_to_en(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        # nếu lỗi mạng, fallback dùng nguyên văn
        return text

st.title("🎭 Phân loại Thể loại (Naive Bayes)")
st.caption("Bạn có thể nhập mô tả bằng tiếng Việt; hệ thống sẽ dịch sang tiếng Anh rồi phân loại.")

text = st.text_area(
    "Nhập mô tả phim:",
    "Một sát thủ đã nghỉ hưu trở lại con đường báo thù sau khi gia đình anh bị hại..."
)

if st.button("Phân loại"):
    text_en = translate_to_en(text)
    st.write("**📘 Bản dịch sang tiếng Anh:**")
    st.info(text_en)

    label = predict_text(text_en)
    st.success(f"🎯 Thể loại dự đoán: **{label}**")

    # Xác suất top-5
    nb_model = load_model()
    proba = nb_model.predict_proba(pd.Series([text_en]))[0]
    classes = nb_model.classes_
    top = proba.argsort()[::-1][:5]
    import plotly.express as px
    dfp = pd.DataFrame({"Genre": classes[top], "Probability": proba[top]})
    st.plotly_chart(
        px.bar(dfp, x="Probability", y="Genre", orientation="h", title="Top-5 xác suất"),
        use_container_width=True
    )
