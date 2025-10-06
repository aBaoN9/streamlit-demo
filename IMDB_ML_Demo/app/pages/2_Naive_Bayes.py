# --- fix import path ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# ---------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
from src.predict_nb import predict_text, load_model

# --- thêm thư viện dịch ---
from googletrans import Translator

st.title("🎭 Phân loại Thể loại (Naive Bayes)")
st.caption("Mô tả phim có thể viết bằng tiếng Việt hoặc tiếng Anh. Ứng dụng sẽ tự dịch sang tiếng Anh trước khi phân loại.")

# Nhập nội dung phim
text = st.text_area(
    "Nhập mô tả phim:",
    "Một sát thủ đã nghỉ hưu trở lại con đường báo thù sau khi gia đình anh bị hại..."
)

# Nút dự đoán
if st.button("Phân loại"):
    translator = Translator()
    try:
        # --- Dịch sang tiếng Anh ---
        translation = translator.translate(text, src='auto', dest='en')
        text_en = translation.text
        st.write("**📘 Bản dịch sang tiếng Anh:**")
        st.info(text_en)

        # --- Dự đoán bằng Naive Bayes ---
        label = predict_text(text_en)
        st.success(f"🎯 Thể loại dự đoán: **{label}**")

        # --- Hiển thị xác suất (Top 5 thể loại) ---
        nb_model = load_model()
        proba = nb_model.predict_proba(pd.Series([text_en]))[0]
        classes = nb_model.classes_

        top_idx = proba.argsort()[::-1][:5]
        df_proba = pd.DataFrame({
            "Genre": classes[top_idx],
            "Probability": proba[top_idx]
        })

        st.plotly_chart(
            px.bar(df_proba, x="Probability", y="Genre", orientation="h",
                   title="Top 5 thể loại có xác suất cao nhất", text_auto=".2f"),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Lỗi khi dịch hoặc phân loại: {e}")
