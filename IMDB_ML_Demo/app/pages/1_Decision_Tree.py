# --- add these 3 lines at top ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# ---------------------------------

from src.predict_tree import predict_one  # hoặc predict_text / load_raw tuỳ file

import streamlit as st
from src.predict_tree import predict_one

st.title("🎬 Dự đoán Rating (Decision Tree)")
st.write("Nhập thông tin phim:")

col1, col2, col3 = st.columns(3)
with col1:
    year_num = st.number_input("Năm (yyyy)", 1970, 2030, 2022)
    duration_min = st.number_input("Thời lượng (phút)", 1, 500, 120)
with col2:
    votes_num = st.number_input("Số votes (ước lượng)", 0, 5_000_000, 150_000)
    stars_count = st.slider("Số diễn viên chính", 1, 10, 3)
with col3:
    desc_len = st.slider("Độ dài mô tả (số từ)", 0, 400, 30)
    certificate = st.selectbox("Certificate", ["PG", "PG-13", "R", "TV-MA", "TV-14", "G","Not Rated"])
genre_primary = st.selectbox("Thể loại chính", ["Action","Drama","Comedy","Thriller","Adventure","Crime","Animation","Horror","Romance","Sci-Fi"])

if st.button("Dự đoán"):
    sample = dict(year_num=year_num, duration_min=duration_min, votes_num=votes_num,
                  stars_count=stars_count, desc_len=desc_len, certificate=certificate,
                  genre_primary=genre_primary)
    pred = predict_one(sample)
    st.success(f"⭐ Dự đoán rating: **{pred:.2f}**")
