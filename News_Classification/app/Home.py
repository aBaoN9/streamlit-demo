import streamlit as st
st.set_page_config(page_title="News Demo", layout="wide")
st.title("📰 News Classification & Search (KNN + VSM)")
st.markdown("""
Chọn tab ở sidebar:
1) **Giới thiệu** – mô tả web & cách dùng  
2) **KNN Classifier** – dự đoán nhãn
3) **VSM Search** – tìm văn bản tương tự 
4) **Giải thích dữ liệu** – label distribution, text length  
5) **Lịch sử** – lưu các truy vấn/gõ thử gần đây
""")
