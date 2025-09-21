import streamlit as st

st.set_page_config(page_title="Graduate Career Outcome", page_icon="🎓", layout="wide")

st.title("🎓 Graduate Career Outcome — Demo App")
st.markdown("""
Dùng thanh **sidebar** để chuyển trang:

- **Overview Dashboard**: Tổng quan dữ liệu & hiệu năng.
- **Data Browser**: Xem dữ liệu *raw/processed/train/test*.
- **Data Dictionary**: Nguồn gốc, mô tả cột, mục tiêu, ưu/nhược.
- **Project Flow**: Quy trình của toàn bộ thư mục cha.
- **Predictor**: Dự đoán có việc (manual/random/from test/upload CSV).
- **Prediction History**: Xem & quản lý lịch sử dự đoán.
""")

st.info("Trước khi vào Predictor: chạy `python src/train.py` để tạo `models/pipeline_latest.pkl`.\n"
        "Muốn có các biểu đồ ROC/PR/Confusion: chạy `python src/evaluate.py`.")
