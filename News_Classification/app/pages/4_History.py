import streamlit as st, pandas as pd, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.append(str(PROJECT_ROOT))

from src.history import read_history, clear_history

st.set_page_config(page_title="Lịch sử", layout="wide")
st.title("🕓 Lịch sử thao tác")

hist = read_history()
if not hist:
    st.info("Chưa có lịch sử.")
else:
    st.dataframe(pd.DataFrame(hist))

if st.button("Xoá lịch sử"):
    clear_history()
    st.success("Đã xoá lịch sử. Refresh trang để cập nhật.")
