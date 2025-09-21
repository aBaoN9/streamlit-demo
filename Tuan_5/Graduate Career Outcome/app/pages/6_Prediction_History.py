import streamlit as st
from components.utils import read_history, clear_history

st.set_page_config(page_title="Prediction History", page_icon="🧾", layout="wide")
st.title("🧾 Prediction History")

df = read_history()
st.caption(f"Total records: {len(df)}")
st.dataframe(df)

col1, col2 = st.columns(2)
with col1:
    st.download_button("Export CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="pred_history.csv", mime="text/csv")
with col2:
    if st.button("Clear History"):
        clear_history()
        st.success("Đã xoá lịch sử (app/pred_history.csv).")
