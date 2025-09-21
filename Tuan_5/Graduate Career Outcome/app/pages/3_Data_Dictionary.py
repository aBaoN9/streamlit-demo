import streamlit as st

st.set_page_config(page_title="Data Dictionary", page_icon="📘", layout="centered")
st.title("📘 Data Dictionary & Project Notes")

st.markdown("""
**Nguồn gốc**: Campus Placement — file `Placement_Data_Full_Class.csv`.

**Mục tiêu**: Phân loại `status` (Placed/Not Placed).

**Ý nghĩa cột (tóm tắt)**  
- `gender`: M/F  
- `ssc_p, hsc_p, degree_p, etest_p, mba_p`: điểm %  
- `ssc_b, hsc_b`: loại trường  
- `hsc_s, degree_t, specialisation`: ngành/chuyên ngành  
- `workex`: Yes/No  
- `status`: target (Placed/Not Placed)  
- `salary`: chỉ có nếu Placed (bỏ khi train classification)

**Điểm mạnh**: dữ liệu gọn, rõ ràng; demo nhanh LR.  
**Điểm yếu**: size nhỏ; `salary` missing khi Not Placed; dễ lệch lớp.  
**Hướng phát triển**: calibration, fairness checks, thêm đặc trưng, thử nhiều mô hình.
""")
