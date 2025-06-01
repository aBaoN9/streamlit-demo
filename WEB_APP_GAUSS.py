import numpy as np
import streamlit as st
import pandas as pd

# Thiết lập trang
st.set_page_config(
    page_title="Giải hệ phương trình Gauss",
    page_icon="🧮",
    layout="wide"
)

# Tiêu đề ứng dụng
st.title("🧮 Giải hệ phương trình tuyến tính bằng phương pháp Gauss")
st.markdown("---")

# Phần nhập liệu trong sidebar
with st.sidebar:
    st.header("⚙️ Nhập thông tin hệ phương trình")
    n = st.slider("Số ẩn của hệ", 2, 5, 3)
    
    st.subheader("Ma trận hệ số A:")
    A = np.zeros((n, n))
    for i in range(n):
        cols = st.columns(n)
        for j in range(n):
            A[i,j] = cols[j].number_input(f"A[{i+1},{j+1}]", value=float(1 if i==j else 0), key=f"a_{i}_{j}")
    
    st.subheader("Vector hệ số tự do b:")
    b = np.zeros(n)
    for i in range(n):
        b[i] = st.number_input(f"b[{i+1}]", value=float(i+1), key=f"b_{i}")

# Hàm giải Gauss với bảng hội tụ
def gauss_elimination_with_table(A, b):
    n = len(b)
    Ab = np.column_stack((A.astype(float), b.astype(float)))
    convergence_data = []
    
    # Thêm trạng thái ban đầu
    initial_x = np.zeros(n)
    convergence_data.append({
        "Bước": 0,
        **{f"x[{i+1}]": initial_x[i] for i in range(n)},
        "Sai số": "-",
        "Số điều kiện": np.linalg.cond(A)
    })
    
    for step in range(n):
        # Pivotting
        max_row = np.argmax(np.abs(Ab[step:, step])) + step
        Ab[[step, max_row]] = Ab[[max_row, step]]
        
        # Chuẩn hóa hàng chính
        pivot = Ab[step, step]
        if np.isclose(pivot, 0):
            st.error("Ma trận suy biến trong quá trình tính toán!")
            return None, None, None
        
        Ab[step] = Ab[step] / pivot
        
        # Khử các hàng dưới
        for row in range(step + 1, n):
            Ab[row] = Ab[row] - Ab[row, step] * Ab[step]
        
        # Tính nghiệm tạm thời
        temp_x = np.zeros(n)
        for i in range(step, -1, -1):
            temp_x[i] = Ab[i, -1] - np.dot(Ab[i, i+1:n], temp_x[i+1:])
        
        # Tính sai số
        error = np.max(np.abs(temp_x - [convergence_data[-1][f"x[{i+1}]"] for i in range(n)])) if step > 0 else "-"
        
        convergence_data.append({
            "Bước": step+1,
            **{f"x[{i+1}]": temp_x[i] for i in range(n)},
            "Sai số": f"{error:.4e}" if step > 0 else "-",
            "Số điều kiện": np.linalg.cond(Ab[:, :-1])
        })
    
    # Thế ngược cuối cùng
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])
    
    return x, pd.DataFrame(convergence_data), Ab

# Hiển thị kết quả
if st.button("🔍 Giải hệ phương trình", type="primary"):
    st.markdown("---")
    
    # Kiểm tra điều kiện
    det_A = np.linalg.det(A)
    cond_A = np.linalg.cond(A)
    
    col1, col2 = st.columns(2)
    col1.metric("Định thức det(A)", f"{det_A:.6f}")
    col2.metric("Số điều kiện cond(A)", f"{cond_A:.6f}")
    
    if np.isclose(det_A, 0):
        st.error("Hệ phương trình suy biến (det(A) ≈ 0)!")
        st.stop()
    
    # Giải hệ
    with st.spinner("Đang giải hệ phương trình..."):
        x, convergence_df, final_Ab = gauss_elimination_with_table(A, b)
    
    if x is None:
        st.stop()
    
    # Hiển thị nghiệm
    st.subheader("🎯 Nghiệm của hệ phương trình")
    cols = st.columns(n)
    for i in range(n):
        cols[i].metric(label=f"x[{i+1}]", value=f"{x[i]:.8f}")
    
    # Bảng hội tụ
    st.markdown("---")
    st.subheader("📊 Bảng quá trình hội tụ")
    st.dataframe(
        convergence_df,
        width=1000,
        height=400,
        hide_index=True,
        column_config={
            "Bước": st.column_config.NumberColumn(format="%d"),
            **{f"x[{i+1}]": st.column_config.NumberColumn(format="%.6f") for i in range(n)},
            "Sai số": st.column_config.TextColumn(),
            "Số điều kiện": st.column_config.NumberColumn(format="%.2f")
        }
    )
    
    # Kiểm tra nghiệm
    st.subheader("✅ Kiểm tra nghiệm")
    residual = np.dot(A, x) - b
    st.write("**Sai số:** A*x - b =")
    st.dataframe(
        pd.DataFrame(residual, columns=["Sai số"]),
        width=300
    )

# Hướng dẫn
with st.expander("📖 Hướng dẫn sử dụng"):
    st.write("""
    1. Nhập số ẩn và các hệ số
    2. Nhấn nút **Giải hệ phương trình**
    3. Xem:
       - Nghiệm của hệ
       - Bảng quá trình hội tụ
       - Kiểm tra sai số nghiệm
    """)

st.markdown("---")
st.caption("© 2023 - Ứng dụng giải hệ phương trình tuyến tính")