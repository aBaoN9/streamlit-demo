import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import random
import os
import json
import streamlit as st
import tkinter as tk
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load và lưu biểu thức
def load_expressions():
    if os.path.exists("expressions.json"):
        with open("expressions.json", "r") as f:
            return json.load(f)
    return ["cos(x)", "exp(-x)", "(-x**3 + 12*x + 1) / 11", "0.5 * (x + 3/x)", "1 / (1 + x)", "sqrt(2 + x)"]

def save_expressions(expressions):
    with open("expressions.json", "w") as f:
        json.dump(expressions, f, indent=2)

# Định dạng số theo khoa học
import numpy as np

def format_positive_sci_notation(x):
    if not np.isfinite(x):
        return r"$\infty$"
    
    base, exponent = f"{x:.2e}".split("e")
    base = float(base)
    exponent = int(exponent)
    
    # Trả về định dạng khoa học với mũ được hiển thị rõ ràng
    return rf"${base:.2f} \times 10^{{{exponent}}}$"


# Tính hệ số co q
def tinh_q(g_expr_str, a, b):
    x = sp.symbols('x')
    try:
        g_expr = sp.sympify(g_expr_str)
    except Exception:
        return float('inf')
    g_prime = sp.diff(g_expr, x)
    g_prime_lambda = sp.lambdify(x, g_prime, modules=["numpy"])
    sample_points = np.linspace(a, b, 200)
    # Lọc bỏ điểm x=0 (hoặc rất gần 0) để tránh chia cho 0
    sample_points = sample_points[np.abs(sample_points) > 1e-12]
    q_values = []
    for pt in sample_points:
        try:
            val = g_prime_lambda(pt)
            if np.isfinite(val):
                q_values.append(abs(val))
        except:
            continue
    return max(q_values) if q_values else float('inf')

def so_sanh_chu_so_thap_phan(a, b, tol_exp):
    a = float(a)
    b = float(b)
    str_a = f"{a:.{tol_exp + 5}f}"
    str_b = f"{b:.{tol_exp + 5}f}"
    dec_a = str_a.split(".")[1]
    dec_b = str_b.split(".")[1]
    count = 0
    for ca, cb in zip(dec_a, dec_b):
        if ca == cb:
            count += 1
        else:
            break
    return count >= tol_exp
# Phương pháp lặp đơn
def lap_don_chi_tiet(g_expr_str, x0, tol=1e-4, max_iter=50, a=None, b=None,tol_exp=4):
    x = sp.symbols('x')
    try:
        g_expr = sp.sympify(g_expr_str)
    except Exception:
        raise ValueError("Biểu thức φ(x) không hợp lệ!")

    g_lambda = sp.lambdify(x, g_expr, modules=["numpy"])
    q = tinh_q(g_expr_str, a, b) if a is not None and b is not None else float('inf')
    steps = []
    table = []
    xn_1 = x0

    for n in range(1, max_iter + 1):
        try:
            xn = g_lambda(xn_1)
            if isinstance(xn, complex):
                raise ValueError("Nghiệm phức, không xử lý được.")
        except Exception:
            raise ValueError(f"Không thể tính φ({xn_1})")

        diff = abs(xn - xn_1)
        error_est = float('inf') if q >= 1 else q / (1 - q) * diff

        steps.append(f"Lần lặp {n}: xₙ = φ({xn_1:.6f}) = {xn:.6f}")
        steps.append(f"         |xₙ - xₙ₋₁| = {diff:.6f}")
        steps.append(f"         q = {q:.6f}")
        steps.append(f"         Ước lượng sai số ≈ {error_est:.2e}\n")

        table.append({
            "Lần lặp (n)": n,
            "xₙ = φ(xₙ₋₁)": xn,
            "|xₙ - xₙ₋₁|": diff,
            "q/(1-q)|xₙ - xₙ₋₁|": error_est,
            "Ước lượng sai số": error_est
        })

        if so_sanh_chu_so_thap_phan(xn, xn_1, tol_exp):
            steps.append(f"✅ Dừng vì phần thập phân giống nhau ít nhất {tol_exp} chữ số.")
            return xn, steps, table, q

        xn_1 = xn

    steps.append("⚠️ Dừng vì đạt số lần lặp tối đa.")
    return xn, steps, table, q

# ===== Streamlit App =====
st.set_page_config(page_title="Phương Pháp Lặp đơn", layout="wide")
st.sidebar.title("📚 Menu")
menu = st.sidebar.radio("Chọn chức năng", [
    "Giới thiệu",
    "Phương pháp Lặp đơn",
    "Luyện tập (random)",
    "Góp ý"
])

st.title("🧮 Ứng dụng Giải Phương Trình Bằng Lặp Đơn")

if menu == "Giới thiệu":
    st.header("📌 Giới thiệu")
    st.markdown("""
Phương pháp **lặp đơn** (Fixed-point Iteration) dùng để giải phương trình dạng:

$$
x = \\varphi(x)
$$

Ta bắt đầu với một giá trị khởi đầu \( x_0 \), sau đó lặp lại theo công thức:

$$
x_{n+1} = \\varphi(x_n)
$$

cho đến khi dãy số hội tụ đến nghiệm gần đúng.

---

### ✅ Điều kiện hội tụ:
- Hàm $\\varphi(x)$ liên tục trên khoảng đang xét.  
- Đạo hàm của hàm $\\varphi(x)$ thỏa mãn điều kiện:

$$
|\\varphi'(x)| < 1
$$

trên toàn bộ khoảng. Khi đó, phương pháp đảm bảo hội tụ.
""")




elif menu == "Phương pháp Lặp đơn":
    st.header("🔁 Phương pháp Lặp đơn")

    if "reset" not in st.session_state:
        st.session_state.reset = False

    if st.button("🔄 Reset nhập liệu"):
        st.session_state.reset = True

    if st.session_state.reset:
        g_input = ""
        x0 = 0.0
        tol_exp = 4
        max_iter = 50
        a = None
        b = None
        st.session_state.reset = False
    else:
        g_input = st.text_input("Nhập biểu thức φ(x):", "")
        x0 = st.number_input("Giá trị khởi tạo x₀:", value=0.0, format="%.6f")
        tol_exp = st.slider("Sai số ε = 10⁻ⁿ", min_value=1, max_value=100, value=4)
        max_iter = st.number_input("Số lần lặp tối đa:", value=50, step=1, format="%d")
        
        a_input = st.text_input("Nhập a (trái đoạn [a, b]) (để trống nếu không biết):", "")
        b_input = st.text_input("Nhập b (phải đoạn [a, b]) (để trống nếu không biết):", "")
        
        # Xử lý nhập a,b
        try:
            a = float(a_input) if a_input.strip() != "" else None
        except:
            a = None
        try:
            b = float(b_input) if b_input.strip() != "" else None
        except:
            b = None

    tol = 10 ** (-tol_exp)

    if st.button("📀 Bắt đầu lặp"):
        if max_iter <= 0:
            st.error("Số lần lặp tối đa phải lớn hơn 0!")
        elif g_input.strip() == "":
            st.error("Vui lòng nhập biểu thức φ(x)!")
        else:
            if a is None or b is None:
                a = x0 - 1
                b = x0 + 1
            try:
                result, steps, table, q = lap_don_chi_tiet(g_input, x0, tol, max_iter, a, b, tol_exp)
                if np.isfinite(q):
                    if q >= 1:
                        st.warning(f"⚠️ Hệ số co q = {q:.6f} ≥ 1 → Có thể không hội tụ.")
                    else:
                        st.success(f"✅ Hệ số co q = {q:.6f} < 1 → Có khả năng hội tụ.")
                else:
                    st.warning("⚠️ Không thể tính hệ số co q.")

                st.success(f"✨ Nghiệm gần đúng: x = {result:.6f}")

                st.subheader("📄 Các bước chi tiết:")
                for s in steps:
                    st.text(s)

                st.subheader("📊 Bảng kết quả:")
                df = pd.DataFrame(table)
                df["xₙ = φ(xₙ₋₁)"] = df["xₙ = φ(xₙ₋₁)"].map(lambda x: f"{x:.6f}")
                df["|xₙ - xₙ₋₁|"] = df["|xₙ - xₙ₋₁|"].map(format_positive_sci_notation)
                df["q/(1-q)|xₙ - xₙ₋₁|"] = df["q/(1-q)|xₙ - xₙ₋₁|"].map(format_positive_sci_notation)
                df["Ước lượng sai số"] = df["Ước lượng sai số"].map(format_positive_sci_notation)
                st.table(df)
                

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")


elif menu == "Luyện tập (random)":
    st.header("🎯 Luyện tập phương pháp lặp đơn")

    if "expressions" not in st.session_state:
        st.session_state.expressions = load_expressions()

    if "random_expr" not in st.session_state:
        st.session_state.random_expr = random.choice(st.session_state.expressions)

    with st.expander("➕ Thêm bài mới"):
        with st.form("add_expr_form"):
            new_expr = st.text_input("Nhập biểu thức mới:", "")
            if st.form_submit_button("➕ Thêm bài"):
                if not new_expr.strip():
                    st.warning("⚠️ Vui lòng nhập biểu thức!")
                else:
                    try:
                        sp.sympify(new_expr)
                        if new_expr in st.session_state.expressions:
                            st.warning("❗ Biểu thức đã tồn tại!")
                        else:
                            st.session_state.expressions.append(new_expr)
                            save_expressions(st.session_state.expressions)
                            st.success("✅ Đã thêm biểu thức mới!")
                    except Exception as e:
                        st.error(f"❌ Biểu thức không hợp lệ: {e}")

    st.subheader("📘 Bài tập ngẫu nhiên:")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**φ(x)** = {st.session_state.random_expr}")
    with col2:
        if st.button("🎲 Bài tập mới"):
            st.session_state.random_expr = random.choice(st.session_state.expressions)
            st.rerun()

    # Tạo khoảng [a, b] ngẫu nhiên
    length = random.randint(1, 5)
    a, b = length - 1, length
    x0 = (a + b) / 2
    st.write(f"🔁 Khoảng ngẫu nhiên: [a = {a}, b = {b}], x₀ = {x0:.6f}")

    # Cho phép người dùng tùy chỉnh sai số và số lần lặp
    tol_exp = st.slider("Chọn sai số ε = 10⁻ⁿ", min_value=0, max_value=100, value=6)
    tol = 10 ** (-tol_exp)
    max_iter = st.number_input("Số lần lặp tối đa:", min_value=1, max_value=50, value=50, step=1)

    if st.button("🧪 Thực hiện giải"):
        try:
            result, steps, table, q = lap_don_chi_tiet(st.session_state.random_expr, x0, tol, max_iter, a, b, tol_exp)
            if q >= 1:
                st.warning(f"⚠️ q = {q:.6f} ≥ 1 → Có thể không hội tụ.")
            else:
                st.info(f"✅ q = {q:.6f} < 1 → Có khả năng hội tụ.")
            st.success(f"✨ Kết quả: x = {result:.6f}")

            st.subheader("🔍 Chi tiết:")
            for s in steps:
                st.text(s)

            st.subheader("📊 Bảng kết quả:")
            df = pd.DataFrame(table)
            for col in ["xₙ = φ(xₙ₋₁)", "|xₙ - xₙ₋₁|", "q/(1-q)|xₙ - xₙ₋₁|", "Ước lượng sai số"]:
                df[col] = df[col].map(format_positive_sci_notation)
            st.table(df)

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")


elif menu == "Góp ý":
    st.header("💌 Góp ý cho ứng dụng")

    name = st.text_input("👤 Họ và tên:")
    email = st.text_input("📧 Địa chỉ email:")
    feedback = st.text_area("✍️ Ý kiến hoặc phản hồi của bạn:")

    if "feedback_sent" not in st.session_state:
        st.session_state.feedback_sent = False

    if st.button("📬 Gửi góp ý"):
        if not name.strip():
            st.warning("⚠️ Vui lòng nhập họ và tên.")
        elif not email.strip():
            st.warning("⚠️ Vui lòng nhập địa chỉ email.")
        elif not feedback.strip():
            st.warning("⚠️ Vui lòng nhập ý kiến góp ý.")
        else:
            try:
                with open("feedback.txt", "a", encoding="utf-8") as f:
                    f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Họ tên: {name.strip()}\n")
                    f.write(f"Email: {email.strip()}\n")
                    f.write(f"Góp ý: {feedback.strip()}\n")
                    f.write("---\n")
                
                st.session_state.feedback_sent = True

            except Exception as e:
                st.error(f"❌ Lỗi khi lưu góp ý: {e}")

    if st.session_state.feedback_sent:
        st.success("✅ Cảm ơn bạn đã góp ý! 💖")
        with st.expander("📄 Thông tin bạn đã gửi"):
            st.write(f"**Họ tên:** {name.strip()}")
            st.write(f"**Email:** {email.strip()}")
            st.write(f"**Nội dung góp ý:** {feedback.strip()}")
            st.write(f"**Thời gian gửi:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


 