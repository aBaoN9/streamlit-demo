import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from datetime import datetime

# --------- CSS tuỳ chỉnh ----------
st.markdown("""
    <style>
        .main { background-color: #f4f6f8; }
        .block-container { padding: 2rem; }
        .title { font-size: 28px; font-weight: bold; color: #333333; }
        .subtitle { font-size: 20px; color: #555; }
        code { color: #c7254e; background-color: #f9f2f4; }
    </style>
""", unsafe_allow_html=True)

# ---------- Tiêu đề ----------
st.markdown('<div class="title">🧮 Ứng Dụng Giải Phương Trình</div>', unsafe_allow_html=True)

# ---------- Lý thuyết phương pháp Secant ----------
secant_theory = """
### 📘 Phương pháp dây cung (Secant)

Phương pháp dây cung là một kỹ thuật số để tìm nghiệm gần đúng của phương trình phi tuyến `f(x) = 0`.
Nó sử dụng hai điểm gần nghiệm `x0` và `x1` để dựng đường thẳng (dây cung) và xác định điểm cắt trục hoành tiếp theo.

**Công thức:**
"""

secant_formula = r"""
x_{n+1} = x_n - f(x_n) \cdot \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}
"""

# ---------- Phương pháp Secant ----------
def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    steps = [f"Bắt đầu với x0 = {x0}, x1 = {x1}"]
    for i in range(max_iter):
        f_x0 = f(x0)
        f_x1 = f(x1)
        denom = f_x1 - f_x0
        if abs(denom) < 1e-12:
            raise ValueError("Mẫu số gần 0 – thất bại.")
        x2 = x1 - f_x1 * (x1 - x0) / denom
        steps.append(f"Lặp {i+1}: x2 = {x2}, f(x2) = {f(x2)}")
        if abs(x2 - x1) < tol:
            steps.append(f"Hội tụ sau {i+1} lần lặp: x = {x2}")
            return x2, i + 1, steps
        x0, x1 = x1, x2
    raise ValueError("Không hội tụ sau số lần lặp tối đa.")

# ---------- Giải phương trình lượng giác ----------
def solve_trig_equation(eq_str, var_str='x', domain=(0, 2 * np.pi)):
    x = sp.symbols(var_str)
    steps = []

    try:
        lhs_str, rhs_str = eq_str.split('=')
        lhs = sp.sympify(lhs_str)
        rhs = sp.sympify(rhs_str)
        eq = sp.Eq(lhs, rhs)
    except Exception as e:
        st.error(f"Lỗi cú pháp: {e}")
        return None, []

    steps.append(f"Phương trình đã nhập: {eq_str}")
    steps.append(f"Chuyển về dạng: {sp.pretty(eq)}")
    simplified_eq = sp.simplify(lhs - rhs)
    steps.append(f"Rút gọn: f(x) = {simplified_eq}")

    try:
        sol = sp.solveset(eq, x, domain=sp.Interval(domain[0], domain[1]))
        steps.append(f"Nghiệm trong khoảng [{domain[0]}, {domain[1]}]: {sol}")
        return sol, steps
    except Exception as e:
        st.error(f"Lỗi khi giải: {e}")
        return None, steps

# ---------- Vẽ đồ thị ----------
def plot_function(f, x0, x1, width=2.0):
    x_vals = np.linspace(x0 - width, x1 + width, 400)
    y_vals = [f(x) for x in x_vals]
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label='f(x)')
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(x0, color='red', linestyle=':', label=f'x0 = {x0}')
    ax.axvline(x1, color='blue', linestyle=':', label=f'x1 = {x1}')
    ax.set_title("Đồ thị hàm số f(x)")
    ax.legend()
    st.pyplot(fig)

# ---------- Sidebar chức năng ----------
st.sidebar.title("📂 Chọn chức năng")
menu = st.sidebar.radio("", [
    "Giới thiệu", 
    "Giải phi tuyến (Secant)", 
    "Giải lượng giác", 
    "Bài tập ngẫu nhiên", 
    "Góp ý"
])

# ---------- Màn hình chính ----------
if menu == "Giới thiệu":
    st.markdown(secant_theory)
    st.latex(secant_formula)
    st.caption("Công thức dùng trong mỗi bước lặp để cập nhật nghiệm x.")

elif menu == "Giải phi tuyến (Secant)":
    st.subheader("📉 Giải phương trình phi tuyến bằng Secant")
    st.markdown(secant_theory)
    st.latex(secant_formula)

    expr = st.text_input("Nhập f(x)", "x**3 - x - 2")
    x0 = st.number_input("x0", value=1.0)
    x1 = st.number_input("x1", value=2.0)
    tol = st.number_input("Sai số cho phép", value=1e-6, format="%.1e")
    max_iter = st.number_input("Số lần lặp tối đa", value=100)

    if st.button("Giải"):
        if expr.strip() == "":
            st.warning("Vui lòng nhập biểu thức f(x).")
        elif x0 == x1:
            st.warning("Giá trị x0 và x1 không được trùng nhau.")
        else:
            try:
                f = lambda x: eval(expr, {"x": x, "np": np})
                plot_function(f, x0, x1)
                root, iterations, steps = secant_method(f, x0, x1, tol, int(max_iter))
                st.success(f"✅ Đã tìm được nghiệm gần đúng")
                st.markdown(fr"**x ≈ {root:.6f} \; \text{{(sau {iterations} lần lặp)}}**")
                st.markdown("### 📘 Các bước lặp:")
                for step in steps:
                    st.code(step)
            except Exception as e:
                st.error(f"Lỗi: {e}")

elif menu == "Giải lượng giác":
    st.subheader("📐 Giải phương trình lượng giác bằng phương pháp dây cung")

    if "trig_expr_buffer" not in st.session_state:
        st.session_state.trig_expr_buffer = "np.sin(x) - 0.5"

    st.markdown("**Chèn ký hiệu lượng giác vào f(x):**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    if col1.button("sin"):
        st.session_state.trig_expr_buffer += "np.sin("
    if col2.button("cos"):
        st.session_state.trig_expr_buffer += "np.cos("
    if col3.button("tan"):
        st.session_state.trig_expr_buffer += "np.tan("
    if col4.button("cot"):
        st.session_state.trig_expr_buffer += "1/np.tan("
    if col5.button("sec"):
        st.session_state.trig_expr_buffer += "1/np.cos("
    if col6.button("csc"):
        st.session_state.trig_expr_buffer += "1/np.sin("

    trig_expr = st.text_area("Nhập f(x)", value=st.session_state.trig_expr_buffer, height=100)

    x0 = st.number_input("x0 (rad)", value=0.0)
    x1 = st.number_input("x1 (rad)", value=1.0)
    tol = st.number_input("Sai số", value=1e-6, format="%.1e")
    max_iter = st.number_input("Số lần lặp tối đa", value=50)

    if st.button("Giải bằng Secant"):
        try:
            f = lambda x: eval(trig_expr, {"x": x, "np": np})
            plot_function(f, x0, x1, width=1.5)
            root, iterations, steps = secant_method(f, x0, x1, tol, int(max_iter))
            st.success(f"✅ Đã tìm được nghiệm gần đúng")
            st.markdown(fr"x \approx {root:.6f} \quad \text{{(sau {iterations} lần lặp)}}**")
            st.markdown("### 📘 Các bước lặp:")
            for step in steps:
                st.code(step)
        except Exception as e:
            st.error(f"Lỗi: {e}")

    if st.button("🔄 Reset f(x)"):
        st.session_state.trig_expr_buffer = ""

    st.subheader("📐 Giải phương trình lượng giác (dạng sin/cos = ?)")
    if "eq_input" not in st.session_state:
        st.session_state.eq_input = "sin(x) + cos(x) = 1"

    eq_input = st.text_input("Nhập phương trình", value=st.session_state.eq_input)
    domain_min = st.number_input("Từ (rad)", value=0.0)
    domain_max = st.number_input("Đến (rad)", value=2 * np.pi)

    if st.button("Giải lượng giác"):
        sol, steps = solve_trig_equation(eq_input, 'x', (domain_min, domain_max))
        if sol is not None:
            st.success(f"Nghiệm: {sol}")
            st.markdown("### 📘 Giải thích từng bước:")
            for s in steps:
                st.code(s)

elif menu == "Bài tập ngẫu nhiên":
    st.subheader("🎲 Bài tập ngẫu nhiên với phương pháp Secant")

    question_type = st.radio("Chọn loại bài toán", ["Phi tuyến", "Lượng giác"])

    if st.button("📌 Sinh bài tập"):
        if question_type == "Phi tuyến":
            nonlinear_problems = [
                ("x**2 - 4", 0.0, 3.0),
                ("x**3 - x - 2", 1.0, 2.0),
                ("np.exp(x) - 2", 0.0, 1.0),
                ("x**3 - 6*x**2 + 11*x - 6", 1.5, 3.5),
                ("np.log(x) - 1", 1.0, 3.0)
            ]
            expr, x0, x1 = nonlinear_problems[np.random.randint(0, len(nonlinear_problems))]
            st.markdown(f"**Bài toán:** Tìm nghiệm gần đúng của:")
            st.latex(f"f(x) = {expr}")
            st.markdown(f"Với: `x0 = {x0}`, `x1 = {x1}`")

            f = lambda x: eval(expr, {"x": x, "np": np, "math": np})
            try:
                plot_function(f, x0, x1)
                root, iterations, steps = secant_method(f, x0, x1)
                st.success(f"✅ Nghiệm gần đúng: x ≈ {root:.6f} (sau {iterations} lần lặp)")
                for step in steps:
                    st.code(step)
            except Exception as e:
                st.error(f"Lỗi: {e}")

        elif question_type == "Lượng giác":
            trig_problems = [
                ("np.sin(x) - 0.5", 0.0, 1.0),
                ("np.cos(x) - 0.5", 0.0, 2.0),
                ("np.sin(x) - np.cos(x)", 0.0, 2.0),
                ("np.sin(x) + 0.2*x", 0.0, 2.0),
                ("np.cos(x) - x", 0.0, 1.0)
            ]
            expr, x0, x1 = trig_problems[np.random.randint(0, len(trig_problems))]
            st.markdown(f"**Bài toán:** Tìm nghiệm gần đúng của:")
            st.latex(f"f(x) = {expr}")
            st.markdown(f"Với: `x0 = {x0}`, `x1 = {x1}`")

            f = lambda x: eval(expr, {"x": x, "np": np, "math": np})
            try:
                plot_function(f, x0, x1, width=1.5)
                root, iterations, steps = secant_method(f, x0, x1)
                st.success(f"✅ Nghiệm gần đúng: x ≈ {root:.6f} (sau {iterations} lần lặp)")
                for step in steps:
                    st.code(step)
            except Exception as e:
                st.error(f"Lỗi: {e}")

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
