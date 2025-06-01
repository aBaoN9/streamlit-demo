import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import random

from datetime import datetime

# ---------- Giao diện chung ----------
st.set_page_config(page_title="Ứng dụng Giải Phương Trình", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #f4f6f8; }
        .block-container { padding: 2rem; }
        .title { font-size: 2rem; font-weight: bold; color: #333; }
    </style>
""", unsafe_allow_html=True)

# ---------- Hàm Secant ----------
def secant_method(f, d, x1, tol=1e-6, max_iter=100):
    xs = [d, x1]
    steps = []
    for i in range(1, max_iter+1):
        x_prev, x_curr = xs[-2], xs[-1]
        denom = f(x_curr) - f(d)
        if abs(denom) < 1e-12:
            raise ValueError("Mẫu số gần bằng 0 – không thể tiếp tục.")
        x_next = x_curr - f(x_curr) * (x_curr - d) / denom
        xs.append(x_next)
        steps.append((i, x_prev, x_curr, x_next, f(x_next)))
        if abs(x_next - x_curr) < tol:
            return xs, steps
    raise ValueError("Không hội tụ sau tối đa số lặp.")

# ---------- Vẽ đồ thị ----------
def plot_function(f, d, x1, width=2.0):
    xs = np.linspace(min(d, x1)-width, max(d, x1)+width, 400)
    ys = np.array([f(v) for v in xs])
    fig, ax = plt.subplots()
    ax.plot(xs, ys, label='f(x)')
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(d, color='red', linestyle=':', label=f'd = {d}')
    ax.axvline(x1, color='blue', linestyle=':', label=f'x₁ = {x1}')
    ax.set_title('Đồ thị hàm số f(x)')
    ax.legend()
    st.pyplot(fig)

# ---------- Giải lượng giác ----------
def solve_trig_equation(eq_str, var='x', domain=(0, 2*np.pi)):
    x = sp.symbols(var)
    steps = []
    try:
        lhs, rhs = eq_str.split('=')
        eq = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
    except Exception as e:
        st.error(f"Lỗi cú pháp: {e}")
        return None, []
    steps.append(f"Phương trình: {eq}")
    expr = sp.simplify(sp.sympify(lhs) - sp.sympify(rhs))
    steps.append(f"Rút gọn: f(x) = {expr}")
    try:
        sol = sp.solveset(eq, x, domain=sp.Interval(*domain))
        steps.append(f"Nghiệm trong [{domain[0]}, {domain[1]}]: {sol}")
        return sol, steps
    except Exception as e:
        st.error(f"Không giải được: {e}")
        return None, steps

# ---------- Giao diện chính ----------
st.sidebar.title("📂 Chức năng")
menu = st.sidebar.radio("Chức năng", 
                       ["Giới thiệu", "Secant một đầu cố định", "Giải lượng giác", "Bài tập ngẫu nhiên", "Góp ý"], 
                       index=0)

# === Mục Giới thiệu ===
if menu == "Giới thiệu":
    st.markdown('<div class="title">Ứng Dụng Giải Phương Trình</div>', unsafe_allow_html=True)
    st.markdown('### 📘 Phương pháp dây cung một đầu cố định')
    st.write('Phương pháp sử dụng một điểm cố định **d** (chọn theo dấu của f và f"), chỉ cập nhật xₙ.')
    st.markdown('---')
    st.markdown('#### 🔢 Công thức')
    st.latex(r"x_{n+1} = x_n - \frac{f(x_n)\,(x_n - d)}{f(x_n) - f(d)}")
    st.markdown('---')
    st.markdown('#### 📏 Đánh giá sai số (chọn 1 trong 2)')
    st.write('1) Nếu tồn tại m₁>0 sao cho |f\'(x)| >= m₁ với mọi x in [a,b], thì:')
    st.latex(r"|x_n - \alpha| \le \frac{|f(x_n)|}{m_1}")
    st.write('2) Nếu tồn tại m₁, M₁ > 0 sao cho m₁ <= |f\'(x)| <= M₁ với mọi x in [a,b], thì:')
    st.latex(r"|x_n - \alpha| \le \frac{M_1 - m_1}{m_1}|x_n - x_{n-1}|")

# === Mục Secant ===
elif menu == "Secant một đầu cố định":
    st.markdown('<div class="title">Secant Một Đầu Cố Định</div>', unsafe_allow_html=True)
    st.write('Nhập hàm f(x) và khoảng [a, b]')
    expr = st.text_input('Hàm f(x)', 'x**5 - x - 10')
    a = st.number_input('a', 1.5)
    b = st.number_input('b', 2.0)
    tol = st.number_input('Sai số tol', 1e-4, format='%.0e')
    max_iter = st.number_input('Số lặp tối đa', 100)
    method = st.selectbox('Chọn đánh giá sai số', ['|f(x_n)|/m1', '(M1-m1)/m1 * |x_n - x_{n-1}|'])

    if st.button('Giải'):
        x = sp.symbols('x')
        f_sym = sp.sympify(expr)
        f = lambda v: float(sp.N(f_sym.subs(x, v)))
        f1 = sp.diff(f_sym, x)
        f2 = sp.diff(f1, x)
        f1f = sp.lambdify(x, f1, 'numpy')
        f2f = sp.lambdify(x, f2, 'numpy')

        # Tính m1, M1
        xsamp = np.linspace(a, b, 1001)
        derivs = np.abs(f1f(xsamp))
        m1 = np.min(derivs)
        M1 = np.max(derivs)
        st.write(f'Tính m₁ = {m1:.6f}, M₁ = {M1:.6f} trên [{a},{b}]')

        # Chọn d và điểm khởi tạo thứ hai
        d = a if f(a)*f2f(a) > 0 else b
        x1_init = b if d == a else a

        # Secant lặp
        xs, steps = secant_method(f, d, x1_init, tol, max_iter)
        root, prev = xs[-1], xs[-2]

        st.markdown('**Các bước lặp:**')
        for i, xp, xc, xn, fxn in steps:
            st.write(f'Bước {i}: xₙ₋₁={xp:.6f}, xₙ={xc:.6f} → xₙ₊₁={xn:.6f}, f={fxn:.2e}')

        st.markdown('**Đánh giá sai số:**')
        if method == '|f(x_n)|/m1':
            err = abs(f(root))/m1
            st.write(f'|xₙ - α| ≤ |f(xₙ)|/m₁ = {abs(f(root)):.2e}/{m1:.6f} = {err:.2e}')
        else:
            err = (M1-m1)/m1 * abs(root-prev)
            st.write(f'|xₙ - α| ≤ (M₁-m₁)/m₁·|Δx| = ({M1:.6f}-{m1:.6f})/{m1:.6f}·{abs(root-prev):.2e} = {err:.2e}')

        st.success(f'Vậy nghiệm gần đúng α ≈ {root:.6f} ± {err:.2e}')
        plot_function(f, d, x1_init)

# === Mục Lượng giác ===
# === Mục Lượng giác (Cải tiến) ===
elif menu == "Giải lượng giác":
    st.markdown('<div class="title">🧮 Giải Phương Trình Lượng Giác</div>', unsafe_allow_html=True)
    st.markdown("#### ✏️ Nhập phương trình lượng giác (dạng `f(x) = g(x)`)")

    if 'trig_expr' not in st.session_state:
        st.session_state.trig_expr = 'sin(x) - 0.5'

    expr_input = st.text_input("Phương trình", st.session_state.trig_expr)
    st.session_state.trig_expr = expr_input

    st.markdown("#### 💡 Mẫu nhanh")
    cols = st.columns(4)
    samples = ['sin(x) - 0.5', 'cos(x) - x', 'tan(x) - 1', '2*sin(x) - 1', 'sin(2*x)', 'cos(x)**2 - 0.25']
    for i, val in enumerate(samples):
        if cols[i % 4].button(val):
            st.session_state.trig_expr = val

    st.divider()

    st.markdown("#### 📍 Khoảng tìm nghiệm và tùy chọn phương pháp")
    a = st.number_input('a (trái)', 0.0)
    b = st.number_input('b (phải)', np.pi)
    tol = st.number_input('Sai số (tol)', 1e-5, format='%.0e')
    max_iter = st.number_input('Số lặp tối đa', 100)
    err_method = st.selectbox('Chọn công thức đánh giá sai số:', ['|f(xₙ)|/m₁', '(M₁-m₁)/m₁ × |Δx|'])

    col_bt1, col_bt2 = st.columns(2)
    trig_eq = st.session_state.trig_expr

    with col_bt1:
        if st.button("⚙️ Giải bằng phương pháp Secant"):
            try:
                x = sp.symbols('x')
                f_sym = sp.sympify(trig_eq)
                f = lambda v: float(sp.N(f_sym.subs(x, v)))
                f1 = sp.diff(f_sym, x)
                f2 = sp.diff(f1, x)
                f1f = sp.lambdify(x, f1, 'numpy')
                f2f = sp.lambdify(x, f2, 'numpy')

                # Tính m1, M1
                xsamp = np.linspace(a, b, 1001)
                derivs = np.abs(f1f(xsamp))
                m1, M1 = np.min(derivs), np.max(derivs)
                st.info(f'm₁ = {m1:.6f}, M₁ = {M1:.6f} trên [{a:.2f}, {b:.2f}]')

                d = a if f(a)*f2f(a) > 0 else b
                x1 = b if d == a else a

                xs, steps = secant_method(f, d, x1, tol, max_iter)
                root, prev = xs[-1], xs[-2]

                st.markdown("### 🔄 Các bước lặp")
                for i, xp, xc, xn, fx in steps:
                    st.write(f"Bước {i}: xₙ₋₁ = {xp:.6f}, xₙ = {xc:.6f} → xₙ₊₁ = {xn:.6f}, f(xₙ₊₁) = {fx:.2e}")

                st.markdown("### 📏 Đánh giá sai số")
                if err_method == '|f(xₙ)|/m₁':
                    err = abs(f(root)) / m1
                    st.write(f"|xₙ - α| ≤ {abs(f(root)):.2e}/{m1:.6f} = {err:.2e}")
                else:
                    err = (M1 - m1) / m1 * abs(root - prev)
                    st.write(f"|xₙ - α| ≤ ({M1:.6f} - {m1:.6f})/{m1:.6f} × {abs(root-prev):.2e} = {err:.2e}")

                st.success(f"Nghiệm gần đúng: α ≈ {root:.6f} ± {err:.2e}")
                plot_function(f, d, x1)
            except Exception as e:
                st.error(f"Lỗi: {e}")

    with col_bt2:
        if st.button("📐 Giải bằng SymPy (tượng trưng)"):
            sol, steps = solve_trig_equation(trig_eq, var='x', domain=(a, b))
            if sol:
                st.markdown("### 📜 Các bước giải tượng trưng")
                for s in steps:
                    st.write("•", s)
                st.success(f"Nghiệm trong [{a}, {b}]: {sol}")
            else:
                st.warning("Không tìm được nghiệm tượng trưng hoặc cú pháp sai.")

# === Bài tập ngẫu nhiên ===
elif menu == "Bài tập ngẫu nhiên":
    st.markdown('<div class="title">Bài Tập Ngẫu Nhiên</div>', unsafe_allow_html=True)

    problems = [
        (lambda x: x**3 - x - 2, (1,2), 'x³ - x - 2 = 0'),
        (lambda x: np.exp(x) - 3, (0,2), 'eˣ - 3 = 0'),
        (lambda x: np.log(x) - 1, (1,3), 'ln(x) - 1 = 0'),
        (lambda x: x**2 - 2, (0,2), 'x² - 2 = 0'),
        (lambda x: np.cos(x) - x, (0,1), 'cos(x) - x = 0'),
        (lambda x: np.sin(x) - 0.5, (0,np.pi), 'sin(x) - 0.5 = 0')
    ]

    if 'prob' not in st.session_state:
        st.session_state.prob = random.choice(problems)

    if st.button("🔄 Bài tập mới"):
        st.session_state.prob = random.choice(problems)

    f_func, (x0, x1), desc = st.session_state.prob
    st.write(f"**Bài toán:** {desc}")

    tol2 = st.number_input('Sai số tol', 1e-6, format='%.0e', key='tol2')
    max2 = st.number_input('Max lặp', 100, key='max2')

    if st.button('Giải (Random)'):
        try:
            xs, steps = secant_method(f_func, x0, x1, tol2, max2)
            root, prev = xs[-1], xs[-2]
            st.success(f'Nghiệm: {root:.6f}')
            st.markdown('**Các bước:**')
            for i, xp, xc, xn, fx in steps:
                st.write(f'{i}: xₙ₋₁={xp:.6f}, xₙ={xc:.6f} → xₙ₊₁={xn:.6f}, f={fx:.2e}')
            st.success(f'Kết luận: α ≈ {root:.6f} ± {abs(root-prev):.2e}')
            plot_function(f_func, x0, x1)
        except Exception as e:
            st.error(f'Lỗi: {e}')



# === Góp ý ===
elif menu == "Góp ý":
    st.markdown('<div class="title">Góp Ý</div>', unsafe_allow_html=True)
    fb = st.text_area('Ý kiến của bạn:')
    if st.button('Gửi góp ý'):
        if fb.strip():
            with open('feedback.txt', 'a', encoding='utf-8') as f:
                f.write(fb.strip() + '\n---\n')
            st.success('Cảm ơn bạn đã góp ý!')
        else:
            st.warning('Vui lòng nhập góp ý.')
