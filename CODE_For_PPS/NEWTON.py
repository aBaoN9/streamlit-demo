import sympy as sp




# Phương pháp Newton-Raphson
def newton_method(f_expr, x0, ex=1e-5, max_iter=100):
    # Tạo ký hiệu đại số x
    x = sp.symbols('x')
    
    # Tính đạo hàm bậc 1 của f(x)
    f = sp.sympify(f_expr)
    f1 = sp.diff(f, x)  # Đạo hàm bậc 1

    x_value = x0  # giá trị khởi tạo

    for i in range(max_iter):
        fx_value = f.subs(x, x_value)  # Giá trị hàm tại x_value
        f1x_value = f1.subs(x, x_value)  # Giá trị đạo hàm bậc 1 tại x_value

        if f1x_value == 0:  # Tránh chia cho 0
            print("Đạo hàm bậc 1 bằng 0, không thể tiếp tục")
            return None
        
        # Tính x mới theo công thức Newton
        x_new = x_value - fx_value / f1x_value

        # Kiểm tra điều kiện dừng (xấp xỉ nghiệm và sai số nhỏ hơn ex)
        if abs(x_new - x_value) < ex:
            print(f"Phương pháp hội tụ sau {i+1} lần lặp.")
            return x_new
        
        x_value = x_new
    
    print("Không hội tụ sau số lần lặp tối đa.")
    return x_value

# Định nghĩa bài toán
f_expr = input("Nhập hàm f(x): ")
x0 = float(input("Nhập giá trị x0: "))
ex = float(input("Nhập sai số ex (ví dụ: 1e-5): "))

# Áp dụng phương pháp Newton
result = newton_method(f_expr, x0, ex)
if result is not None:
    print(f"Nghiệm gần đúng là: {result}")
