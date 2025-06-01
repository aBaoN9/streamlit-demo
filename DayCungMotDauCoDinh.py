import sympy as sp
import numpy as np

def phuong_phap_day_cung(f_str, a, b, sai_so, max_iter=1000):
    # Khởi tạo biến và kiểm tra điều kiện
    x = sp.symbols('x')
    f = sp.sympify(f_str)
    f_prime = sp.diff(f, x)
    f_double_prime = sp.diff(f_prime, x)
    
    print("\n=== THÔNG TIN PHƯƠNG TRÌNH ===")
    print(f"Phương trình cần giải: f(x) = {f} = 0")
    print(f"Đạo hàm f'(x) = {f_prime}")
    print(f"Đạo hàm cấp 2 f''(x) = {f_double_prime}")
    
    # Kiểm tra điều kiện tồn tại nghiệm
    f_a = f.subs(x, a).evalf()
    f_b = f.subs(x, b).evalf()
    
    if f_a * f_b >= 0:
        print("\nCẢNH BÁO: f(a) và f(b) không trái dấu - không nghiệm trên [a,b]")
        return None
    
    # Kiểm tra dấu của đạo hàm
    f_prime_func = sp.lambdify(x, f_prime)
    f_double_prime_func = sp.lambdify(x, f_double_prime)
    
    x_vals = np.linspace(a, b, 1000)
    f_prime_vals = f_prime_func(x_vals)
    f_double_prime_vals = f_double_prime_func(x_vals)
    
    if not (np.all(f_prime_vals > 0) or np.all(f_prime_vals < 0)):
        print("\nCẢNH BÁO: f'(x) đổi dấu trên [a,b]")
        return None
    
    if not (np.all(f_double_prime_vals > 0) or np.all(f_double_prime_vals < 0)):
        print("\nCẢNH BÁO: f''(x) đổi dấu trên [a,b]")
        return None
    
    # Xác định đầu cố định d
    if f_b * f_double_prime_vals[0] > 0:
        d = b
        x_old = a
    else:
        d = a
        x_old = b
    
    print(f"\nĐầu cố định d = {d:.8f}")
    print(f"Giá trị ban đầu x0 = {x_old:.8f}")
    
    # Tính min và max của f'(x)
    min_f_prime = min(abs(f_prime_vals))
    max_f_prime = max(abs(f_prime_vals))
    
    print(f"\nmin|f'(x)| = {min_f_prime:.8f}")
    print(f"max|f'(x)| = {max_f_prime:.8f}")
    
    # Chuyển đổi hàm
    f_func = sp.lambdify(x, f)
    
    print("\n=== QUÁ TRÌNH LẶP ===")
    for i in range(1, max_iter + 1):
        numerator = f_func(x_old) * (x_old - d)
        denominator = f_func(x_old) - f_func(d)
        x_new = x_old - numerator / denominator
        
        alpha = ((max_f_prime - min_f_prime) / min_f_prime) * abs(x_new - x_old)
        
        print(f"\nLần lặp {i}:")
        print(f"x_old = {x_old:.8f}")
        print(f"x_new = {x_new:.8f}")
        print(f"alpha = {alpha:.8f}")
        
        if alpha < sai_so:
            print(f"\nKẾT THÚC: Đạt độ chính xác sau {i} lần lặp")
            print(f"Nghiệm gần đúng: {x_new:.8f}")
            return float(x_new)
        
        x_old = x_new
    
    print("\nCẢNH BÁO: Không hội tụ sau số lần lặp tối đa")
    return float(x_new)


print("=== PHƯƠNG PHÁP DÂY CUNG (MỘT ĐẦU CỐ ĐỊNH) ===")
f_input = input("Nhập hàm f(x): ")
a = float(input("Nhập a (đầu khoảng): "))
b = float(input("Nhập b (cuối khoảng): "))
sai_so = float(input("Nhập sai số mong muốn (ví dụ 1e-6): "))

nghiem = phuong_phap_day_cung(f_input, a, b, sai_so)