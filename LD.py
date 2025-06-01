import sympy as sp
import numpy as np

def phuong_phap_lap_chi_tiet(f_str, a, b, x0, sai_so, max_iter=1000):
    # Khai báo biến và chuyển đổi hàm
    x = sp.symbols('x')
    f = sp.sympify(f_str)
    
    print("\n=== THÔNG TIN PHƯƠNG TRÌNH ===")
    print(f"Phương trình cần giải: f(x) = {f} = 0")
    
    # Tính đạo hàm f'(x)
    f_prime = sp.diff(f, x)
    print(f"\nĐạo hàm f'(x) = {f_prime}")
    
    # Tính M = max|f'(x)| trên [a,b]
    f_prime_func = sp.lambdify(x, f_prime)
    x_vals = np.linspace(a, b, 1000)
    M = max(abs(f_prime_func(x_vals)))
    print(f"\nGiá trị M = max|f'(x)| trên [{a}, {b}] = {M:.6f}")
    
    # Định nghĩa hàm lặp g(x) = x - f(x)/M
    g = x - f/M
    print(f"\nHàm lặp g(x) = x - f(x)/M = {g}")
    
    # Tính đạo hàm g'(x)
    g_prime = sp.diff(g, x)
    print(f"\nĐạo hàm g'(x) = {g_prime}")
    
    # Tính q = max|g'(x)| trên [a,b]
    g_prime_func = sp.lambdify(x, g_prime)
    q = max(abs(g_prime_func(x_vals)))
    print(f"\nGiá trị q = max|g'(x)| trên [{a}, {b}] = {q:.6f}")
    print(f"Điều kiện hội tụ q < 1: {'Thỏa mãn' if q < 1 else 'Không thỏa mãn'}")
    
    # Bắt đầu vòng lặp
    print("\n=== QUÁ TRÌNH LẶP ===")
    x_old = x0
    g_func = sp.lambdify(x, g)
    
    for i in range(1, max_iter+1):
        x_new = g_func(x_old)
        alpha = (q/(1-q)) * abs(x_new - x_old)
        
        print(f"\nLần lặp {i}:")
        print(f"x_old = {x_old:.8f}")
        print(f"x_new = g(x_old) = {x_new:.8f}")
        print(f"alpha = (q/(1-q))*|x_new - x_old| = {alpha:.8f}")
        
        if alpha < sai_so:
            print(f"\nKẾT THÚC: Đạt độ chính xác sau {i} lần lặp")
            print(f"Nghiệm gần đúng: {x_new:.8f}")
            return float(x_new)
        
        x_old = x_new
    
    print("\nCẢNH BÁO: Không hội tụ sau số lần lặp tối đa")
    return float(x_new)


print("=== GIẢI PHƯƠNG TRÌNH BẰNG PHƯƠNG PHÁP LẶP ===")
f_input = input("Nhập hàm f(x):")
a = float(input("Nhập a (đầu khoảng): "))
b = float(input("Nhập b (cuối khoảng): "))
x0 = float(input("Nhập x0 (giá trị ban đầu): "))
sai_so = float(input("Nhập sai số mong muốn (ví dụ 1e-2): "))

nghiem = phuong_phap_lap_chi_tiet(f_input, a, b, x0, sai_so)