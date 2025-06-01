import numpy as np
from tabulate import tabulate

def nhap_he_phuong_trinh():
    n = int(input("Nhập số ẩn của hệ phương trình: "))
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    print("\nNhập các hệ số của ma trận A:")
    for i in range(n):
        for j in range(n):
            A[i, j] = float(input(f"A[{i+1}][{j+1}] = "))
    
    print("\nNhập các hệ số tự do của vector b:")
    for i in range(n):
        b[i] = float(input(f"b[{i+1}] = "))
    
    return A, b

def gauss_elimination(A, b):
    n = len(b)
    
    # Tạo ma trận B và vector C từ A và b
    B = np.zeros((n, n))
    C = np.zeros(n)
    for i in range(n):
        C[i] = b[i] / A[i, i]
        for j in range(n):
            if j != i:
                B[i, j] = -A[i, j] / A[i, i]
    
    B_norm = np.linalg.norm(B, np.inf)
    q = B_norm if B_norm < 1 else None
    
    # Tạo ma trận mở rộng [A|b]
    Ab = np.column_stack((A.astype(float), b.astype(float)))
    print("\nMa trận mở rộng ban đầu [A|b]:")
    print(tabulate(Ab, floatfmt=".2f", tablefmt="grid"))
    
    # Khởi tạo bảng hội tụ
    headers = ["Bước"] + [f"x[{i+1}]" for i in range(n)] + ["Sai số ||x(k)-x(k-1)||", "Ước lượng sai số"]
    convergence_table = []
    
    # Khởi tạo nghiệm
    x_current = C.copy()
    convergence_table.append([0] + list(x_current) + ["-", "-"])
    
    # Quá trình khử Gauss
    for step in range(1, n+1):
        x_prev = x_current.copy()
        
        # Thực hiện phép khử
        col = step - 1
        max_row = np.argmax(np.abs(Ab[col:, col])) + col
        Ab[[col, max_row]] = Ab[[max_row, col]]
        Ab[col] = Ab[col] / Ab[col, col]
        for row in range(col + 1, n):
            Ab[row] = Ab[row] - Ab[row, col] * Ab[col]
        
        # Tính nghiệm tạm thời
        x_current = np.zeros(n)
        for i in range(col, -1, -1):
            x_current[i] = Ab[i, -1] - np.dot(Ab[i, i+1:n], x_current[i+1:])
        
        # Tính sai số
        if step > 1:
            error = np.linalg.norm(x_current - x_prev, np.inf)
            error_estimate = (q/(1-q)) * error if q is not None else "-"
            convergence_table.append([step] + list(x_current) + [f"{error:.4e}", f"{error_estimate:.4e}" if q is not None else "-"])
        else:
            convergence_table.append([step] + list(x_current) + ["-", "-"])
        
        print(f"\nSau bước khử cột {step}:")
        print(tabulate(Ab, floatfmt=".6f", tablefmt="grid"))
    
    # Thế ngược hoàn chỉnh
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])
    
    # Thêm kết quả cuối cùng
    final_error = np.linalg.norm(x - x_current, np.inf)
    final_estimate = (q/(1-q)) * final_error if q is not None else "-"
    convergence_table.append([n+1] + list(x) + [f"{final_error:.4e}", f"{final_estimate:.4e}" if q is not None else "-"])
    
    return x, convergence_table, headers, B_norm

def main():
    print("GIẢI HỆ PHƯƠNG TRÌNH TUYẾN TÍNH BẰNG PHƯƠNG PHÁP GAUSS")
    print("-----------------------------------------------------")
    
    A, b = nhap_he_phuong_trinh()
    
    try:
        det_A = np.linalg.det(A)
        print(f"\nĐịnh thức ma trận A: det(A) = {det_A:.6f}")
        if np.isclose(det_A, 0):
            raise ValueError("Ma trận suy biến (det(A) ≈ 0). Hệ có thể vô nghiệm hoặc vô số nghiệm.")
        
        x, convergence_table, headers, B_norm = gauss_elimination(A, b)
        
        print(f"\nChuẩn vô cùng của ma trận B: ||B|| = {B_norm:.6f}")
        if B_norm < 1:
            print("Điều kiện hội tụ: ||B|| < 1 thỏa mãn")
        else:
            print("Cảnh báo: ||B|| ≥ 1, ước lượng sai số có thể không chính xác")
        
        print("\nBảng quá trình hội tụ:")
        print(tabulate(convergence_table, headers=headers, floatfmt=".6f", tablefmt="grid"))
        
        print("\nNghiệm của hệ phương trình:")
        for i in range(len(x)):
            print(f"x[{i+1}] = {x[i]:.8f}")
        
        residual = np.dot(A, x) - b
        print("\nKiểm tra nghiệm (A*x - b):")
        print(residual)
        
    except ValueError as e:
        print(f"\nLỗi: {e}")

if __name__ == "__main__":
    main()