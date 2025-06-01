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
    
    epsilon = float(input("\nNhập sai số epsilon (ví dụ 1e-3, 1e-4,...): "))
    
    return A, b, epsilon

def iterative_method(A, b, epsilon):
    n = len(b)
    max_iter = 10  # Mặc định 10 lần lặp
    
    # Kiểm tra điều kiện hội tụ
    det_A = np.linalg.det(A)
    print(f"\n1. Kiểm tra điều kiện hội tụ:")
    print(f"- Định thức ma trận A: det(A) = {det_A:.6f}")
    if det_A == 0:
        raise ValueError("Ma trận A suy biến (det(A) = 0), hệ phương trình không có nghiệm duy nhất")
    
    # Biến đổi hệ về dạng x = Bx + C
    B = np.zeros_like(A, dtype=float)
    C = np.zeros_like(b, dtype=float)
    
    for i in range(n):
        C[i] = b[i] / A[i, i]
        for j in range(n):
            if j != i:
                B[i, j] = -A[i, j] / A[i, i]
    
    B_norm = np.linalg.norm(B, np.inf)
    q = B_norm if B_norm < 1 else None
    
    # Khởi tạo nghiệm
    x = C.copy()
    table_data = []
    headers = ["k"] + [f"x[{i+1}]" for i in range(n)] + ["Ước lượng sai số"]
    
    # Thêm kết quả khởi tạo (k=0)
    table_data.append([0] + list(x) + ["-"])
    
    for iteration in range(1, max_iter + 1):
        x_new = B @ x + C
        error = np.linalg.norm(x_new - x, np.inf)
        
        # Tính ước lượng sai số nếu có q
        error_estimate = (q/(1-q))*error if q is not None else "-"
        
        # Thêm vào bảng kết quả
        table_data.append([iteration] + list(x_new) + [error_estimate if error_estimate != "-" else "-"])
        
        # Kiểm tra điều kiện dừng
        if error < epsilon:
            print(f"\n3. Kết quả: Hội tụ sau {iteration} lần lặp")
            break
            
        x = x_new
    else:
        print(f"\n3. Kết quả: Đạt số lần lặp tối đa {max_iter}")
    
    # Hiển thị bảng kết quả
    print("\nBảng giá trị nghiệm qua các bước lặp:")
    print(tabulate(table_data, headers=headers, floatfmt=".6f", tablefmt="grid"))
    
    return x_new, iteration, table_data

def main():
    print("GIẢI HỆ PHƯƠNG TRÌNH TUYẾN TÍNH BẰNG PHƯƠNG PHÁP LẶP ĐƠN")
    print("------------------------------------------------------")
    
    # Nhập hệ phương trình từ bàn phím
    A, b, epsilon = nhap_he_phuong_trinh()
    
    # Giải hệ bằng phương pháp lặp đơn
    try:
        x, iterations, table_data = iterative_method(A, b, epsilon)
        
        # In kết quả cuối cùng
        print("\n4. Nghiệm gần đúng cuối cùng:")
        for i in range(len(x)):
            print(f"x[{i+1}] = {x[i]:.8f}")
        
        # Vẽ đồ thị quá trình hội tụ
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        
        # Chuẩn bị dữ liệu để vẽ
        k_values = [row[0] for row in table_data]
        x_values = [row[1:1+len(x)] for row in table_data]
        
        for i in range(len(x)):
            plt.plot(k_values, [x[i] for x in x_values], '-o', label=f'x[{i+1}]')
        
        plt.xlabel('Số lần lặp (k)')
        plt.ylabel('Giá trị nghiệm')
        plt.title('QUÁ TRÌNH HỘI TỤ CỦA NGHIỆM')
        plt.xticks(k_values)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except ValueError as e:
        print(f"\nLỗi: {e}")

if __name__ == "__main__":
    # Cài đặt thư viện tabulate nếu chưa có
    try:
        from tabulate import tabulate
    except ImportError:
        print("\nYêu cầu cài đặt thư viện tabulate để hiển thị bảng đẹp")
        print("Chạy lệnh: pip install tabulate")
        exit()

    main()