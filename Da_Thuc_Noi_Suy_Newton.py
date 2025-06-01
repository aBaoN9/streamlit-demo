def tinh_ti_sai_phan(x, y):
    n = len(y)
    coef = y.copy()
    
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x[i] - x[i-j])
    
    return coef

def da_thuc_newton(x, x_data, coef):
    n = len(coef)
    result = coef[-1]
    
    for i in range(n-2, -1, -1):
        result = result * (x - x_data[i]) + coef[i]
    
    return result

def main():
    print("ĐA THỨC NỘI SUY NEWTON")
    print("-----------------------")
    
    # Nhập số điểm mốc
    n = int(input("Nhập số điểm mốc nội suy: "))
    
    # Nhập các điểm mốc và giá trị hàm
    x_data = []
    y_data = []
    print("Nhập các điểm mốc và giá trị hàm:")
    for i in range(n):
        x = float(input(f"x[{i}] = "))
        y = float(input(f"y[{i}] = "))
        x_data.append(x)
        y_data.append(y)
    
    # Tính các hệ số (tỉ sai phân)
    coef = tinh_ti_sai_phan(x_data, y_data)
    
    # In bảng tỉ sai phân
    print("\nBảng tỉ sai phân:")
    print("i\tx[i]\t\tf[.]\t\tf[.,.]\t\t...")
    for i in range(n):
        print(f"{i}\t{x_data[i]:.4f}", end="\t")
        for j in range(i+1):
            if j < len(coef):
                print(f"{coef[j]:.4f}", end="\t")
        print()
    
    # Nhập điểm cần nội suy
    x_val = float(input("\nNhập giá trị x cần nội suy: "))
    
    # Tính giá trị nội suy
    y_val = da_thuc_newton(x_val, x_data, coef)
    print(f"Giá trị nội suy tại x = {x_val:.4f} là: {y_val:.4f}")
    
    # In đa thức Newton
    print("\nĐa thức Newton:")
    poly_str = f"P(x) = {coef[0]:.4f}"
    term = ""
    for i in range(1, n):
        term += f"(x - {x_data[i-1]:.4f})"
        poly_str += f" + {coef[i]:.4f}" + term
    print(poly_str)

if __name__ == "__main__":
    main()