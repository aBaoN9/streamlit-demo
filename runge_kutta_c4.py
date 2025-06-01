def runge_kutta_4(f, x0, y0, h, n):
    """
    Giải phương trình vi phân y' = f(x, y) bằng phương pháp Runge-Kutta cấp 4
    
    Parameters:
    f: Hàm f(x, y) trong phương trình y' = f(x, y)
    x0: Giá trị x ban đầu
    y0: Giá trị y ban đầu (y(x0) = y0)
    h: Bước nhảy
    n: Số bước
    
    Returns:
    Hai list chứa các giá trị x và y tương ứng
    """
    x_values = [x0]
    y_values = [y0]
    
    for _ in range(n):
        x = x_values[-1]
        y = y_values[-1]
        
        # Tính các hệ số k
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        
        # Tính giá trị mới của y
        y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x_new = x + h
        
        x_values.append(x_new)
        y_values.append(y_new)
    
    return x_values, y_values


# Ví dụ sử dụng
if __name__ == "__main__":
    # Định nghĩa phương trình vi phân: y' = x + y
    def example_equation(x, y):
        return x + y
    
    # Điều kiện ban đầu
    x0 = 0
    y0 = 1
    h = 0.1
    n = 10
    
    # Giải bằng phương pháp Runge-Kutta cấp 4
    x_vals, y_vals = runge_kutta_4(example_equation, x0, y0, h, n)
    
    # In kết quả
    print("Kết quả phương pháp Runge-Kutta cấp 4:")
    for x, y in zip(x_vals, y_vals):
        print(f"x = {x:.2f}, y ≈ {y:.6f}")