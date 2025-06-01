def improved_euler_method(f, x0, y0, h, n):
    """
    Giải phương trình vi phân bằng phương pháp Euler cải tiến
    
    Parameters giống như phương pháp Euler cơ bản
    """
    x_values = [x0]
    y_values = [y0]
    
    for _ in range(n):
        x = x_values[-1]
        y = y_values[-1]
        
        # Bước dự đoán
        y_pred = y + h * f(x, y)
        
        # Bước hiệu chỉnh
        y_new = y + h * 0.5 * (f(x, y) + f(x + h, y_pred))
        x_new = x + h
        
        x_values.append(x_new)
        y_values.append(y_new)
    
    return x_values, y_values


# Ví dụ sử dụng
if __name__ == "__main__":
    # Định nghĩa phương trình vi phân: y' = x - y
    def example_equation(x, y):
        return x - y
    
    # Điều kiện ban đầu
    x0 = 0
    y0 = 1
    h = 0.1
    n = 10
    
    # Giải bằng phương pháp Euler cải tiến
    x_vals, y_vals = improved_euler_method(example_equation, x0, y0, h, n)
    
    # In kết quả
    print("\nKết quả phương pháp Euler cải tiến:")
    for x, y in zip(x_vals, y_vals):
        print(f"x = {x:.2f}, y ≈ {y:.6f}")