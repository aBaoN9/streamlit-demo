def lagrange_interpolation(x_values, y_values):
    # Kiểm tra số lượng phần tử của x và y có bằng nhau không
    if len(x_values) != len(y_values):
        raise ValueError("Số lượng phần tử của x và y phải bằng nhau")
    
    n = len(x_values)
    
    # Hàm nội suy trả về giá trị tại vị trí x
    def interpolate(x):
        result = 0.0
        for i in range(n):
            term = y_values[i]
            for j in range(n):
                if j != i:
                    term *= (x - x_values[j]) / (x_values[i] - x_values[j])
            result += term
        return result
    
    return interpolate


# Dữ liệu mẫu
x_data = [1, 2, 3]
y_data = [1, 4, 9]

# Tạo hàm nội suy
poly = lagrange_interpolation(x_data, y_data)

# Tính giá trị tại x = 2.5
print(poly(2.5))  # Kết quả: 6.25