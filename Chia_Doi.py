import math

def chia_doi(f, a, b, E):
    if f(a) * f(b) > 0:
        print("Không có nghiệm trên đoạn này hoặc không thỏa mãn điều kiện f(a)*f(b) < 0")
        return None
    n = math.floor(math.log2((b - a) / E)) + 1
    for _ in range(n):
        c = (a + b) / 2
        if f(c) == 0:
            return c
        if f(c) * f(a) > 0:
            a = c
        else:
            b = c
    return (a + b) / 2

def main():
    # Ví dụ 1: Tìm nghiệm của phương trình x^2 - 2 = 0 trên đoạn [1, 2] với sai số 1e-6
    f1 = lambda x: x**3 + 3*(x**2) - 3
    a1, b1 = -3,-2
    E1 = 1e-3
    nghiem1 = chia_doi(f1, a1, b1, E1)
    print(f"Nghiệm của phương trình f(x) trên [{a1}, {b1}] là: {nghiem1:.6f}")
    
    
if __name__ == "__main__":
    main()