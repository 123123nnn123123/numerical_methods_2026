import math
import numpy as np
def F(x):
    return math.cos(x) - x
def dF(x):
    return -math.sin(x) - 1
def d2F(x):
    return -math.cos(x)

# --- МЕТОДИ РОЗВ'ЯЗКУ НЕЛІНІЙНИХ РІВНЯНЬ ---

def simple_iteration(x0, eps, max_iter=1000):
    # Метод простої ітерації
    tau = 0.6
    x = x0
    for i in range(max_iter):
        try:
            x_next = x + tau * F(x)
            if not math.isfinite(x_next): break
            if abs(F(x_next)) < eps and abs(x_next - x) < eps:
                return x_next, i + 1
            x = x_next
        except OverflowError:
            break
    return None, 0


def newton_method(x0, eps):
    # Метод Ньютона
    x = x0
    for i in range(1000):
        x_next = x - F(x) / dF(x)
        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, i + 1
        x = x_next
    return x, 1000


def chebyshev_method(x0, eps):
    # Метод Чебишева
    x = x0
    for i in range(1000):
        fx, dfx, d2fx = F(x), dF(x), d2F(x)
        x_next = x - fx / dfx - 0.5 * (fx ** 2 * d2fx) / (dfx ** 3)
        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, i + 1
        x = x_next
    return x, 1000


def chord_method(x0, x1, eps):
    # Метод хорд
    for i in range(1000):
        fx0, fx1 = F(x0), F(x1)
        x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(F(x_next)) < eps and abs(x_next - x1) < eps:
            return x_next, i + 1
        x0, x1 = x1, x_next
    return x1, 1000


def parabola_method(x0, x1, x2, eps):
    # Метод парабол
    for i in range(1000):
        f0, f1, f2 = F(x0), F(x1), F(x2)
        f01 = (f1 - f0) / (x1 - x0)
        f12 = (f2 - f1) / (x2 - x1)
        f012 = (f12 - f01) / (x2 - x0)
        w = f12 + (x2 - x1) * f012
        det = w ** 2 - 4 * f2 * f012
        sqrt_det = math.sqrt(max(0, det))
        denom = w + sqrt_det if abs(w + sqrt_det) > abs(w - sqrt_det) else w - sqrt_det
        x_next = x2 - 2 * f2 / denom
        if abs(F(x_next)) < eps and abs(x_next - x2) < eps:
            return x_next, i + 1
        x0, x1, x2 = x1, x2, x_next
    return x2, 1000


def inverse_interpolation(x0, x1, x2, eps):
    # Метод зворотної інтерполяції
    for i in range(1000):
        y0, y1, y2 = F(x0), F(x1), F(x2)
        x_next = (y1 * y2) / ((y0 - y1) * (y0 - y2)) * x0 + \
                 (y0 * y2) / ((y1 - y0) * (y1 - y2)) * x1 + \
                 (y0 * y1) / ((y2 - y0) * (y2 - y1)) * x2
        if abs(F(x_next)) < eps and abs(x_next - x2) < eps:
            return x_next, i + 1
        x0, x1, x2 = x1, x2, x_next
    return x2, 1000


# --- АЛГЕБРАЇЧНІ РІВНЯННЯ ---

def horner_eval(coeffs, x):
    # Схема Горнера
    m = len(coeffs) - 1
    b = [0] * (m + 1)
    b[m] = coeffs[m]
    for i in range(m - 1, -1, -1):
        b[i] = coeffs[i] + x * b[i + 1]

    c = [0] * (m + 1)
    c[m] = b[m]
    for i in range(m - 1, 0, -1):
        c[i] = b[i] + x * c[i + 1]
    return b[0], c[1]


def solve_algebraic(coeffs, x0, eps):
    x = x0
    for i in range(1000):
        val, der = horner_eval(coeffs, x)
        x_next = x - val / der
        if abs(x_next - x) < eps:
            return x_next, i + 1
        x = x_next
    return x, 1000


def lin_method(a, alpha0, beta0, eps):
    # Метод Ліна для комплексних коренів
    alpha, beta = alpha0, beta0
    m = len(a) - 1
    for i in range(1000):
        p, q = -2 * alpha, alpha ** 2 + beta ** 2
        b = [0] * (m + 1)
        b[m] = a[m]
        b[m - 1] = a[m - 1] + p * b[m]
        for j in range(m - 2, 1, -1):
            b[j] = a[j] + p * b[j + 1] + q * b[j + 2]
        new_q = a[0] / b[2]
        new_p = (a[1] - new_q * b[3]) / b[2] if m > 2 else a[1] / b[2]
        new_alpha = -new_p / 2
        new_beta = math.sqrt(abs(new_q - new_alpha ** 2))
        if abs(new_alpha - alpha) < eps and abs(new_beta - beta) < eps:
            return (new_alpha, new_beta), i + 1
        alpha, beta = new_alpha, new_beta
    return (alpha, beta), 1000


# --- ВИКОНАННЯ ---

if __name__ == "__main__":
    EPS = 1e-10

    # 1. Створення файлу з коефіцієнтами
    with open("coeffs.txt", "w") as f:
        f.write("1 -2 1 -2")

    # 2. Табуляція
    print("--- Табуляція ---")
    for x in np.arange(0, 1.1, 0.1):
        print(f"x: {x:.1f} | F(x): {F(x):.6f}")

    # 3. Порівняння методів
    print("\n--- Порівняння методів ---")
    methods = [
        ("Проста ітерація", lambda: simple_iteration(0.5, EPS)),
        ("Метод Ньютона", lambda: newton_method(0.5, EPS)),
        ("Метод Чебишева", lambda: chebyshev_method(0.5, EPS)),
        ("Метод хорд", lambda: chord_method(0.0, 1.0, EPS)),
        ("Метод парабол", lambda: parabola_method(0.0, 0.5, 1.0, EPS)),
        ("Зворотна інт.", lambda: inverse_interpolation(0.0, 0.5, 1.0, EPS))
    ]

    for name, func in methods:
        res, it = func()
        if res is not None:
            print(f"{name:16}: x = {res:.10f}, ітерацій: {it}")
        else:
            print(f"{name:16}: Розбігається")

    # 4. Алгебраїчне рівняння
    with open("coeffs.txt", "r") as f:
        alg_coeffs = [float(x) for x in f.read().split()]

    print("\n--- Алгебраїчне рівняння ---")
    root, it_h = solve_algebraic(alg_coeffs, 1.5, EPS)
    print(f"Дійсний корінь: {root:.10f}, ітерацій: {it_h}")

    (re, im), it_l = lin_method(alg_coeffs, 0.5, 0.5, EPS)
    print(f"Комплексні корені: {re:.5f} ± {im:.5f}i, ітерацій: {it_l}")