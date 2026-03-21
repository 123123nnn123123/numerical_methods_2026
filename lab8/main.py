import math
import numpy as np
import matplotlib.pyplot as plt


# --- 1. ТРАНСЦЕНДЕНТНА ФУНКЦІЯ ---
def F(x):
    return math.cos(x) - x


def dF(x):
    return -math.sin(x) - 1


def d2F(x):
    return -math.cos(x)


# --- 2. МЕТОДИ РОЗВ'ЯЗКУ ---

def simple_iteration(x0, eps, max_iter=1000):
    tau = -0.5
    x = x0
    for i in range(max_iter):
        x_next = x + tau * F(x)
        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, i + 1
        x = x_next
    return x, max_iter


def newton_method(x0, eps):
    x = x0
    for i in range(1000):
        x_next = x - F(x) / dF(x)
        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, i + 1
        x = x_next
    return x, 1000


def chebyshev_method(x0, eps):
    x = x0
    for i in range(1000):
        fx, dfx, d2fx = F(x), dF(x), d2F(x)
        x_next = x - fx / dfx - 0.5 * (fx ** 2 * d2fx) / (dfx ** 3)
        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, i + 1
        x = x_next
    return x, 1000


def chord_method(x0, x1, eps):
    for i in range(1000):
        fx0, fx1 = F(x0), F(x1)
        x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(F(x_next)) < eps and abs(x_next - x1) < eps:
            return x_next, i + 1
        x0, x1 = x1, x_next
    return x1, 1000


def parabola_method(x0, x1, x2, eps):
    for i in range(1000):
        # ВИПРАВЛЕНО: коректне присвоєння значень функції
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
    for i in range(1000):
        y0, y1, y2 = F(x0), F(x1), F(x2)
        if abs((y0 - y1) * (y0 - y2) * (y1 - y2)) < 1e-15: break
        x_next = (y1 * y2) / ((y0 - y1) * (y0 - y2)) * x0 + \
                 (y0 * y2) / ((y1 - y0) * (y1 - y2)) * x1 + \
                 (y0 * y1) / ((y2 - y0) * (y2 - y1)) * x2
        if abs(F(x_next)) < eps and abs(x_next - x2) < eps:
            return x_next, i + 1
        x0, x1, x2 = x1, x2, x_next
    return x2, 1000


# --- 3. АЛГЕБРАЇЧНІ РІВНЯННЯ ---

def horner_eval(coeffs, x):
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


def solve_newton_horner(coeffs, x0, eps):
    x = x0
    for i in range(1000):
        val, der = horner_eval(coeffs, x)
        x_next = x - val / der
        if abs(x_next - x) < eps:
            return x_next, i + 1
        x = x_next
    return x, 1000


def lin_method(a, alpha0, beta0, eps):
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
        new_p = (a[1] - new_q * b[3]) / b[2]
        new_alpha = -new_p / 2
        new_beta = math.sqrt(abs(new_q - new_alpha ** 2))
        if abs(new_alpha - alpha) < eps and abs(new_beta - beta) < eps:
            return (new_alpha, new_beta), i + 1
        alpha, beta = new_alpha, new_beta
    return (alpha, beta), 1000


# --- ВИКОНАННЯ ---

if __name__ == "__main__":
    EPS = 1e-10
    plt.ion()

    fig1 = plt.figure(1, figsize=(8, 4))
    x_vals = np.arange(-2, 2.1, 0.1)
    y_vals = [F(x) for x in x_vals]
    plt.plot(x_vals, y_vals, label="F(x) = cos(x) - x")
    plt.axhline(0, color='black', lw=1)
    plt.title("Табуляція трансцендентної функції")
    plt.grid(True)
    plt.legend()

    fig2 = plt.figure(2, figsize=(8, 4))
    alg_coeffs = [-1.0, 1.0, -1.0, 1.0]
    ax = np.linspace(-2, 2, 100)
    ay = [sum(c * (xi ** i) for i, c in enumerate(alg_coeffs)) for xi in ax]
    plt.plot(ax, ay, label="P(x) = x^3 - x^2 + x - 1", color='red')
    plt.axhline(0, color='black')
    plt.title("Графік алгебраїчного многочлена")
    plt.grid(True)
    plt.legend()

    plt.show(block=False)

    print("--- 1. РЕЗУЛЬТАТИ ТАБУЛЯЦІЇ ---")
    print(f"{'x':<10} | {'F(x)':<10}")
    print("-" * 25)
    with open("tabulation.txt", "w") as f:
        f.write("x\tF(x)\n")
        for x, y in zip(x_vals, y_vals):
            print(f"{x:<10.2f} | {y:<10.6f}")
            f.write(f"{x:.2f}\t{y:.6f}\n")

    x_start = 0.73
    print(f"\n--- 2. УТОЧНЕННЯ КОРЕНЯ (початкове наближення {x_start}) ---")

    methods = [
        ("Проста ітерація", lambda: simple_iteration(x_start, EPS)),
        ("Метод Ньютона", lambda: newton_method(x_start, EPS)),
        ("Метод Чебишева", lambda: chebyshev_method(x_start, EPS)),
        ("Метод хорд", lambda: chord_method(0.5, 0.9, EPS)),
        ("Метод парабол", lambda: parabola_method(0.5, 0.7, 0.9, EPS)),
        ("Зворотна інтерп.", lambda: inverse_interpolation(0.5, 0.7, 0.9, EPS))
    ]

    for name, func in methods:
        res, it = func()
        print(f"{name:20}: x = {res:.10f}, ітерацій: {it}")

    print("\n--- 3. АЛГЕБРАЇЧНЕ РІВНЯННЯ ---")
    root, it_h = solve_newton_horner(alg_coeffs, 1.5, EPS)
    print(f"Дійсний корінь (Горнер):  {root:.10f}, ітерацій: {it_h}")

    (re, im), it_l = lin_method(alg_coeffs, 0.1, 0.9, EPS)
    print(f"Комплексні корені (Лін):  {re:.5f} ± {im:.5f}i, ітерацій: {it_l}")
    plt.ioff()
    plt.show()