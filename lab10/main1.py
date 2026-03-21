import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# 1. ВИХІДНІ ДАНІ ТА ФУНКЦІЇ
# =====================================================================

def f(x, y):
    """Права частина диференціального рівняння dy/dx = f(x, y)"""
    return y - x ** 2 + 1


def y_exact(x):
    """Точний аналітичний розв'язок рівняння"""
    return x ** 2 + 2 * x + 1 - 0.5 * np.exp(x)


# =====================================================================
# 2. МЕТОД РУНГЕ-КУТТА 4-ГО ПОРЯДКУ (Сталий крок)
# =====================================================================

def rk4_step(f, x, y, h):
    """Один крок за методом Рунге-Кутта 4-го порядку"""
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def runge_kutta_4th_fixed(f, x0, y0, xN, h):
    """Чисельний розв'язок методом РК4 із фіксованим кроком"""
    steps = int(np.ceil((xN - x0) / h))
    x = [x0]
    y = [y0]

    print("-" * 85)
    print(
        f"{'Крок (n)':<9}|{'x_n':<8}|{'y_exact':<12}|{'y_num (RK4)':<14}|{'Похибка реальна':<17}|{'Похибка Рунге':<14}")
    print("-" * 85)
    print(f"{0:<9}|{x[0]:<8.3f}|{y_exact(x[0]):<12.6f}|{y[0]:<14.6f}|{0.0:<17.6e}|{0.0:<14.6e}")

    for n in range(steps):
        xn = x[n]
        yn = y[n]

        yn1 = rk4_step(f, xn, yn, h)

        yn_half_1 = rk4_step(f, xn, yn, h / 2)
        yn_half_2 = rk4_step(f, xn + h / 2, yn_half_1, h / 2)

        x.append(xn + h)
        y.append(yn1)

        err_runge = (16.0 / 15.0) * abs(yn1 - yn_half_2)
        err_real = abs(yn1 - y_exact(xn + h))

        print(f"{n + 1:<9}|{xn + h:<8.3f}|{y_exact(xn + h):<12.6f}|{yn1:<14.6f}|{err_real:<17.6e}|{err_runge:<14.6e}")

    print("-" * 85)
    return np.array(x), np.array(y)


# =====================================================================
# 3. АВТОМАТИЧНИЙ ВИБІР КРОКУ ЗА МЕТОДОМ РУНГЕ
# =====================================================================

def runge_kutta_4th_auto(f, x0, y0, xN, h_start, eps=1e-5):
    """Чисельний розв'язок РК4 з автоматичним вибором кроку"""
    x = [x0]
    y = [y0]
    h_values = []
    x_h_points = []

    h = h_start
    current_x = x0
    current_y = y0

    print("\n--- Процес розрахунку з автоматичним кроком ---")

    while current_x < xN:
        if current_x + h > xN:
            h = xN - current_x

        y_step_h = rk4_step(f, current_x, current_y, h)

        y_half_1 = rk4_step(f, current_x, current_y, h / 2)
        y_step_h2 = rk4_step(f, current_x + h / 2, y_half_1, h / 2)

        R_runge = (16.0 / 15.0) * abs(y_step_h - y_step_h2)

        if R_runge > eps:
            h /= 2
            print(f"Похибка {R_runge:.2e} > eps у точці x = {current_x:.3f}. Крок зменшено до h = {h:.5f}")
        else:
            current_x += h
            current_y = y_step_h
            x.append(current_x)
            y.append(current_y)
            h_values.append(h)
            x_h_points.append(current_x)

            if R_runge < eps / 32:
                h *= 2
                print(f"Похибка {R_runge:.2e} дуже мала у точці x = {current_x:.3f}. Крок збільшено до h = {h:.5f}")

    return np.array(x_h_points), np.array(h_values)


# =====================================================================
# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
# =====================================================================
if __name__ == "__main__":
    x0, y0 = 0.0, 0.5
    xN = 2.0
    h_fixed = 0.1
    eps_auto = 1e-6

    print("МЕТОД РУНГЕ-КУТТА 4-ГО ПОРЯДКУ З ФІКСОВАНИМ КРОКОМ\n")
    x_fix, y_fix = runge_kutta_4th_fixed(f, x0, y0, xN, h_fixed)

    local_error_real = np.abs(y_fix - y_exact(x_fix))

    local_error_runge = []
    for i in range(len(x_fix)):
        y_h = rk4_step(f, x_fix[i], y_fix[i], h_fixed)
        y_h2_1 = rk4_step(f, x_fix[i], y_fix[i], h_fixed / 2)
        y_h2_2 = rk4_step(f, x_fix[i] + h_fixed / 2, y_h2_1, h_fixed / 2)
        local_error_runge.append((16.0 / 15.0) * abs(y_h - y_h2_2))
    local_error_runge = np.array(local_error_runge)

    x_auto, h_auto = runge_kutta_4th_auto(f, x0, y0, xN, h_start=h_fixed, eps=eps_auto)

    # =====================================================================
    # ПОБУДОВА ГРАФІКІВ
    # =====================================================================
    plt.figure(figsize=(10, 11))

    # Графік 1
    plt.subplot(3, 1, 1)
    plt.plot(x_fix, local_error_real, 'g-s', label=r'Реальна похибка $|y_n - y(x_n)|$')
    plt.title('Графік локальної похибки (за точним розв\'язком)')
    plt.xlabel('x')
    plt.ylabel('Похибка')
    plt.legend()
    plt.grid(True)

    # Графік 2
    plt.subplot(3, 1, 2)
    plt.plot(x_fix, local_error_runge, 'c--o', label=r'Похибка за Рунге $\frac{16}{15}|y_n(h) - y_n(h/2)|$')
    plt.title('Графік локальної похибки (за методом Рунге)')
    plt.xlabel('x')
    plt.ylabel('Чисельна похибка')
    plt.legend()
    plt.grid(True)

    # Графік 3
    plt.subplot(3, 1, 3)
    plt.step(x_auto, h_auto, where='post', color='darkorange', linewidth=2, label='Величина кроку $h(x)$')
    plt.plot(x_auto, h_auto, 'darkorange', marker='o', alpha=0.5)
    plt.title(fr'Графік залежності автоматичного кроку від координати $x$ ($\epsilon = {eps_auto}$)')
    plt.xlabel('x')
    plt.ylabel('Крок h')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()