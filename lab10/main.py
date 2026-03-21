import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# 1. ВИХІДНІ ДАНІ ТА ФУНКЦІЇ (Задаються індивідуально за варіантом)
# =====================================================================

def f(x, y):
    """Права частина диференціального рівняння dy/dx = f(x, y)"""
    return y - x ** 2 + 1


def y_exact(x):
    """Точний аналітичний розв'язок рівняння (Пункт 1)"""
    return x ** 2 + 2 * x + 1 - 0.5 * np.exp(x)


def d3y_dx3(x, y):
    """
    Третя похідна y'''(x), яка необхідна для теоретичної оцінки похибки R2.
    Виводиться аналітично з вихідного рівняння:
    y' = y - x^2 + 1
    y'' = y' - 2x = y - x^2 - 2x + 1
    y''' = y'' - 2 = y - x^2 - 2x - 1
    """
    return y - x ** 2 - 2 * x - 1


# =====================================================================
# 2. МЕТОД ПРОГНОЗУ ТА КОРЕКЦІЇ АДАМСА 2-ГО ПОРЯДКУ (Сталий крок)
# =====================================================================

def adams_2nd_order_fixed(f, x0, y0, xN, h, eps=1e-5):
    """Розрахунок методом Адамса 2-го порядку з фіксованим кроком (Пункт 2)"""
    steps = int(np.ceil((xN - x0) / h))
    x = [x0]
    y = [y0]

    # Метод другого порядку двокроковий, тому перший крок (y1)
    # знаходимо за допомогою модифікованого методу Ейлера
    x1 = x0 + h
    y1_pred = y0 + h * f(x0, y0)
    y1 = y0 + (h / 2) * (f(x0, y0) + f(x1, y1_pred))
    x.append(x1)
    y.append(y1)

    iterations_per_step = [0, 0]

    print("-" * 82)
    print(f"{'Крок (n)':<9}|{'x_n':<8}|{'y_exact':<12}|{'y_num (Adams)':<15}|{'Ітерації':<10}|{'Похибка':<12}")
    print("-" * 82)
    print(f"{0:<9}|{x[0]:<8.3f}|{y_exact(x[0]):<12.6f}|{y[0]:<15.6f}|{'—':<10}|{0.0:<12.6e}")
    print(f"{1:<9}|{x[1]:<8.3f}|{y_exact(x[1]):<12.6f}|{y[1]:<15.6f}|{'—':<10}|{abs(y[1] - y_exact(x[1])):<12.6e}")

    for n in range(1, steps):
        xn = x[n]
        xn1 = xn + h
        yn = y[n]
        yn_minus_1 = y[n - 1]

        # Етап Прогнозу (Явна формула Адамса)
        y_pred = yn + (h / 2) * (3 * f(xn, yn) - f(x[n - 1], yn_minus_1))

        # Етап Корекції (Ітераційний процес)
        y_corr_prev = y_pred
        iter_count = 0
        while True:
            iter_count += 1
            # Неявна формула корекції
            y_corr_curr = yn + (h / 2) * (f(xn1, y_corr_prev) + f(xn, yn))

            # Перевірка умови збіжності
            if abs(y_corr_curr - y_corr_prev) <= eps or iter_count > 20:
                break
            y_corr_prev = y_corr_curr

        x.append(xn1)
        y.append(y_corr_curr)
        iterations_per_step.append(iter_count)

        error = abs(y_corr_curr - y_exact(xn1))
        print(f"{n + 1:<9}|{xn1:<8.3f}|{y_exact(xn1):<12.6f}|{y_corr_curr:<15.6f}|{iter_count:<10}|{error:<12.6e}")

    print("-" * 82)
    return np.array(x), np.array(y), iterations_per_step


# =====================================================================
# 3. АВТОМАТИЧНИЙ ВИБІР КРОКУ (Пункт 5)
# =====================================================================

def adams_2nd_order_auto(f, x0, y0, xN, h_start, eps=1e-4):
    """Розрахунок з автоматичною зміною кроку сітки (Пункт 5)"""
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

        # Крок h
        x_half = current_x + h / 2
        y_half_pred = current_y + (h / 2) * f(current_x, current_y)
        y_half = current_y + (h / 4) * (f(current_x, current_y) + f(x_half, y_half_pred))

        y_step_h_pred = current_y + (h / 2) * (3 * f(x_half, y_half) - f(current_x, current_y))
        y_step_h = current_y + (h / 2) * (f(current_x + h, y_step_h_pred) + f(x_half, y_half))

        # Крок h/2 (два рази)
        h2 = h / 2
        x_quarter = current_x + h2 / 2
        y_quarter_pred = current_y + (h2 / 2) * f(current_x, current_y)
        y_quarter = current_y + (h2 / 4) * (f(current_x, current_y) + f(x_quarter, y_quarter_pred))

        y_step_h2_1_pred = current_y + (h2 / 2) * (3 * f(x_quarter, y_quarter) - f(current_x, current_y))
        y_step_h2_1 = current_y + (h2 / 2) * (f(current_x + h2, y_step_h2_1_pred) + f(x_quarter, y_quarter))

        x_three_quarter = current_x + h2 + h2 / 2
        y_three_quarter_pred = y_step_h2_1 + (h2 / 2) * f(current_x + h2, y_step_h2_1)
        y_three_quarter = y_step_h2_1 + (h2 / 4) * (
                    f(current_x + h2, y_step_h2_1) + f(x_three_quarter, y_three_quarter_pred))

        y_step_h2_2_pred = y_step_h2_1 + (h2 / 2) * (
                    3 * f(x_three_quarter, y_three_quarter) - f(current_x + h2, y_step_h2_1))
        y_step_h2_2 = y_step_h2_1 + (h2 / 2) * (
                    f(current_x + h, y_step_h2_2_pred) + f(x_three_quarter, y_three_quarter))

        # Оцінка похибки за правилом Рунге
        R_runge = abs(y_step_h - y_step_h2_2) / 3

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

            if R_runge < eps / 8:
                h *= 2
                print(f"Похибка {R_runge:.2e} досить мала у точці x = {current_x:.3f}. Крок збільшено до h = {h:.5f}")

    return np.array(x_h_points), np.array(h_values)


# =====================================================================
# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
# =====================================================================
if __name__ == "__main__":
    # Параметри інтегрування (Пункт 2)
    x0, y0 = 0.0, 0.5
    xN = 2.0
    h_fixed = 0.1
    eps_auto = 1e-4

    print("ЧАСТИНА 1. МЕТОД ПРОГНОЗУ ТА КОРЕКЦІЇ АДАМСА З ФІКСОВАНИМ КРОКОМ\n")
    x_fix, y_fix, iters = adams_2nd_order_fixed(f, x0, y0, xN, h_fixed)

    # Обчислення похибок для графіків
    y_exact_vals = y_exact(x_fix)
    local_error_real = np.abs(y_fix - y_exact_vals)  # Похибка для Пункту 3 [cite: 274]
    local_error_theor = (h_fixed ** 3 / 12) * np.abs(d3y_dx3(x_fix, y_fix))  # Похибка для Пункту 4 [cite: 275]

    # Розрахунок автоматичного кроку для Пункту 5 [cite: 276]
    x_auto, h_auto = adams_2nd_order_auto(f, x0, y0, xN, h_start=h_fixed, eps=eps_auto)

    # =====================================================================
    # ПОБУДОВА ГРАФІКІВ (Чітко за Пунктами 3, 4, 5 ходу роботи)
    # =====================================================================
    plt.figure(figsize=(10, 11))

    # 1. Графік для ПУНКТУ 3
    plt.subplot(3, 1, 1)
    plt.plot(x_fix, local_error_real, 'b-o', label=r'Реальна похибка $|y_n - y(x_n)|$')
    plt.title('Пункт 3: Графік локальної похибки (використовуючи точний розв\'язок)')
    plt.xlabel('x')
    plt.ylabel('Похибка')
    plt.legend()
    plt.grid(True)

    # 2. Графік для ПУНКТУ 4
    plt.subplot(3, 1, 2)
    plt.plot(x_fix, local_error_theor, 'm--s', label=r'Оцінка похибки $R_2^{kop} = \frac{h^3}{12}|y\'\'\'|$')
    plt.title('Пункт 4: Графік локальної похибки (використовуючи вираз для оцінки похибки)')
    plt.xlabel('x')
    plt.ylabel('Теоретична похибка')
    plt.legend()
    plt.grid(True)

    # 3. Графік для ПУНКТУ 5
    plt.subplot(3, 1, 3)
    plt.step(x_auto, h_auto, where='post', color='purple', linewidth=2, label='Величина кроку $h(x)$')
    plt.plot(x_auto, h_auto, 'purple', marker='o', alpha=0.5)
    plt.title(fr'Пункт 5: Графік залежності величини автоматичного кроку від координати $x$ ($\epsilon = {eps_auto}$)')
    plt.xlabel('x')
    plt.ylabel('Крок h')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()