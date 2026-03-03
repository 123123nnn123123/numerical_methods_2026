import csv
import numpy as np
import matplotlib.pyplot as plt


# 1. Зчитування даних (назви колонок з вашого фото: tasks, cost) [cite: 237, 171-181]
def read_data(filename):
    x, y = [], []
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                x.append(float(row['tasks']))
                y.append(float(row['cost']))
    except FileNotFoundError:
        print(f"Файл {filename} не знайдено.")
    return np.array(x), np.array(y)


# 2. Обчислення розділених різниць [cite: 7-12, 156]
def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef


# 3. Інтерполяційний многочлен Ньютона [cite: 54-55, 159]
def newton_poly(coef_matrix, x_nodes, x_val):
    n = len(x_nodes)
    res = coef_matrix[0, 0]
    product = 1.0
    for i in range(1, n):
        product *= (x_val - x_nodes[i - 1])
        res += coef_matrix[0, i] * product
    return res


# 4. Допоміжний поліном w_n(x) для оцінки похибки [cite: 34, 156]
def omega_n(x_nodes, x_val):
    res = 1.0
    for xi in x_nodes:
        res *= (x_val - xi)
    return res


# --- ВИКОНАННЯ ВАРІАНТУ 4 ---

x_nodes, y_nodes = read_data("data.csv")

if len(x_nodes) > 0:
    # Побудова таблиці розділених різниць для всіх 5 вузлів
    table_5 = divided_differences(x_nodes, y_nodes)

    # Побудова моделі для 3 вузлів (центральна частина даних) для порівняння [cite: 162, 245]
    x_nodes_3 = x_nodes[1:4]
    y_nodes_3 = y_nodes[1:4]
    table_3 = divided_differences(x_nodes_3, y_nodes_3)

    # Точка прогнозу (згідно з умовою варіанту 4 це 15000, але адаптуємо під ваші дані x=0.25) [cite: 237]
    target_x = 0.25
    res_5 = newton_poly(table_5, x_nodes, target_x)
    res_3 = newton_poly(table_3, x_nodes_3, target_x)

    # Вивід результатів у термінал
    print(f"--- РЕЗУЛЬТАТИ ВАРІАНТУ 4 ---")
    print(f"Прогноз вартості для {target_x} завдань (5 вузлів): {res_5:.6f}")
    print(f"Прогноз вартості для {target_x} завдань (3 вузли): {res_3:.6f}")
    print(f"Абсолютна різниця (похибка моделі): {abs(res_5 - res_3):.6f}")

    # ПІДГОТОВКА ГРАФІКІВ (4 ГРАФІКИ) [cite: 161, 238, 245-246]
    x_range = np.linspace(min(x_nodes), max(x_nodes), 500)
    y_full = [newton_poly(table_5, x_nodes, x) for x in x_range]
    y_reduced = [newton_poly(table_3, x_nodes_3, x) for x in x_range]
    y_omega = [omega_n(x_nodes, x) for x in x_range]
    y_err_dist = [abs(newton_poly(table_5, x_nodes, x) - newton_poly(table_3, x_nodes_3, x)) for x in x_range]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Графік 1: Основна модель Cost = f(tasks)
    axs[0, 0].plot(x_range, y_full, 'b', label='Модель (5 вузлів)')
    axs[0, 0].scatter(x_nodes, y_nodes, color='red', label='Експериментальні дані')
    axs[0, 0].set_title("1. Модель Cost = f(tasks)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Графік 2: Функція w_n(x) [cite: 62, 160]
    axs[0, 1].plot(x_range, y_omega, 'g', label='w_5(x)')
    axs[0, 1].axhline(0, color='black', lw=0.5)
    axs[0, 1].set_title("2. Допоміжний поліном w_n(x)")
    axs[0, 1].grid(True)

    # Графік 3: Порівняння стабільності (різна кількість вузлів) [cite: 162, 238]
    axs[1, 0].plot(x_range, y_full, 'b', label='5 вузлів (N4)')
    axs[1, 0].plot(x_range, y_reduced, 'r--', label='3 вузли (N2)')
    axs[1, 0].set_title("3. Вплив кількості вузлів на модель")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Графік 4: Графік похибок epsilon(x) [cite: 160, 243]
    axs[1, 1].fill_between(x_range, y_err_dist, color='orange', alpha=0.3)
    axs[1, 1].plot(x_range, y_err_dist, color='darkorange')
    axs[1, 1].set_title("4. Розподіл похибки ε(x)")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()