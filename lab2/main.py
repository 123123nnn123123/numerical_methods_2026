import csv
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    x, y = [], []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['tasks']))
            y.append(float(row['cost']))
    return np.array(x), np.array(y)

def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef

def newton_poly(coef_matrix, x_nodes, x_val):
    n = len(x_nodes)
    res = coef_matrix[0, 0]
    product = 1.0
    for i in range(1, n):
        product *= (x_val - x_nodes[i-1])
        res += coef_matrix[0, i] * product
    return res

# --- ПІДГОТОВКА ДАНИХ ---
x_nodes, y_nodes = read_data("data.csv")
table_full = divided_differences(x_nodes, y_nodes)

# Для дослідження (стор. 14): беремо обмежену кількість вузлів
x_small = x_nodes[1:-1] # Наприклад, 3 вузли для порівняння
table_small = divided_differences(x_small, y_nodes[1:-1])

x_plot = np.linspace(min(x_nodes), max(x_nodes), 500)
y_full = [newton_poly(table_full, x_nodes, x) for x in x_plot]
y_small = [newton_poly(table_small, x_small, x) for x in x_plot]
y_omega = [abs(np.prod([x - xi for xi in x_nodes])) for x in x_plot]
y_error = [abs(y_f - y_s) for y_f, y_s in zip(y_full, y_small)]

# --- ВІЗУАЛІЗАЦІЯ (4 ГРАФІКИ) ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. Основна інтерполяція [cite: 161, 208]
axs[0, 0].plot(x_plot, y_full, 'b', label='N_n(x) повна')
axs[0, 0].scatter(x_nodes, y_nodes, color='red')
axs[0, 0].set_title("1. Інтерполяція (всі вузли)")
axs[0, 0].grid(True)

# 2. Допоміжний поліном w_n(x) [cite: 160]
axs[0, 1].plot(x_plot, y_omega, 'g', label='|w_n(x)|')
axs[0, 1].set_title("2. Поведінка |w_n(x)|")
axs[0, 1].grid(True)

# 3. Дослідження кількості вузлів (Ефект Рунге) [cite: 245, 247]
axs[1, 0].plot(x_plot, y_full, 'b', label='5 вузлів')
axs[1, 0].plot(x_plot, y_small, 'r--', label='3 вузли')
axs[1, 0].set_title("3. Вплив кількості вузлів")
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4. Графік похибки epsilon(x) [cite: 160, 246]
axs[1, 1].fill_between(x_plot, y_error, color='orange', alpha=0.3)
axs[1, 1].plot(x_plot, y_error, color='orange')
axs[1, 1].set_title("4. Графік похибки ε(x)")
axs[1, 1].grid(True)



plt.tight_layout()
plt.show()