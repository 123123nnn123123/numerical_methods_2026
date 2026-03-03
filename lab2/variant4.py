import csv
import matplotlib.pyplot as plt
import numpy as np


N_NODES = 20
X_TARGET = 15000

def read_data(filename, limit):
    tasks = []
    cost = []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i >= limit: break # Беремо лише ту кількість вузлів, яку вказали
            tasks.append(float(row['tasks']))
            cost.append(float(row['cost']))
    return tasks, cost

def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef[0, :]

def newton_interpolation(x_data, y_data, x_target):
    coef = divided_differences(x_data, y_data)
    n = len(x_data)
    result = coef[0]
    product = 1.0
    for i in range(1, n):
        product *= (x_target - x_data[i - 1])
        result += coef[i] * product
    return result

# Виконання програми
x_nodes, y_nodes = read_data("data.csv", N_NODES)

if len(x_nodes) > 0:
    prediction = newton_interpolation(x_nodes, y_nodes, X_TARGET)
    print(f"Використано вузлів (n): {len(x_nodes)}")
    print(f"Прогноз для {X_TARGET} завдань: ${prediction:.4f}")

    # Графік
    x_range = np.linspace(min(x_nodes), max(x_nodes), 100)
    y_interp = [newton_interpolation(x_nodes, y_nodes, xi) for xi in x_range]

    plt.figure(figsize=(10, 5))
    plt.plot(x_range, y_interp, label=f'Інтерполяція (n={N_NODES})', color='blue')
    plt.scatter(x_nodes, y_nodes, color='red', label='Вузли з CSV')
    plt.axvline(X_TARGET, color='green', linestyle='--', label=f'Ціль ({X_TARGET})')
    plt.title(f'Дослідження впливу кількості вузлів n={N_NODES}')
    plt.legend()
    plt.grid(True)
    plt.show()