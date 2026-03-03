import numpy as np
import matplotlib.pyplot as plt
import csv

# ----------------------------------
# ЗАДАНА ФУНКЦІЯ (можна змінити)
# ----------------------------------
def f(x):
    return 1 / (1 + 25 * x**2)   # зручно для демонстрації ефекту Рунге


# ----------------------------------
# Табуляція та запис у файл
# ----------------------------------
def tabulate_function(a, b, n, filename):
    x = np.linspace(a, b, n)
    y = f(x)

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])
        for xi, yi in zip(x, y):
            writer.writerow([xi, yi])

    return x, y


# ----------------------------------
# Розділені різниці
# ----------------------------------
def divided_differences(x, y):
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table


# ----------------------------------
# Поліном Ньютона
# ----------------------------------
def newton_polynomial(x, table, value):
    n = len(x)
    result = table[0, 0]
    product = 1.0

    for i in range(1, n):
        product *= (value - x[i - 1])
        result += table[0, i] * product

    return result


# ----------------------------------
# Дослідження для різної кількості вузлів
# ----------------------------------
def research(a, b, node_counts):
    x_dense = np.linspace(a, b, 1000)
    y_true = f(x_dense)

    plt.figure()
    plt.plot(x_dense, y_true)
    plt.title("Ефект Рунге та вплив кількості вузлів")
    plt.grid(True)

    for n in node_counts:
        x_nodes = np.linspace(a, b, n)
        y_nodes = f(x_nodes)

        table = divided_differences(x_nodes, y_nodes)

        y_interp = [newton_polynomial(x_nodes, table, xi) for xi in x_dense]

        error = np.max(np.abs(y_true - y_interp))
        print(f"Максимальна похибка при {n} вузлах: {error}")

        plt.plot(x_dense, y_interp, label=f"n = {n}")

    plt.legend()
    plt.show()


# ----------------------------------
# ОСНОВНА ПРОГРАМА
# ----------------------------------
if __name__ == "__main__":

    a = -1
    b = 1

    # Табуляція (5 вузлів як базовий приклад)
    x, y = tabulate_function(a, b, 5, "data.csv")

    # Побудова таблиці різниць
    table = divided_differences(x, y)

    print("Таблиця розділених різниць:")
    print(table)

    # Обчислення значення в конкретній точці
    x_value = 0.3
    y_value = newton_polynomial(x, table, x_value)

    print(f"\nЗначення полінома в точці {x_value} = {y_value}")
    print(f"Точне значення = {f(x_value)}")
    print(f"Похибка = {abs(f(x_value) - y_value)}")

    # Графік для базового випадку
    x_dense = np.linspace(a, b, 1000)
    y_true = f(x_dense)
    y_interp = [newton_polynomial(x, table, xi) for xi in x_dense]

    plt.figure()
    plt.plot(x_dense, y_true)
    plt.plot(x_dense, y_interp)
    plt.scatter(x, y)
    plt.title("Інтерполяція Ньютона")
    plt.grid(True)
    plt.show()

    # -------------------------------
    # ДОСЛІДНИЦЬКА ЧАСТИНА
    # -------------------------------
    print("\nДОСЛІДЖЕННЯ:")
    research(a, b, [5, 10, 20])