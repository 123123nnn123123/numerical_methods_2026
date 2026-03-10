import numpy as np
import matplotlib.pyplot as plt
import csv


# -----------------------------
# зчитування даних з CSV
# -----------------------------
def read_csv(filename):
    x = []
    y = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    return np.array(x), np.array(y)


# -----------------------------
# формування матриці
# -----------------------------
def form_matrix(x, m):
    A = np.zeros((m + 1, m + 1))

    for i in range(m + 1):
        for j in range(m + 1):
            A[i][j] = np.sum(x ** (i + j))

    return A


# -----------------------------
# формування вектора
# -----------------------------
def form_vector(x, y, m):
    b = np.zeros(m + 1)

    for i in range(m + 1):
        b[i] = np.sum(y * (x ** i))

    return b


# -----------------------------
# метод Гауса
# -----------------------------
def gauss(A, b):
    n = len(b)

    for k in range(n):

        max_row = np.argmax(abs(A[k:, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.sum(A[i, i + 1:] * x[i + 1:])) / A[i, i]

    return x


# -----------------------------
# поліном
# -----------------------------
def polynomial(x, coef):
    y = np.zeros_like(x, dtype=float)

    for i, c in enumerate(coef):
        y += c * (x ** i)

    return y


# -----------------------------
# дисперсія
# -----------------------------
def variance(y1, y2):
    return np.mean((y1 - y2) ** 2)


# -----------------------------
# основна програма
# -----------------------------
x, y = read_csv("data.csv")

max_degree = 4
variances = []

for m in range(1, max_degree + 1):
    A = form_matrix(x, m)
    b = form_vector(x, y, m)

    coef = gauss(A.copy(), b.copy())

    y_ap = polynomial(x, coef)

    var = variance(y, y_ap)

    variances.append(var)

optimal_m = np.argmin(variances) + 1

print("Оптимальний степінь:", optimal_m)

A = form_matrix(x, optimal_m)
b = form_vector(x, y, optimal_m)

coef = gauss(A, b)

y_ap = polynomial(x, coef)

# -----------------------------
# прогноз
# -----------------------------
x_future = np.array([25, 26, 27])
y_future = polynomial(x_future, coef)

print("Прогноз температур:", y_future)

# -----------------------------
# похибка
# -----------------------------
error = y - y_ap

# -----------------------------
# графік 1
# -----------------------------
plt.figure()

plt.scatter(x, y, label="Дані")
plt.plot(x, y_ap, label="Апроксимація")

plt.xlabel("Місяць")
plt.ylabel("Температура")
plt.title("Апроксимація температур")

plt.legend()
plt.grid()

plt.show()

# -----------------------------
# графік 2
# -----------------------------
plt.figure()

plt.plot(range(1, max_degree + 1), variances, marker='o')

plt.xlabel("Степінь полінома")
plt.ylabel("Дисперсія")
plt.title("Залежність дисперсії")

plt.grid()

plt.show()

# -----------------------------
# графік 3
# -----------------------------
plt.figure()

plt.plot(x, error, marker='o')

plt.xlabel("Місяць")
plt.ylabel("Похибка")
plt.title("Похибка апроксимації")

plt.grid()

plt.show()