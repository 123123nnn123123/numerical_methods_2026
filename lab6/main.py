import numpy as np
def save_to_file(filename, data):
    np.savetxt(filename, data, fmt='%f')
def load_from_file(filename):
    return np.loadtxt(filename)
def get_lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        U[i, i] = 1

    for k in range(n):
        # Обчислення елементів L
        for i in range(k, n):
            L[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(k))

        # Обчислення елементів U
        for i in range(k + 1, n):
            U[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(k))) / L[k, k]

    return L, U


def solve_lu(L, U, B):
    n = len(L)
    # Прямий хід: LZ = B
    z = np.zeros(n)
    for k in range(n):
        z[k] = (B[k] - sum(L[k, j] * z[j] for j in range(k))) / L[k, k]

    # Зворотний хід: UX = Z
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = z[k] - sum(U[k, j] * x[j] for j in range(k + 1, n))

    return x


def get_vector_norm(vector):
    return np.max(np.abs(vector))


# 1. Генерація та запис даних (n=100)
n = 100
A_gen = np.random.uniform(1, 10, (n, n))
x_target = np.full(n, 2.5)
B_gen = A_gen @ x_target

save_to_file('matrix_A.txt', A_gen)
save_to_file('vector_B.txt', B_gen)

# 2. Зчитування та розв'язання
A = load_from_file('matrix_A.txt')
B = load_from_file('vector_B.txt')

L, U = get_lu_decomposition(A)
save_to_file('matrix_L.txt', L)
save_to_file('matrix_U.txt', U)

# Початковий розв'язок X0
x_0 = solve_lu(L, U, B)

# 3.ітераційне уточнення
eps_0 = 1e-14
x_current = x_0.copy()
iterations = 0

while True:
    iterations += 1
    R = B - (A @ x_current)

    if get_vector_norm(R) < eps_0 or iterations > 100:
        break
    delta_x = solve_lu(L, U, R)
    x_current = x_current + delta_x


save_to_file('vector_X.txt', x_current)
print(f"Початкова похибка: {get_vector_norm(B - A @ x_0)}")
print(f"Кількість ітерацій уточнення: {iterations}")
print(f"Кінцева похибка: {get_vector_norm(B - A @ x_current)}")
print(f"Перші 5 елементів знайденого вектора X: {x_current[:5]}")