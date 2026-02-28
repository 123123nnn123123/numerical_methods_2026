import numpy as np

def generate_system(n=100, x_true_val=2.5):
    A = np.random.rand(n, n)
    for i in range(n):
        A[i, i] = sum(np.abs(A[i, :])) + 1

    x_true = np.full(n, x_true_val)
    # Обчислюємо вектор
    b = A @ x_true

    np.savetxt("matrix_A.txt", A)
    np.savetxt("vector_b.txt", b)
def read_data():
    A = np.loadtxt("matrix_A.txt")
    b = np.loadtxt("vector_b.txt")
    return A, b


# Метод простої ітерації
def simple_iteration(A, b, eps, max_iter=10000):
    n = len(b)
    # Вибір параметра tau
    tau = 0.9 / np.linalg.norm(A, ord=np.inf)
    x = np.ones(n)  # Початкове наближення

    for k in range(max_iter):
        x_new = x - tau * (A @ x - b)
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            return x_new, k + 1
        x = x_new
    return x, max_iter


# Метод Якобі
def jacobi_method(A, b, eps, max_iter=10000):
    n = len(b)
    x = np.ones(n)
    D = np.diag(A)
    R = A - np.diagflat(D)

    for k in range(max_iter):
        x_new = (b - R @ x) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            return x_new, k + 1
        x = x_new
    return x, max_iter


# Метод Зейделя
def seidel_method(A, b, eps, max_iter=10000):
    n = len(b)
    x = np.ones(n)

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sum_j = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sum_j) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < eps:
            return x, k + 1
    return x, max_iter

if __name__ == "__main__":
    generate_system()
    A, b = read_data()
    # Задана точність
    eps0 = 1e-14
    methods = [
        ("Проста ітерація", simple_iteration, "vector_X_simple.txt"),
        ("Якобі", jacobi_method, "vector_X_jacobi.txt"),
        ("Зейдель", seidel_method, "vector_X_seidel.txt")
    ]

    print(f"\n{'Метод':<20} | {'Ітерацій':<10} | {'Похибка (max)':<15}")
    print("-" * 55)

    for name, func, filename in methods:
        # Знаходження розв'язку
        sol, iters = func(A, b, eps0)
        # Обчислення похибки
        error = np.max(np.abs(sol - 2.5))
        np.savetxt(filename, sol)

        print(f"{name:<20} | {iters:<10} | {error:.2e}")
    print("-" * 55)