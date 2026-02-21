import numpy as np
# ГЕНЕРАЦІЯ ДАНИХ
def generate_system(n=100, x_true_val=2.5):
    A = np.random.rand(n, n)
    # Забезпечуємо діагональне переважання для збіжності
    for i in range(n):
        A[i, i] = sum(np.abs(A[i, :])) + 1

    #точний розв'язок
    x_true = np.full(n, x_true_val)
    #Обчислюємо b
    b = A @ x_true
    np.savetxt("matrix_A.txt", A)
    np.savetxt("vector_b.txt", b)
    print("Файли matrix_A.txt та vector_b.txt створено.")


#ФУНКЦІЇ РОЗВ'ЯЗКУ
def read_data():
    A = np.loadtxt("matrix_A.txt")
    b = np.loadtxt("vector_b.txt")
    return A, b


def simple_iteration(A, b, eps, max_iter=10000):
    n = len(b)
    tau = 0.9 / np.linalg.norm(A, ord=np.inf)  #tau
    x = np.ones(n)  # Початкове наближення

    for k in range(max_iter):
        x_new = x - tau * (A @ x - b)  # Формула методу
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            return x_new, k + 1
        x = x_new
    return x, max_iter


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

generate_system()
A, b = read_data()
eps0 = 1e-14  #точність

methods = [
    ("Проста ітерація", simple_iteration),
    ("Якобі", jacobi_method),
    ("Зейдель", seidel_method)
]

print(f"{'Метод':<20} | {'Ітерацій':<10} | {'Похибка (max)':<15}")
print("-" * 50)

for name, func in methods:
    sol, iters = func(A, b, eps0)
    error = np.max(np.abs(sol - 2.5))
    print(f"{name:<20} | {iters:<10} | {error:.2e}")