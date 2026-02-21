import numpy as np
import matplotlib.pyplot as plt

# 1. Задана функція навантаження на сервер
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

# Лічильник для адаптивного алгоритму
f_calls = 0
def f_count(x):
    global f_calls
    f_calls += 1
    return f(x)

# 2. Складова функція Сімпсона
def simpson_method(func, a, b, n):
    if n % 2 != 0: n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    # Формула Сімпсона
    s = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return s * h / 3

# Параметри
a, b = 0, 24
eps_target = 1e-12

# 3. Обчислення еталона
I0 = simpson_method(f, a, b, 2_000_000)

# 4. Пошук оптимального N (N_opt)
n_opt = 2
while n_opt < 10000:
    res = simpson_method(f, a, b, n_opt)
    if abs(res - I0) < eps_target:
        break
    n_opt += 2
eps_opt = abs(simpson_method(f, a, b, n_opt) - I0)

# 5. Обчислення для N0 (кратне 8)
n0 = int(n_opt / 10)
if n0 < 8: n0 = 8
while n0 % 8 != 0: n0 += 1
i_n0 = simpson_method(f, a, b, n0)
eps0 = abs(i_n0 - I0)

# 6. Метод Рунге-Ромберга
i_n0_half = simpson_method(f, a, b, n0 // 2)
i_runge = i_n0 + (i_n0 - i_n0_half) / 15
eps_runge = abs(i_runge - I0)

# 7. Метод Ейткена
q = 2
I_h = simpson_method(f, a, b, n0 * 4)
I_2h = simpson_method(f, a, b, n0 * 2)
I_4h = simpson_method(f, a, b, n0)

denom_aitken = (2 * I_2h - (I_4h + I_h))
if abs(denom_aitken) > 1e-18:
    i_aitken = (I_2h**2 - I_4h * I_h) / denom_aitken
else:
    i_aitken = I_h

p_aitken = (1 / np.log(q)) * np.log(abs((I_4h - I_2h) / (I_2h - I_h)))
eps_aitken = abs(i_aitken - I0)

# 8. Адаптивний алгоритм
def adaptive_simpson(func, a, b, eps, whole):
    mid = (a + b) / 2
    left = simpson_method(func, a, mid, 2)
    right = simpson_method(func, mid, b, 2)
    if abs(whole - (left + right)) <= 15 * eps:
        return left + right + (left + right - whole) / 15
    return adaptive_simpson(func, a, mid, eps/2, left) + \
           adaptive_simpson(func, mid, b, eps/2, right)

f_calls = 0
i_adaptive = adaptive_simpson(f_count, a, b, 1e-7, simpson_method(f_count, a, b, 2))


print("="*65)
print(f"Еталонне значення I0:          {I0:.14f}")
print("-"*65)
print(f"Складова формула Сімпсона:")
print(f"   - Оптимальне N_opt:            {n_opt}")
print(f"   - Похибка при N_opt:           {eps_opt:.2e}")
print(f"   - Значення при N0={n0}:         {i_n0:.14f}")
print(f"   - Похибка при N0:              {eps0:.2e}")
print("-"*65)
print(f"Метод Рунге-Ромберга:")
print(f"   - Уточнене значення IR:        {i_runge:.14f}")
print(f"   - Похибка epsR:                {eps_runge:.2e}")
print("-"*65)
print(f"Метод Ейткена:")
print(f"   - Порядок точності p:          {p_aitken:.4f}")
print(f"   - Похибка за Ейткеном:         {eps_aitken:.2e}")
print("-"*65)
print(f"Адаптивний алгоритм (eps=1e-7):")
print(f"   - Обчислене значення:          {i_adaptive:.14f}")
print(f"   - Кількість викликів f(x):     {f_calls}")
print("="*65)
plt.figure(figsize=(12, 5))

# Графік 1: Підінтегральна функція
plt.subplot(1, 2, 1)
x_vals = np.linspace(a, b, 500)
plt.plot(x_vals, f(x_vals), 'b-', label='f(x)')
plt.fill_between(x_vals, f(x_vals), color='blue', alpha=0.1)
plt.title('Графік навантаження на сервер')
plt.xlabel('Час (год)')
plt.ylabel('Навантаження')
plt.grid(True, alpha=0.3)
plt.legend()

# Графік 2: Аналіз похибки (без лінії цілі)
plt.subplot(1, 2, 2)
ns = np.arange(10, 501, 10)
errs = [abs(simpson_method(f, a, b, n) - I0) for n in ns]
plt.semilogy(ns, errs, 'r-o', markersize=4, label='Похибка Сімпсона')
plt.title('Залежність похибки від N')
plt.xlabel('N (кількість вузлів)')
plt.ylabel('Похибка')
plt.grid(True, which="both", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()