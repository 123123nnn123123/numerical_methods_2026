import numpy as np
import matplotlib.pyplot as plt

#  Визначення функції вологості M(t) та її аналітичної похідної
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def M_prime_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

# Формула центральної різниці для апроксимації похідної
def central_diff(f, t, h):
    return (f(t + h) - f(t - h)) / (2 * h)

# Вихідні дані
t0 = 1.0
h_fixed = 1e-3
exact_val = M_prime_exact(t0)

print(f"--- Аналітичний розрахунок ---")
print(f"Точне значення M'({t0}): {exact_val:.10f}\n")

# Дослідження залежності похибки від кроку h
h_values = np.logspace(-20, 3, num=500)
errors = []
best_h = h_values[0]
min_error = float('inf')

for h in h_values:
    approx = central_diff(M, t0, h)
    error = abs(approx - exact_val)
    errors.append(error)
    if error < min_error:
        min_error = error
        best_h = h

print(f"--- Пошук оптимального кроку ---")
print(f"Оптимальний крок h0: {best_h:.2e}")
print(f"Найкраща точність R0: {min_error:.2e}\n")

# Метод Рунге-Ромберга
d_h = central_diff(M, t0, h_fixed)
d_2h = central_diff(M, t0, 2 * h_fixed)

y_runge = d_h + (d_h - d_2h) / 3
error_runge = abs(y_runge - exact_val)

print(f"--- Метод Рунге-Ромберга (h={h_fixed}) ---")
print(f"Значення y'(h): {d_h:.10f}")
print(f"Значення y'(2h): {d_2h:.10f}")
print(f"Уточнене значення y_R: {y_runge:.10f}")
print(f"Похибка R2: {error_runge:.2e}\n")

#Метод Ейткена
d_4h = central_diff(M, t0, 4 * h_fixed)

# Уточнення за Ейткеном
denom_aitken = 2 * d_2h - (d_4h + d_h)
if denom_aitken != 0:
    y_aitken = (d_2h**2 - d_4h * d_h) / denom_aitken
else:
    y_aitken = d_h

# Порядок точності p
p_val = (1 / np.log(2)) * np.log(abs((d_4h - d_2h) / (d_2h - d_h)))
error_aitken = abs(y_aitken - exact_val)

print(f"--- Метод Ейткена ---")
print(f"Значення y'(4h): {d_4h:.10f}")
print(f"Уточнене значення y_E: {y_aitken:.10f}")
print(f"Оцінка порядку точності p: {p_val:.2f}")
print(f"Похибка R3: {error_aitken:.2e}\n")

# --- Побудова графіків ---

# Графік 1: Залежність похибки від кроку h
plt.figure(figsize=(10, 5))
plt.loglog(h_values, errors, label='|y\'_approx - y\'_exact|', color='blue')
plt.axvline(best_h, color='red', linestyle='--', label=f'h_opt = {best_h:.1e}')
plt.title('Залежність похибки чисельного диференціювання від кроку h')
plt.xlabel('Крок h (log scale)')
plt.ylabel('Абсолютна похибка R (log scale)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# Графік 2: Функція вологості M(t) та її похідна M'(t)
t_range = np.linspace(0, 20, 400)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t_range, M(t_range), color='green')
plt.title('Модель вологості M(t)')
plt.xlabel('Час t')
plt.ylabel('Вологість')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_range, M_prime_exact(t_range), color='orange', label='M\'(t) exact')
plt.title('Швидкість зміни вологості M\'(t)')
plt.xlabel('Час t')
plt.ylabel('Швидкість')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()