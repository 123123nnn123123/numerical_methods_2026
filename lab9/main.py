import numpy as np
import matplotlib.pyplot as plt

# СИСТЕМА РІВНЯНЬ
def f1(x, y):
    return x**2 + y**2 - 1
def f2(x, y):
    return x - y


# ЦІЛЬОВА ФУНКЦІЯ
def objective(point):
    x, y = point
    return f1(x, y) ** 2 + f2(x, y) ** 2


# МЕТОД ХУКА-ДЖИВСА
def hooke_jeeves(func, start_point, step=0.5, alpha=0.5, eps=1e-6, max_iter=1000):
    x = np.array(start_point, dtype=float)
    trajectory = [x.copy()]
    iterations_data = []
    iteration = 0

    while step > eps and iteration < max_iter:
        new_x = x.copy()
        for i in range(len(x)):
            temp = new_x.copy()
            temp[i] += step
            if func(temp) < func(new_x):
                new_x = temp
            else:
                temp = new_x.copy()
                temp[i] -= step
                if func(temp) < func(new_x):
                    new_x = temp
        if func(new_x) < func(x):
            pattern = new_x + (new_x - x)
            if func(pattern) < func(new_x):
                x = pattern
            else:
                x = new_x
            trajectory.append(x.copy())
        else:
            step *= alpha
        iteration += 1
        iterations_data.append([iteration, x[0], x[1], func(x)])

    return x, func(x), trajectory, iterations_data


# ПАРАМЕТРИ
start_point = [1.5, -0.5]
minimum_point, minimum_value, trajectory, iterations_data = hooke_jeeves(objective, start_point)
trajectory = np.array(trajectory)

# --- ЗАПИС УСІХ ІТЕРАЦІЙ В ОКРЕМИЙ ФАЙЛ ---
with open("trajectory.txt", "w", encoding="utf-8") as f:
    f.write(f"{'Ітерація':<10} | {'X':<10} | {'Y':<10} | {'f(x,y)':<12}\n")
    f.write("-" * 50 + "\n")
    for data in iterations_data:
        f.write(f"{data[0]:<10} | {data[1]:.6f} | {data[2]:.6f} | {data[3]:.10f}\n")

# ПІДГОТОВКА ДАНИХ ДЛЯ ГРАФІКІВ
x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_range, y_range)

Z1 = f1(X, Y)
Z2 = f2(X, Y)
F = objective([X, Y])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- ГРАФІК 1: СИСТЕМА РІВНЯНЬ ---
ax1.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
ax1.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
ax1.plot(start_point[0], start_point[1], 'ko', label='Старт')
ax1.plot(minimum_point[0], minimum_point[1], 'gx', markersize=10, mew=3, label='Розв’язок')

ax1.set_title("Перетин ліній f1(x,y)=0 та f2(x,y)=0")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# --- ГРАФІК 2: КОНТУРНИЙ ГРАФІК ЦІЛЬОВОЇ ФУНКЦІЇ ---
cp = ax2.contour(X, Y, F, levels=50, cmap='viridis')
plt.colorbar(cp, ax=ax2, label='Φ(X)')
ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', linewidth=1.5, label='Траєкторія')
ax2.plot(trajectory[0, 0], trajectory[0, 1], 'bo', label='Початок')
ax2.plot(trajectory[-1, 0], trajectory[-1, 1], 'go', markersize=8, label='Фініш')

ax2.set_title("Контурний графік цільової функції Φ(X)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.show()

# ВИВІД У КОНСОЛЬ
for data in iterations_data:
    print(f"Ітерація {data[0]}: x = {data[1]:.6f}, y = {data[2]:.6f}, f = {data[3]:.10f}")

print(f"\nРезультат: x = {minimum_point[0]:.6f}, y = {minimum_point[1]:.6f}")
print(f"Значення функції: {minimum_value:.10f}")
print(f"Ітерацій: {len(iterations_data)}")