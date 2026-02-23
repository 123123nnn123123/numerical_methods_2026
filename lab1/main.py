import requests
import numpy as np
import matplotlib.pyplot as plt

# 1. ОТРИМАННЯ ДАНИХ (Пункт 1-3 методички)
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

try:
    response = requests.get(url)
    data = response.json()
    results = data["results"]
except Exception as e:
    print(f"Помилка при отриманні даних: {e}")
    exit()

# Координати та висоти [cite: 94, 130]
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])
n_total = len(elevations)


# 2. Обчислення кумулятивної відстані (Гаверсинус) [cite: 121-129]
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


distances = [0]
for i in range(1, n_total):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

x_full = np.array(distances)
y_full = elevations


# 3. МЕТОД ПРОГОНКИ (Згідно стор. 3-4 методички) [cite: 53-72]
def thomas_algorithm(alpha, beta, gamma, delta):
    n = len(delta)
    A = np.zeros(n)
    B = np.zeros(n)

    # Пряма прогонка [cite: 63, 64]
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]

    for i in range(1, n - 1):
        denom = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denom

    # Зворотна прогонка [cite: 69, 71]
    res = np.zeros(n)
    res[-1] = (delta[-1] - alpha[-1] * B[-2]) / (alpha[-1] * A[-2] + beta[-1])
    for i in range(n - 2, -1, -1):
        res[i] = A[i] * res[i + 1] + B[i]
    return res


# 4. ПОБУДОВА СПЛАЙНА (Пункт 6-9) [cite: 15-27, 36-41]
def build_spline(x, y):
    n = len(x)
    h = np.diff(x)

    alpha = np.zeros(n)
    beta = np.ones(n)
    gamma = np.zeros(n)
    delta = np.zeros(n)

    # Умови вільного сплайна c1=0 [cite: 45, 48]
    beta[0] = 1.0

    for i in range(1, n - 1):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Крайня умова [cite: 46, 49]
    alpha[-1] = h[-2]
    beta[-1] = 2 * (h[-2] + 0.1)  # додаємо мале значення для стабільності
    delta[-1] = 0

    c = thomas_algorithm(alpha, beta, gamma, delta)

    a = y[:-1]
    d = np.zeros(n - 1)
    b = np.zeros(n - 1)

    for i in range(n - 1):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = (y[i + 1] - y[i]) / h[i] - (h[i] / 3) * (c[i + 1] + 2 * c[i])

    return a, b, c, d


def get_spline_y(x_eval, x_nodes, a, b, c, d):
    res = np.zeros_like(x_eval)
    for i in range(len(x_nodes) - 1):
        mask = (x_eval >= x_nodes[i]) & (x_eval <= x_nodes[i + 1])
        dx = x_eval[mask] - x_nodes[i]
        # Формула кубічного сплайна [cite: 11]
        res[mask] = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
    return res


# 5. Побудова графіків (10, 15, 20 вузлів)
node_counts = [10, 15, 20]
plt.figure(figsize=(12, 10))
xx = np.linspace(x_full[0], x_full[-1], 500)

last_y_approx = None  # для графіка похибки

for idx, count in enumerate(node_counts):
    # Вибираємо рівновіддалені вузли
    indices = np.linspace(0, n_total - 1, count, dtype=int)
    x_n = x_full[indices]
    y_n = y_full[indices]

    a, b, c, d = build_spline(x_n, y_n)
    yy = get_spline_y(xx, x_n, a, b, c, d)

    if count == 20:
        last_y_approx = get_spline_y(x_full, x_n, a, b, c, d)

    plt.subplot(2, 2, idx + 1)
    plt.plot(x_full, y_full, 'ro', markersize=3, label='Точки API')
    plt.plot(xx, yy, 'b-', label=f'Сплайн ({count} вузлів)')
    plt.title(f"Інтерполяція: {count} вузлів")
    plt.grid(True)
    plt.legend()

# 6. ГРАФІК ПОХИБКИ (Пункт 12)
if last_y_approx is not None:
    error = np.abs(y_full - last_y_approx)
    plt.subplot(2, 2, 4)
    plt.plot(x_full, error, 'g-', label='Похибка ε')
    plt.fill_between(x_full, error, color='green', alpha=0.1)
    plt.title("Графік похибки (для 20 вузлів)")
    plt.xlabel("Відстань (м)")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# 7. ХАРАКТЕРИСТИКИ [cite: 152-157, 170-180]
total_ascent = sum(max(y_full[i] - y_full[i - 1], 0) for i in range(1, n_total))
energy = 80 * 9.81 * total_ascent

print("\n--- РЕЗУЛЬТАТИ ---")
print(f"Загальна відстань: {x_full[-1]:.2f} м")
print(f"Сумарний підйом: {total_ascent:.2f} м")
print(f"Механічна робота: {energy / 1000:.2f} кДж")
print(f"Енергія: {energy / 4184:.2f} ккал")