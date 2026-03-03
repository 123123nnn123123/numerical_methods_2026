import requests
import numpy as np
import matplotlib.pyplot as plt

# 1. ОТРИМАННЯ ДАНИХ
raw_coords = "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

url = "https://api.open-elevation.com/api/v1/lookup"

try:
    print("Запит до API...")
    response = requests.get(url, params={"locations": raw_coords}, timeout=15)
    response.raise_for_status()
    data = response.json()
    results = data["results"]
except Exception:
    print("Використовуємо локальні дані...")
    test_elevations = [1450, 1462, 1480, 1495, 1510, 1535, 1520, 1490, 1505, 1485, 1460, 1430, 1410, 1380, 1365, 1340,
                       1355, 1330, 1315, 1305, 1300]
    results = [
        {"latitude": float(p.split(',')[0]), "longitude": float(p.split(',')[1]), "elevation": test_elevations[i]}
        for i, p in enumerate(raw_coords.split('|'))]

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])
n_total = len(elevations)


# 2. МАТЕМАТИЧНІ ФУНКЦІЇ (Haversine, Thomas, Spline)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


distances = [0]
for i in range(1, n_total):
    distances.append(distances[-1] + haversine(*coords[i - 1], *coords[i]))

x_full, y_full = np.array(distances), elevations


def thomas_algorithm(alpha, beta, gamma, delta):
    n = len(delta)
    A, B = np.zeros(n), np.zeros(n)
    A[0], B[0] = -gamma[0] / beta[0], delta[0] / beta[0]
    for i in range(1, n - 1):
        denom = alpha[i] * A[i - 1] + beta[i]
        A[i], B[i] = -gamma[i] / denom, (delta[i] - alpha[i] * B[i - 1]) / denom
    res = np.zeros(n)
    res[-1] = (delta[-1] - alpha[-1] * B[-2]) / (alpha[-1] * A[-2] + beta[-1])
    for i in range(n - 2, -1, -1):
        res[i] = A[i] * res[i + 1] + B[i]
    return res


def build_spline(x, y):
    n = len(x)
    h = np.diff(x)
    alpha, beta, gamma, delta = np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n)
    for i in range(1, n - 1):
        alpha[i], beta[i], gamma[i] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    c = thomas_algorithm(alpha, beta, gamma, delta)
    a, d = y[:-1], (c[1:] - c[:-1]) / (3 * h)
    b = (y[1:] - y[:-1]) / h - (h / 3) * (c[1:] + 2 * c[:-1])
    return a, b, c[:-1], d


def get_spline_y(x_eval, x_nodes, a, b, c, d):
    res = np.zeros_like(x_eval)
    for i in range(len(x_nodes) - 1):
        mask = (x_eval >= x_nodes[i]) & (x_eval <= x_nodes[i + 1] + 1e-9)
        dx = x_eval[mask] - x_nodes[i]
        res[mask] = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
    return res


# 3. ПОБУДОВА ГРАФІКІВ (3 рядки: Профіль | Похибка)
node_counts = [10, 15, 20]
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
xx = np.linspace(x_full[0], x_full[-1], 500)

for i, count in enumerate(node_counts):
    # Розрахунок
    indices = np.linspace(0, n_total - 1, count, dtype=int)
    x_n, y_n = x_full[indices], y_full[indices]
    a, b, c, d = build_spline(x_n, y_n)

    yy_smooth = get_spline_y(xx, x_n, a, b, c, d)
    y_approx = get_spline_y(x_full, x_n, a, b, c, d)
    error = np.abs(y_full - y_approx)

    # Ліва колонка: Графік сплайна
    axes[i, 0].plot(x_full, y_full, 'ro', markersize=4, alpha=0.4, label='Дані API')
    axes[i, 0].plot(xx, yy_smooth, 'b-', linewidth=2, label=f'Сплайн ({count} вузлів)')
    axes[i, 0].scatter(x_n, y_n, color='black', zorder=5, label='Вузли')
    axes[i, 0].set_title(f"Профіль висот: {count} вузлів")
    axes[i, 0].grid(True, linestyle='--', alpha=0.6)
    axes[i, 0].legend()

    # Права колонка: Графік похибки
    axes[i, 1].fill_between(x_full, error, color='green', alpha=0.3)
    axes[i, 1].plot(x_full, error, 'g-', label='Абсолютна похибка')
    axes[i, 1].set_title(f"Похибка: {count} вузлів")
    axes[i, 1].set_xlabel("Відстань (м)")
    axes[i, 1].set_ylabel("Похибка (м)")
    axes[i, 1].grid(True, linestyle='--', alpha=0.6)
    axes[i, 1].legend()

plt.tight_layout()
plt.show()

# 4. РЕЗУЛЬТАТИ В КОНСОЛЬ
print("\nАНАЛІЗ ТОЧНОСТІ:")
for count in node_counts:
    indices = np.linspace(0, n_total - 1, count, dtype=int)
    a, b, c, d = build_spline(x_full[indices], y_full[indices])
    err = np.abs(y_full - get_spline_y(x_full, x_full[indices], a, b, c, d))
    print(f"Вузлів: {count:2d} | Макс. похибка: {np.max(err):.2f} м | Сер. похибка: {np.mean(err):.2f} м")