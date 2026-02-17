import requests
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Отримання даних з API
# -------------------------------

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

data = requests.get(url).json()
results = data["results"]

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])
n = len(elevations)

# -------------------------------
# 2. Кумулятивна відстань
# -------------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

distances = [0]
for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)

x = np.array(distances)
y = elevations

# -------------------------------
# 3. Метод прогонки
# -------------------------------

def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_ = np.zeros(n)
    d_ = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n):
        temp = b[i] - a[i] * c_[i-1]
        c_[i] = c[i] / temp if i < n-1 else 0
        d_[i] = (d[i] - a[i] * d_[i-1]) / temp

    x = np.zeros(n)
    x[-1] = d_[-1]

    for i in reversed(range(n-1)):
        x[i] = d_[i] - c_[i] * x[i+1]

    return x

# -------------------------------
# 4. Кубічний сплайн (натуральний)
# -------------------------------

def cubic_spline(x, y):
    n = len(x)
    h = np.diff(x)

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)

    B[0] = 1
    B[-1] = 1

    for i in range(1, n-1):
        A[i] = h[i-1]
        B[i] = 2*(h[i-1] + h[i])
        C[i] = h[i]
        D[i] = 3*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    c = thomas_algorithm(A, B, C, D)

    for i in range(n-1):
        a[i] = y[i]
        b[i] = (y[i+1]-y[i])/h[i] - h[i]*(2*c[i]+c[i+1])/3
        d[i] = (c[i+1]-c[i])/(3*h[i])

    return a, b, c, d

# -------------------------------
# 5. Обчислення сплайна
# -------------------------------

a, b, c, d = cubic_spline(x, y)

def spline_eval(xp, x, a, b, c, d):
    yp = np.zeros_like(xp)
    for i in range(len(x)-1):
        mask = (xp >= x[i]) & (xp <= x[i+1])
        dx = xp[mask] - x[i]
        yp[mask] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return yp

xx = np.linspace(x[0], x[-1], 1000)
yy = spline_eval(xx, x, a, b, c, d)

# -------------------------------
# 6. Графік
# -------------------------------

plt.figure(figsize=(10,6))
plt.plot(x, y, 'o', label="Дискретні точки")
plt.plot(xx, yy, '-', label="Кубічний сплайн")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Профіль маршруту Заросляк – Говерла")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# 7. Характеристики маршруту
# -------------------------------

total_ascent = sum(max(y[i]-y[i-1],0) for i in range(1,n))
total_descent = sum(max(y[i-1]-y[i],0) for i in range(1,n))

print("Загальна довжина маршруту (м):", x[-1])
print("Сумарний підйом (м):", total_ascent)
print("Сумарний спуск (м):", total_descent)

# Енергія
mass = 80
g = 9.81
energy = mass * g * total_ascent

print("Механічна робота (Дж):", energy)
print("Механічна робота (кДж):", energy/1000)
print("Енергія (ккал):", energy/4184)