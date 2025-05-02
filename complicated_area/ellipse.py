import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Параметры задачи
R = 1.0           # Радиус круга
c = 1.0           # Скорость распространения волны
h = 0.02          # Шаг пространственной сетки
dt = 0.9 * h / (c * np.sqrt(2))  # Шаг времени (условие Куранта)
Tmax = 5.0        # Время моделирования
sigma = 0.1       # Параметр начального гауссова импульса

print(f"dt = {dt}")

# Создание сетки

a, b = 2, 1
max_a_b = max(a, b)
N = int(2 * max(a, b) / h) + 1

x = np.linspace(-a, a, N)
y = np.linspace(-b, b, N)
X, Y = np.meshgrid(x, y, indexing='xy')
mask = ((X / a)**2 + (Y / b)**2) <= 1  # Маска внутренних точек элипса

u = np.ones_like(X) * mask

print("u.shape =", u.shape)

fig = plt.figure(figsize=(12, 10))

ax = fig.add_subplot(121, projection='3d')
ax.set_xlabel('X')
ax.set_xlim(-max_a_b, max_a_b)
ax.set_ylabel('Y')
ax.set_ylim(-max_a_b, max_a_b)
# ax.set_zlim(-1, 1)
ax.plot_surface(X, Y, u, cmap='viridis')

phi = np.pi / 4

new_X, new_Y = X * np.cos(phi) + Y * -np.sin(phi), X * np.sin(phi) + Y * np.cos(phi)

ax = fig.add_subplot(122, projection='3d')
ax.set_xlabel('new_X')
ax.set_xlim(-max_a_b, max_a_b)
ax.set_ylabel('new_Y')
ax.set_ylim(-max_a_b, max_a_b)
# ax.set_zlim(-1, 1)
ax.plot_surface(new_X, new_Y, u, cmap='viridis')

plt.show()
