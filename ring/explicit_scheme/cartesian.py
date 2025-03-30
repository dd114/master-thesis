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
N = int(2 * R / h) + 1
x = np.linspace(-R, R, N)
y = np.linspace(-R, R, N)
X, Y = np.meshgrid(x, y)
mask = (X**2 + Y**2) <= R**2  # Маска внутренних точек круга

# Начальные условия
# u_curr = (1 * np.exp( - ((X - 0.8) ** 2) / (2 * 0.001)) * np.exp( - ((Y - 0.8) ** 2) / (2 * 0.001))) * mask
u_curr = (np.exp(-(X**2 + Y**2) / (sigma**2)) * 1 * np.sin(1 * np.arctan2(Y, X))) * mask
# u_curr = (np.exp(- np.abs((X**2 + Y**2 - 0.81) )/ (sigma**2)) ) * mask
u_prev = np.zeros_like(u_curr)

# Вычисление лапласиана для начального условия
laplacian = np.zeros_like(u_curr)
for i in range(N):
    for j in range(N):
        if mask[i, j]:
            left = u_curr[i-1, j] if i > 0 else 0
            right = u_curr[i+1, j] if i < N-1 else 0
            up = u_curr[i, j+1] if j < N-1 else 0
            down = u_curr[i, j-1] if j > 0 else 0
            laplacian[i, j] = left + right + up + down - 4 * u_curr[i, j]

u_prev = u_curr + 0.5 * (c * dt / h)**2 * laplacian
u_prev *= mask  # Применение граничных условий

# Настройка анимации
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, u_curr, cmap='viridis', rstride=1, cstride=1)
ax.set_zlim(-1, 1)

def update(frame):
    global u_prev, u_curr
    laplacian = np.zeros_like(u_curr)
    
    # Вычисление лапласиана
    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                left = u_curr[i-1, j] if i > 0 else 0
                right = u_curr[i+1, j] if i < N-1 else 0
                up = u_curr[i, j+1] if j < N-1 else 0
                down = u_curr[i, j-1] if j > 0 else 0
                laplacian[i, j] = left + right + up + down - 4 * u_curr[i, j]
    
    # Обновление решения
    u_next = 2 * u_curr - u_prev + (c * dt / h)**2 * laplacian
    u_next *= mask
    
    u_prev, u_curr = u_curr.copy(), u_next
    
    # Обновление графика
    ax.clear()
    surf = ax.plot_surface(X, Y, u_curr, cmap='viridis', rstride=1, cstride=1)
    ax.set_zlim(-1, 1)
    return surf,

# Создание анимации
ani = FuncAnimation(fig, update, frames=int(Tmax/dt), interval=50, blit=False)
plt.show()