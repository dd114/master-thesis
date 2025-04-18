import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Параметры задачи
R = 2.0
r = 1           # Радиус круга
c = 1.0           # Скорость распространения волны
h = 0.01          # Шаг пространственной сетки
dt = 0.9 * h / (c * np.sqrt(2))  # Шаг времени (условие Куранта)
Tmax = 5.0        # Время моделирования
sigma = 0.1       # Параметр начального гауссова импульса

print(f"dt = {dt}")

a, b = 1, 0.5
max_a_b = max(a, b)
N = int(2 * max(a, b) / h) + 1

x = np.linspace(-R, R, N)
y = np.linspace(-R, R, N)
X, Y = np.meshgrid(x, y)

phi = np.pi / 4 

x_0, y_0 = 1, 1

new_X, new_Y = (X - x_0) * np.cos(phi) + (Y - y_0) * -np.sin(phi), (X - x_0) * np.sin(phi) + (Y - y_0) * np.cos(phi)

ellipse_mask = (((new_X) / a)**2 + ((new_Y - 0) / b)**2) <= 1
circle_mask = (X**2 + Y**2) <= r**2

mask = circle_mask | ellipse_mask  # Маска внутренних точек

# Начальные условия
from scipy import special

m = 1 # Bessel order
n = 3 # number of root

alpha_m = special.jn_zeros(m, n) / R 

# Начальные условия (пример: гауссов импульс)
def initial_state(x, y):
    # return special.jv(m, alpha_m[-1] * np.sqrt(x**2 + y**2)) * np.sin(1 * np.arctan2(y, x))
    # return 0.5 * np.exp( - ((x - 0.95) ** 2) / (2 * 0.001)) * np.sin(60 * y)
    # return 0.5 * np.exp(-((X**2 + Y**2 - 0.9) ** 2) / (sigma**2)) * np.cos(60 * np.arctan2(Y, X))
    # return 0.5 * np.exp(-((X**2 + Y**2 - 0.9) ** 2) / (sigma**2))
    return np.ones_like(X)

u_curr = initial_state(X, Y) * mask
# u_curr = (1 * np.exp( - ((X - 0.8) ** 2) / (2 * 0.001)) * np.exp( - ((Y - 0.8) ** 2) / (2 * 0.001))) * mask
# u_curr = (np.exp(-(X**2 + Y**2) / (sigma**2)) * 1 * np.sin(1 * np.arctan2(Y, X))) * mask
# u_curr = (np.exp(- np.abs((X**2 + Y**2 - 0.81) )/ (sigma**2)) ) * mask
u_prev = np.zeros_like(u_curr)



# Начальное состояние
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Начальное состояние', fontsize=14, fontweight="bold")
surface = ax.plot_surface(X, Y, u_curr, cmap='viridis')
ax.set_zlim(-1, 1)

ax.set_xlabel("X", fontsize=14, fontweight="bold")
ax.set_ylabel("Y", fontsize=14, fontweight="bold")
ax.set_zlabel("U", fontsize=14, fontweight="bold")

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
    surf = ax.plot_surface(X, Y, u_curr, cmap='viridis', rstride=2, cstride=2)
    ax.set_zlim(-1, 1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    return surf,

# Создание анимации
ani = FuncAnimation(fig, update, frames=int(Tmax/dt), interval=50, blit=False)
plt.show()