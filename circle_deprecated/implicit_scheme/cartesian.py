import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Параметры задачи
R = 1.0           # Радиус круга
N = 100            # Количество узлов по каждой оси
c = 1.0           # Скорость волны
sigma = 0.2       # Параметр начального гауссова распределения
t_steps = 100     # Количество временных шагов
dt = 0.02         # Шаг по времени
output_step = 1   # Шаг для вывода анимации (уменьшает количество кадров)

# Создание сетки
x = np.linspace(-R, R, N)
y = np.linspace(-R, R, N)
X, Y = np.meshgrid(x, y)
h = x[1] - x[0]  # Шаг сетки

# Маска для внутренних точек круга
mask = (X**2 + Y**2) <= R**2

# Начальные условия
from scipy import special

m = 1 # Bessel order
n = 3 # number of root

alpha_m = special.jn_zeros(m, n) / R 

# Начальные условия (пример: гауссов импульс)
def initial_state(x, y):
    # return special.jv(m, alpha_m[-1] * np.sqrt(x**2 + y**2)) * np.cos(1 * np.arctan2(y, x)) * mask
    return 0.5 * np.exp( - ((x - 0.9) ** 2) / (2 * 0.001)) * np.sin(60 * y) * mask

    return np.zeros_like(x)

def initial_speed(X, Y):

    return np.zeros_like(X) * mask

# Начальные условия (гауссов импульс в центре)
u0 = initial_state(X, Y)

# Создание индексов для внутренних точек
indices = np.full((N, N), -1, dtype=int)
current_idx = 0
for i in range(N):
    for j in range(N):
        if mask[i, j]:
            indices[i, j] = current_idx
            current_idx += 1
M = current_idx  # Число внутренних точек

# Параметр устойчивости
s = (c * dt / h)**2

# Построение матрицы системы
A = sp.lil_matrix((M, M))

for i in range(N):
    for j in range(N):
        if mask[i, j]:
            k = indices[i, j]
            A[k, k] = 1 + 4*s
            # Соседи
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N and mask[ni, nj]:
                    nk = indices[ni, nj]
                    A[k, nk] = -s
A = A.tocsr()

u_prev = u0[mask].flatten().astype(np.float64)

# Вычисление первого шага с использованием начальной скорости

speed = initial_speed(X, Y)
u_curr = u_prev[:] + speed[mask].flatten() * dt

# Сохранение истории для анимации
# U_history = []

# # Временной цикл
# for n in range(1, t_steps):
#     print(u0.shape, u_prev.shape, u_curr.shape)

#     b = 2 * u_curr - u_prev
#     u_next = splinalg.spsolve(A, b)
#     print(u0.shape, u_prev.shape, u_curr.shape, u_next.shape)
#     u_prev, u_curr = u_curr, u_next

#     # u_grid = np.zeros((Nr, Nphi))
#     # u_grid = u_curr[:]
#     U_history.append(u_curr.reshape((Nr, Nphi)))

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Начальное состояние', fontsize=14, fontweight="bold")
surface = ax.plot_surface(X, Y, u0, cmap='viridis')
ax.set_zlim(-1, 1)

ax.set_xlabel("X", fontsize=14, fontweight="bold")
ax.set_ylabel("Y", fontsize=14, fontweight="bold")
ax.set_zlabel("U", fontsize=14, fontweight="bold")

# Начальная скорость
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Начальная скорость', fontsize=14, fontweight="bold")
surface = ax.plot_surface(X, Y, speed, cmap='viridis')
ax.set_zlim(-1, 1)

ax.set_xlabel("X", fontsize=14, fontweight="bold")
ax.set_ylabel("Y", fontsize=14, fontweight="bold")
ax.set_zlabel("U", fontsize=14, fontweight="bold")

# Создание анимации
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-1, 1)

t_curr = 0

def update(frame):
    global u_prev, u_curr,t_curr
    
    t_curr += dt

    b = 2 * u_curr - u_prev
    u_next = splinalg.spsolve(A, b)
    # print(u0.shape, u_prev.shape, u_curr.shape, u_next.shape)
    print(u_next.max())
    u_prev, u_curr = u_curr, u_next

    ax.clear()
    ax.set_title(f'текущее время = {round(t_curr, 4)}, dt = {round(dt, 4)}, dx=dy = {round(h, 4)}', fontsize=14, fontweight="bold")
    
    ax.set_xlabel("X", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y", fontsize=14, fontweight="bold")
    ax.set_zlabel("U", fontsize=14, fontweight="bold")

    ax.set_zlim(-1, 1)

    u_grid = np.zeros((N, N))
    u_grid[mask] = u_curr

    surf = ax.plot_surface(X, Y, u_grid, cmap='viridis', rstride=1, cstride=1)
    return surf,

ani = animation.FuncAnimation(fig, update, frames=5, interval=10, blit=False)
plt.show()