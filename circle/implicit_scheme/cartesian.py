import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Параметры задачи
R = 1.0           # Радиус круга
N = 51            # Количество узлов по каждой оси
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

# Начальные условия (гауссов импульс в центре)
u0 = np.exp(-(X**2 + Y**2)/(2*sigma**2)) * mask

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

# Вычисление первого шага с использованием начальной скорости (здесь нулевая)
laplacian_u0 = np.zeros_like(u0)
for i in range(1, N-1):
    for j in range(1, N-1):
        if mask[i, j]:
            laplacian_u0[i, j] = (u0[i+1, j] - 2*u0[i, j] + u0[i-1, j])/h**2 + \
                                (u0[i, j+1] - 2*u0[i, j] + u0[i, j-1])/h**2
laplacian_u0_flat = laplacian_u0[mask].flatten()
u_curr = u_prev + (c**2 * dt**2 / 2) * laplacian_u0_flat

# Сохранение истории для анимации
U_history = []

# Временной цикл
for n in range(1, t_steps):
    b = 2 * u_curr - u_prev
    u_next = splinalg.spsolve(A, b)



    u_prev, u_curr = u_curr, u_next

    print(u0.shape, u_prev.shape, u_curr.shape, u_next.shape)

    u_grid = np.zeros((N, N))
    u_grid[mask] = u_curr
    U_history.append(u_grid)

# Создание анимации
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-1, 1)

def update(frame):
    ax.clear()
    ax.set_zlim(-1, 1)
    # print(X.shape, Y.shape, U_history[frame].shape)
    surf = ax.plot_surface(X, Y, U_history[frame], cmap='viridis', rstride=1, cstride=1)
    return surf,

ani = animation.FuncAnimation(fig, update, frames=len(U_history), interval=50, blit=False)
plt.show()