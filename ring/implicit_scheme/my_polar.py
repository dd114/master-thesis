import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Параметры задачи
r = 0.1
R = 1.0           # Радиус круга
Nr = 100            # Количество узлов по r
Nphi = 150            # Количество узлов по phi
c = 1.0           # Скорость волны
sigma = 0.2       # Параметр начального гауссова распределения
t_steps = 100     # Количество временных шагов
dt = 0.02         # Шаг по времени
output_step = 1   # Шаг для вывода анимации (уменьшает количество кадров)

# Сетка
rad = np.linspace(r, R, Nr)
phi = np.linspace(0, 2*np.pi, Nphi, endpoint=False)
Rad_grid, Phi_grid = np.meshgrid(rad, phi, indexing='ij')

X = Rad_grid * np.cos(Phi_grid)
Y = Rad_grid * np.sin(Phi_grid)


# Создание сетки
# x = np.linspace(-R, R, N)
# y = np.linspace(-R, R, N)
# X, Y = np.meshgrid(x, y)
dr = rad[1] - rad[0]  # Шаг сетки
dphi = phi[1] - phi[0]  # Шаг сетки

# Маска для внутренних точек круга
# mask = ((X**2 + Y**2) <= R**2) * ((X**2 + Y**2) >= 0.1)

from scipy import special

m = 1 # Bessel order
n = 3 # number of root

alpha_m = special.jn_zeros(m, n) / R 

# Начальные условия (пример: гауссов импульс)
def initial_state(rad, phi):
    return special.jv(m, alpha_m[-1] * rad) * np.cos(m * phi)
    # return 0.1 * np.exp( - ((rad - 0.9) ** 2) / (2 * 0.001)) 
    return 0.5 * np.exp( - ((rad - 0.9) ** 2) / (2 * 0.001)) * np.cos(60 * phi)

    return np.zeros_like(rad)

def initial_speed(rad, phi):
    # return np.exp( - ((rad - 0.9) ** 2) / (2 * 0.001))

    return np.zeros_like(rad)

# Начальные условия (гауссов импульс в центре)
u0 = initial_state(Rad_grid, Phi_grid)


# Создание индексов для внутренних точек
indices = np.full((Nr, Nphi), -1, dtype=int)
current_idx = 0
for i in range(Nr):
    for j in range(Nphi):
        indices[i, j] = current_idx
        current_idx += 1

M = current_idx  # Число внутренних точек

# Параметр устойчивости
# s = (c * dt / h)**2
s = (c * dt)

# Построение матрицы системы
A = sp.lil_matrix((M, M))

for i in range(Nr):
    for j in range(Nphi):
        k = indices[i, j]
        A[k, k] = 1 + 2 * ((s / dr) ** 2) + 2 * (s / (rad[i] * dphi)) ** 2

        # Соседи

        pi, pj, ni, nj = i + 1, (j + 1) % Nphi, i - 1, (j - 1) % Nphi

        if 0 <= pi < Nr:
            cur_u = indices[pi, j]
            A[k, cur_u] = - (s / dr) ** 2 - (s ** 2) / (2 * rad[i] * dr)

        if 0 <= ni < Nr:
            cur_u = indices[ni, j]
            A[k, cur_u] = - (s / dr) ** 2 + (s ** 2) / (2 * rad[i] * dr)

        # if 0 <= pj < Nphi:
        cur_u = indices[i, pj]
        A[k, cur_u] = - (s / (rad[i] * dphi)) ** 2

        # if 0 <= nj < Nphi:
        cur_u = indices[i, nj]
        A[k, cur_u] = - (s / (rad[i] * dphi)) ** 2


A = A.tocsr()

u_prev = u0.flatten().astype(np.float64)

# Вычисление первого шага с использованием начальной скорости

speed = initial_speed(Rad_grid, Phi_grid)
u_curr = u_prev[:] + speed.flatten() * dt

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
    ax.set_title(f'текущее время = {round(t_curr, 4)}, dt = {round(dt, 4)}, dr = {round(dr, 4)}, dphi = {round(dphi, 4)}', fontsize=14, fontweight="bold")
    
    ax.set_xlabel("X", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y", fontsize=14, fontweight="bold")
    ax.set_zlabel("U", fontsize=14, fontweight="bold")

    ax.set_zlim(-1, 1)
    surf = ax.plot_surface(X, Y, u_curr.reshape((Nr, Nphi)), cmap='viridis', rstride=1, cstride=1)
    return surf,

ani = animation.FuncAnimation(fig, update, frames=5, interval=10, blit=False)
plt.show()