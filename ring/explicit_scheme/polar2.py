import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Параметры
# r = 0.0001
# r = 0.035
r = 0.1
# r = 0.8
R = 1.0           # Радиус круга
c = 1.0           # Скорость волны
# Nr = 200           # Число узлов по rad
# Nphi = 300         # Число узлов по φ
Nr = 100           # Число узлов по rad
Nphi = 200         # Число узлов по φ
dr = R / (Nr - 1) # Шаг по rad
dphi = 2 * np.pi / Nphi  # Шаг по φ
dt = 0.99 * min(dr, dphi, r, dphi * r) / (c * np.sqrt(2))  # Шаг по времени (условие Куранта); np.sqrt(3)?
t_curr = 0
t_max = 2.0       # Время моделирования

print(f"frames={int(t_max / dt)}")
print(f'dr = {dr}, dphi = {dphi}, r = {r}, dphi * r = {dphi * r}')
print(f'min(dr, dphi, r, dphi * r) = {min(dr, dphi, r, dphi * r)}')
print(f'dt = {dt}')

offset = +1e-4

# Сетка
rad = np.linspace(r, R, Nr)
phi = np.linspace(0, 2*np.pi, Nphi)
Rad_grid, Phi_grid = np.meshgrid(rad, phi, indexing='ij')

# Инициализация
u_prev = np.zeros((Nr, Nphi))  # u^{n-1}
u_curr = np.zeros((Nr, Nphi))  # u^n

amp = 0.5
expect = 0.8
sigma2 = 0.001

# Начальные условия (пример: гауссов импульс)
def initial_f(rad, phi):
    # return 0.1 * np.exp( - ((rad - 0.9) ** 2) / (2 * 0.001))
    return 0.5 * np.exp( - ((rad - 0.9) ** 2) / (2 * 0.001)) * np.exp( - ((phi - np.pi) ** 2) / (2 * 0.001))
    # return np.exp(-50*(rad - 0.3)**2) * 0.5 * np.sin(phi)
    # return -np.exp(-50*(rad - 0.3)**2)

u_prev = initial_f(Rad_grid, Phi_grid)

# Граничные условия при rad=R и rad=r
u_prev[0, :] = 0.0
u_prev[-1, :] = 0.0

u_curr[:, :] = u_prev[:, :]

X = Rad_grid * np.cos(Phi_grid)
Y = Rad_grid * np.sin(Phi_grid)


# Начальные условия
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Начальные условия', fontsize=14, fontweight="bold")
surface = ax.plot_surface(X, Y, u_curr, cmap='viridis')
ax.set_zlim(-1, 1)

ax.set_xlabel("X", fontsize=14, fontweight="bold")
ax.set_ylabel("Y", fontsize=14, fontweight="bold")
ax.set_zlabel("U", fontsize=14, fontweight="bold")

# Подготовка для анимации
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
# surface = ax.plot_surface(X, Y, u_curr, cmap='viridis')
ax.set_zlim(-1, 1)


# Функция обновления кадра
def update(frame):
    global u_prev, u_curr, t_curr
    
    u_next = np.zeros((Nr, Nphi))
    
    # Вычисление нового слоя
    for i in range(1, Nr-1):
        for j in range(Nphi):
            j_prev = (j - 1) % Nphi
            j_next = (j + 1) % Nphi
            
            # Радиальная часть
            d2u_dr2 = (u_curr[i+1,j] - 2*u_curr[i,j] + u_curr[i-1,j]) / dr**2
            du_dr = (u_curr[i+1,j] - u_curr[i-1,j]) / (2 * dr)
            radial = d2u_dr2 + du_dr / (rad[i])
            
            # Угловая часть
            d2u_dphi2 = (u_curr[i,j_next] - 2*u_curr[i,j] + u_curr[i,j_prev]) / dphi**2
            angular = d2u_dphi2 / (rad[i]**2)
            
            # Обновление
            u_next[i,j] = 2*u_curr[i,j] - u_prev[i,j] + (c**2 * dt**2) * (radial + angular)

            # if np.abs(u_next).max() > 1:
            #     print(np.abs(u_next).max())
    
    # Граничные условия при rad=R
    u_next[-1, :] = 0.0
    u_next[0, :] = 0.0

    # Граничные условия при phi=2*pi
    mean = (u_next[:, 0] + u_next[:, -1]) / 2
    u_next[:, 0] = mean
    u_next[:, -1] = mean
    

    # # Обработка центра (rad=0)
    # if Nr > 1:
    #     avg = np.mean(u_next[1, :])
    #     u_next[0, :] = avg  # Усреднение для rad=0
    
    # Обновление временных слоев
    u_prev[:, :] = u_curr
    u_curr[:, :] = u_next
    
    # Обновление графика
    ax.clear()
    # ax.set_title(f't = {frame * dt}')
    ax.set_title(f'текущее время = {round(t_curr, 4)}, dt = {round(dt, 4)}, dr = {round(dr, 4)}, dphi = {round(dphi, 4)}', fontsize=14, fontweight="bold")
    surface = ax.plot_surface(X, Y, u_curr, cmap='viridis')
    ax.set_zlim(-1, 1)

    ax.set_xlabel("X", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y", fontsize=14, fontweight="bold")
    ax.set_zlabel("U", fontsize=14, fontweight="bold")

    t_curr += dt

    return surface

# Создание анимации
ani = FuncAnimation(fig, update, frames=int(t_max/dt), interval=1, blit=False)
plt.show()