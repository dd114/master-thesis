import numpy as np
from scipy import special

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
    x = np.linspace(0, 10, 500)
    alpha_values = [0, 1, 2]

    # plt.figure(figsize=(10, 6))
    # for alpha in alpha_values:
    #     plt.plot(x, special.jv(alpha, x), label=f'J_{alpha}(x)')

    # plt.xlabel('x')
    # plt.ylabel('J_alpha(x)')
    # plt.title('Bessel Functions of the First Kind')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # Параметры
    # r = 0.0001
    # r = 0.035
    r = 0
    # r = 0.8
    R = 1.0           # Радиус круга
    c = 1.0           # Скорость волны
    # Nr = 200           # Число узлов по rad
    # Nphi = 300         # Число узлов по φ
    Nr = 150           # Число узлов по rad
    Nphi = 250         # Число узлов по φ
    dr = R / (Nr - 1) # Шаг по rad
    dphi = 2 * np.pi / Nphi  # Шаг по φ
    dt = 0.05  # Шаг по времени (условие Куранта); np.sqrt(3)?
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

    m = 1 # Bessel order
    n = 2 # number of root

    alpha_m = special.jn_zeros(m, n) / R 

    # Начальные условия (пример: гауссов импульс)
    def initial_state(rad, phi):
        return special.jv(m, alpha_m[-1] * Rad_grid) * np.cos(m * Phi_grid)
        return 0.5 * np.exp( - ((rad - 0.9) ** 2) / (2 * 0.001)) * np.cos(60 * phi) # подходит


        return np.zeros_like(rad)

    # Начальные условия (пример: гауссов импульс)
    def initial_speed(rad, phi):
        # return -1 * np.exp( - ((rad - 0.9) ** 2) / (2 * 0.001))
        # return 5 * np.exp( - ((rad - 0.9) ** 2) / (2 * 0.001)) * np.cos(30 * phi)
        # return 0.5 * np.exp( - ((rad - 0.9) ** 2) / (2 * 0.001)) * np.exp( - ((phi - np.pi) ** 2) / (2 * 0.001))
        
        # return np.exp(-50*(rad - 0.3)**2) * 0.5 * np.sin(phi)
        # return -np.exp(-50*(rad - 0.3)**2)

        return np.zeros_like(rad)

    u_prev = initial_state(Rad_grid, Phi_grid)

    # Граничные условия при rad=R и rad=r
    u_prev[0, :] = 0.0
    u_prev[-1, :] = 0.0

    speed = initial_speed(Rad_grid, Phi_grid)

    u_curr[:, :] = u_prev[:, :] + speed * dt

    X = Rad_grid * np.cos(Phi_grid)
    Y = Rad_grid * np.sin(Phi_grid)


    # Начальное состояние
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Начальное состояние', fontsize=14, fontweight="bold")
    surface = ax.plot_surface(X, Y, u_prev, cmap='viridis')
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

    # Подготовка для анимации
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # surface = ax.plot_surface(X, Y, u_curr, cmap='viridis')
    ax.set_zlim(-1, 1)


    # Функция обновления кадра
    def update(frame):
        global t_curr
        
        u_curr = np.cos(c * alpha_m[-1] * t_curr) * u_prev

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