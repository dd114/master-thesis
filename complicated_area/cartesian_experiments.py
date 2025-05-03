import numpy as np
from scipy import special, ndimage


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import os
import multiprocessing


def start_calc(core, start, end, step, semi_axis1, semi_axis2):
    print(f"core = {core}")
    print("ID of process running worker: {}".format(os.getpid()))

    # Параметры задачи
    R = 1.1
    r = 1           # Радиус круга
    c = 1.0           # Скорость распространения волны
    h = 0.005          # Шаг пространственной сетки
    dt = 0.9 * h / (c * np.sqrt(2))  # Шаг времени (условие Куранта)
    Tmax = 5.0        # Время моделирования
    sigma = 0.1       # Параметр начального гауссова импульса

    t_curr = 0

    print(f"dt = {dt}")

    # a, b = 0.3, 0.1
    a, b = semi_axis1, semi_axis2
    # a, b = 1 / np.sqrt(2), 1 / np.sqrt(2)
    # max_a_b = max(a, b)
    N = int(2 * R / h) + 1

    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    X, Y = np.meshgrid(x, y)

    phi = np.pi / 4

    x_0, y_0 = 1 / np.sqrt(2), 1 / np.sqrt(2)

    new_X, new_Y = (X - x_0) * np.cos(phi) + (Y - y_0) * np.sin(phi), (X - x_0) * -np.sin(phi) + (Y - y_0) * np.cos(phi)

    ellipse_mask = (((new_X) / a)**2 + ((new_Y) / b)**2) <= 1

    circle_mask = (X**2 + Y**2) <= r**2

    x0r, y0r = r - 0.05, -R
    ar, br = R - r, 2 * R
    rectangle_mask = (X >= x0r) & (X <= x0r + ar) & (Y >= y0r) & (Y <= y0r + br)

    gallery_strip_mask = circle_mask & ((X**2 + Y**2) >= (0.9 * r) ** 2)

    # eroded = ndimage.binary_erosion(mask)
    # boundary = mask & ~eroded

    # xb = X[boundary]
    # yb = Y[boundary]

    xc, yc, = X[circle_mask], Y[circle_mask]
    xe, ye, = X[ellipse_mask], Y[ellipse_mask]
    xr, yr, = X[rectangle_mask], Y[rectangle_mask]
    xgs, ygs, = X[gallery_strip_mask], Y[gallery_strip_mask]

    # метрика
    def metric1(f_u):
        u_abs = np.abs(f_u)
        u_abs_max = u_abs.max()

        return ((u_abs_max - (10 * u_abs.mean())) / u_abs_max)

    # Начальные условия

    m = 1 # Bessel order
    n = 3 # number of root

    alpha_m = special.jn_zeros(m, n) / R 

    # Начальные условия (пример: гауссов импульс)
    def initial_state(x, y, w=60):
        # return special.jv(m, alpha_m[-1] * np.sqrt(x**2 + y**2)) * np.sin(1 * np.arctan2(y, x))
        # return 0.5 * np.exp( - ((x - 0.95) ** 2) / (2 * 0.001)) * np.sin(60 * y) # для круга
        return np.where(y < -0.5 * R, 0.5 * np.exp( - ((x - (x0r + (R - x0r) / 2)) ** 2) / (2 * 0.001)) * np.sin(w * y), 0) # для волновода
        # return np.where((y > -0.5 * R) & (y < 0.5 * R), 0.5 * np.exp( - ((x - (x0r + (R - x0r) / 2)) ** 2) / (2 * 0.001)) * np.sin(5 * y), 0) # для волновода
        # return 0.5 * np.exp(-((X**2 + Y**2 - 0.9) ** 2) / (sigma**2)) * np.cos(60 * np.arctan2(Y, X))
        # return 0.5 * np.exp(-((X**2 + Y**2 - 0.9) ** 2) / (sigma**2))
        return np.ones_like(X)

    # Начальные условия 
    def initial_speed(rad, phi):

        return np.zeros_like(rad)


    # for w in range(80, 90, 2):
    for w in range(start, end, step):

        mask = circle_mask  | rectangle_mask # Маска внутренних точек, объединение
        # mask = ((~ellipse_mask) & circle_mask) | rectangle_mask # Маска внутренних точек, вычитание
        # mask = circle_mask # Маска внутренних точек
        # mask = ellipse_mask # Маска внутренних точек
        # mask = rectangle_mask # Маска внутренних точек

        # u_prev = initial_state(X, Y) * circle_mask
        u_prev = initial_state(X, Y, w) * rectangle_mask

        speed = initial_speed(X, Y)
        u_curr: np.array = u_prev[:, :] + speed * dt



        # Начальное состояние
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_title(f'Начальное состояние', fontsize=14, fontweight="bold")
        # surface = ax.plot_surface(X, Y, u_curr, cmap='viridis', rstride=1, cstride=1)
        # ax.scatter(xc, yc, color='r', linewidth=4)
        # ax.scatter(xe, ye, color='g', linewidth=4)
        # ax.scatter(xr, yr, color='b', linewidth=4)
        # ax.scatter(xgs, ygs, color='y', linewidth=4)
        # ax.set_zlim(-1, 1)

        # ax.set_xlabel("X", fontsize=14, fontweight="bold")
        # ax.set_ylabel("Y", fontsize=14, fontweight="bold")
        # ax.set_zlabel("U", fontsize=14, fontweight="bold")

        # plt.show()

        number_steps = int(20 // dt)

        time2file = np.zeros(number_steps)
        gallery_scores2file = np.zeros(number_steps)



        # # Настройка анимации
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(X, Y, u_curr, cmap='viridis', rstride=1, cstride=1)
        # ax.set_zlim(-1, 1)

        my_iter = 0
        t_curr = 0

        laplacian = np.zeros_like(u_curr)
        while True:

            
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
            
            u_prev, u_curr = u_curr, u_next
            
            # # Обновление графика
            # ax.clear()
            # ax.set_title(f'текущее время = {round(t_curr, 4)}, dt = {round(dt, 4)}, h = {round(h, 4)}', fontsize=14, fontweight="bold")
            # # surf = ax.plot_surface(X, Y, u_curr, cmap='viridis', rstride=5, cstride=5)
            # ax.plot(xc, yc, color='r', linewidth=4)
            # ax.scatter(xe, ye, color='g', linewidth=4)
            # ax.plot(xr, yr, color='b', linewidth=4)
            # ax.scatter(xgs, ygs, color='y', linewidth=4)


            # ax.set_zlim(-1, 1)

            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')

            t_curr += dt

            gallery_score = np.sum(u_curr[gallery_strip_mask] ** 2) / np.sum(u_curr[circle_mask] ** 2)

            print("w =", w)
            print("my_iter =", my_iter)
            print("t_curr =", t_curr)
            print("abs max =", np.abs(u_curr[circle_mask]).max())
            print("abs mean =", np.abs(u_curr[circle_mask]).mean())
            print("energy =", np.sum(u_curr[circle_mask] ** 2))
            print("energy tube =", np.sum(u_curr[rectangle_mask] ** 2))
            print("gallery metric =", gallery_score)
            # print("metric1 =", metric1(u_curr[circle_mask]))

            if my_iter < number_steps:
                time2file[my_iter] = t_curr
                gallery_scores2file[my_iter] = gallery_score
            else:
                print("Files are ready")
                
                np.save(os.path.join('results', f'gallery_score_w{w}_a{a}_b{b}'), gallery_scores2file)
                np.save(os.path.join('results', 'time_series'), time2file)
                break


            if t_curr > 3.0:
                mask = circle_mask

            my_iter +=1 


        # # Создание анимации
        # ani = FuncAnimation(fig, update, frames=int(Tmax/dt), interval=50, blit=False)
        # plt.show()

if __name__ == '__main__':
    processes = []
    params = []

    # from itertools import product
    # comb = list(product([10, 50, 90], [0.05, 0.15, 0.3], [0.05, 0.25, 0.5]))
    # print(comb)

    point_start = 70
    segment_end = 2

    number_of_cores = 15

    for core in range(0, number_of_cores):
       params.append((core, point_start + core * segment_end, point_start + (core + 1) * segment_end, 2))

    # creating processes
    for param in params:
        processes.append(multiprocessing.Process(target=start_calc, args=param))

    # starting processes
    for core, p in enumerate(processes):
        p.start()

        # starting processes
    for core, p in enumerate(processes):
        p.join()

    # p1.start()
    # p2.start()
    # start_calc(1, 2)