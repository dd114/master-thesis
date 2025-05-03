import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Параметры задачи
R = 1.0          # Радиус круга
c = 1.0          # Скорость волны
Nr = 50          # Узлов по радиусу
Nθ = 50          # Узлов по углу
dt = 0.001       # Шаг времени
T = 2.0          # Общее время
n_steps = int(T / dt)
# plot_every = 10  # Частота обновления графика

dr = R / (Nr - 1)
dθ = 2 * np.pi / Nθ

# Сетка в полярных координатах
r = np.linspace(0, R, Nr)
θ = np.linspace(0, 2 * np.pi, Nθ + 1)[:-1]
R_grid, Θ_grid = np.meshgrid(r, θ, indexing='ij')

# Декартовы координаты для визуализации
X = R_grid * np.cos(Θ_grid)
Y = R_grid * np.sin(Θ_grid)

# Инициализация начальных условий
u_prev = np.zeros((Nr, Nθ))
u_curr = np.zeros((Nr, Nθ))

sigma = 0.2
k = 3  # Волновое число по θ
for i in range(Nr):
    for j in range(Nθ):
        u_prev[i, j] = np.exp(-(r[i]**2)/(2*sigma**2)) * np.cos(k * θ[j])

# Первый шаг (используем начальную скорость ut=0)
laplacian = np.zeros_like(u_prev)
for i in range(Nr):
    for j in range(Nθ):
        if i == 0:
            if Nr > 1:
                avg_u1 = np.mean(u_prev[1, :])
                laplacian[i, j] = 4 * (avg_u1 - u_prev[i, j]) / dr**2
        else:
            if i == Nr-1:
                u_r_plus = 0
            else:
                u_r_plus = u_prev[i+1, j]
            u_r_minus = u_prev[i-1, j]
            term1 = (i + 0.5) * (u_r_plus - u_prev[i, j])
            term2 = (i - 0.5) * (u_prev[i, j] - u_r_minus)
            laplacian_r = (term1 - term2) / (i * dr**2)
            
            j_plus = (j + 1) % Nθ
            j_minus = (j - 1) % Nθ
            laplacian_θ = (u_prev[i, j_plus] - 2*u_prev[i, j] + u_prev[i, j_minus]) / ((i*dr * dθ)**2)
            laplacian[i, j] = laplacian_r + laplacian_θ

u_curr = u_prev + 0.5 * (c*dt)**2 * laplacian

# Подготовка анимации
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, u_prev, cmap='viridis', rstride=1, cstride=1)
ax.set_zlim(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Колебания струны в круге')

def update(frame):
    global u_prev, u_curr
    
    u_next = np.zeros_like(u_prev)
    for i in range(Nr):
        for j in range(Nθ):
            if i == 0:
                if Nr > 1:
                    avg_u1 = np.mean(u_curr[1, :])
                    laplacian_ij = 4 * (avg_u1 - u_curr[i, j]) / dr**2
                else:
                    laplacian_ij = 0
            else:
                if i == Nr-1:
                    u_r_plus = 0
                else:
                    u_r_plus = u_curr[i+1, j]
                u_r_minus = u_curr[i-1, j]
                term1 = (i + 0.5) * (u_r_plus - u_curr[i, j])
                term2 = (i - 0.5) * (u_curr[i, j] - u_r_minus)
                laplacian_r = (term1 - term2) / (i * dr**2)
                
                j_plus = (j + 1) % Nθ
                j_minus = (j - 1) % Nθ
                laplacian_θ = (u_curr[i, j_plus] - 2*u_curr[i, j] + u_curr[i, j_minus]) / ((i*dr * dθ)**2)
                laplacian_ij = laplacian_r + laplacian_θ
            
            if i == Nr-1:
                u_next[i, j] = 0
            else:
                u_next[i, j] = 2*u_curr[i, j] - u_prev[i, j] + (c*dt)**2 * laplacian_ij
    
    u_prev, u_curr = u_curr, u_next
    
    ax.clear()
    ax.set_zlim(-1, 1)
    surface = ax.plot_surface(X, Y, u_curr, cmap='viridis')

    return surface,

ani = FuncAnimation(fig, update, frames=n_steps, interval=50, blit=False)
plt.show()