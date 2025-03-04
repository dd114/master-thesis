import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Параметры задачи
R = 1.0         # Радиус круга
c = 1.0         # Скорость волны
Nr = 30         # Шагов по радиусу
Nθ = 40         # Шагов по углу
Nt = 100        # Шагов по времени
Δt = 0.005      # Шаг времени

dr = R / Nr
dθ = 2 * np.pi / Nθ

# Сетка в полярных координатах
r = np.linspace(0, R, Nr+1)
θ = np.linspace(0, 2*np.pi, Nθ+1)[:-1]
R_grid, Θ_grid = np.meshgrid(r, θ, indexing='ij')

# Декартовы координаты для визуализации
X = R_grid * np.cos(Θ_grid)
Y = R_grid * np.sin(Θ_grid)

# Начальные условия (гауссов горб)
u_n = np.zeros((Nr+1, Nθ))
u_nm1 = np.zeros((Nr+1, Nθ))
for i in range(Nr+1):
    u_n[i, :] = np.exp(- (r[i]**2) / 0.1)

# Построение матрицы системы
size = (Nr+1) * Nθ
A = sp.lil_matrix((size, size), dtype=np.float64)

for i in range(Nr+1):
    for j in range(Nθ):
        row = i * Nθ + j
        ri = r[i]
        
        if i == Nr:  # Граница r = R
            A[row, row] = 1.0
            continue
        
        if i == 0:  # Центр (особенность)
            A[row, row] = 1.0 + 4 * c**2 * Δt**2 / dr**2
            A[row, 1*Nθ + j] = -4 * c**2 * Δt**2 / dr**2
            continue
        
        # Коэффициенты
        A[row, row] = 1 + 2 * c**2 * Δt**2 / dr**2 + 2 * c**2 * Δt**2 / (ri**2 * dθ**2)
        
        # Соседи по r
        if i+1 <= Nr:
            A[row, (i+1)*Nθ + j] = -c**2 * Δt**2 / dr**2 + c**2 * Δt**2 / (2 * ri * dr)
        if i-1 >= 0:
            A[row, (i-1)*Nθ + j] = -c**2 * Δt**2 / dr**2 - c**2 * Δt**2 / (2 * ri * dr)
        
        # Соседи по θ
        j_plus = (j + 1) % Nθ
        j_minus = (j - 1) % Nθ
        A[row, i*Nθ + j_plus] = -c**2 * Δt**2 / (ri**2 * dθ**2)
        A[row, i*Nθ + j_minus] = -c**2 * Δt**2 / (ri**2 * dθ**2)

A = A.tocsr()

# Анимация
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-1, 1)
surf = ax.plot_surface(X, Y, u_n, cmap='viridis')

def animate(n):
    global u_n, u_nm1
    b = 2 * u_n.ravel() - u_nm1.ravel()
    u_new = spla.spsolve(A, b).reshape((Nr+1, Nθ))
    u_nm1[:, :] = u_n
    u_n[:, :] = u_new
    
    ax.clear()
    ax.set_zlim(-1, 1)
    surf = ax.plot_surface(X, Y, u_n, cmap='viridis')
    return surf,

ani = animation.FuncAnimation(fig, animate, frames=Nt, blit=False)
plt.show()