import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# --- Параметры симуляции ---
N_PARTICLES = 100      # Количество частиц
BOX_SIZE = 100.0       # Размер ящика (L x L)
RADIUS = 1.5           # Радиус частицы (для визуализации)
N_STEPS = 300          # Количество шагов симуляции
DT = 1.0               # Шаг по времени (условный)
TEMPERATURE = 1.0      # "Температура" (масштаб скоростей)

# --- Инициализация ---
# Координаты: равномерное распределение в ящике
positions = np.random.rand(N_PARTICLES, 2) * BOX_SIZE

# Скорости: нормальное распределение (Гаусс), так как компоненты скоростей в газе распределены нормально
# Дисперсия связана с температурой
velocities = np.random.normal(0, np.sqrt(TEMPERATURE), (N_PARTICLES, 2))

# --- Функция шага симуляции (Метод Монте-Карло / Молекулярная динамика) ---
def simulation_step():
    global positions, velocities
    
    # 1. Движение частиц
    positions += velocities * DT
    
    # 2. Столкновение со стенками (Упругое отражение)
    # Если вылетели за 0 или BOX_SIZE, меняем знак скорости и "возвращаем" внутрь
    
    # Стенки по X
    mask_left = positions[:, 0] < 0
    mask_right = positions[:, 0] > BOX_SIZE
    velocities[mask_left | mask_right, 0] *= -1
    positions[mask_left, 0] = -positions[mask_left, 0] # Отражение
    positions[mask_right, 0] = 2 * BOX_SIZE - positions[mask_right, 0] # Отражение
    
    # Стенки по Y
    mask_bottom = positions[:, 1] < 0
    mask_top = positions[:, 1] > BOX_SIZE
    velocities[mask_bottom | mask_top, 1] *= -1
    positions[mask_bottom, 1] = -positions[mask_bottom, 1]
    positions[mask_top, 1] = 2 * BOX_SIZE - positions[mask_top, 1]

    # Примечание: В этой простой версии мы пока не считаем столкновения частица-частица,
    # чтобы код был простым и быстрым для визуализации. Для распределения Максвелла
    # достаточно рандомизации начальных условий и взаимодействия со стенками (термостат).

# --- Визуализация ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, BOX_SIZE)
ax.set_ylim(0, BOX_SIZE)
ax.set_aspect('equal')
ax.set_title(f'Идеальный газ в ящике (N={N_PARTICLES})')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Создаем объекты частиц (круги)
# Цвет зависит от модуля скорости (будет обновляться)
scat = ax.scatter(positions[:, 0], positions[:, 1], c=np.linalg.norm(velocities, axis=1), 
                  cmap='jet', s=50, alpha=0.7, edgecolors='k')

plt.colorbar(scat, label='Скорость |v|')

def update(frame):
    simulation_step()
    # Обновляем позиции
    scat.set_offsets(positions)
    # Обновляем цвета согласно новым скоростям
    speeds = np.linalg.norm(velocities, axis=1)
    scat.set_array(speeds)
    return scat,

# Создаем анимацию
ani = FuncAnimation(fig, update, frames=N_STEPS, interval=20, blit=True)

# Сохранение в GIF для отчета
print("Генерация GIF... это может занять пару секунд.")
ani.save('gas_box_animation.gif', writer='pillow', fps=30)
print("Анимация сохранена как 'gas_box_animation.gif'")

# Показываем график (если запускаете в Jupyter/Colab)
plt.show()