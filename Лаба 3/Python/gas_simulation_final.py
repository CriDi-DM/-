import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Параметры ---
N_PARTICLES = 150      # Чуть уменьшим для плавности рендеринга
BOX_SIZE = 100.0
TEMPERATURE = 1.5
N_FRAMES = 200         # Количество кадров в GIF
HIST_BINS = 25         # Количество столбиков

# --- Инициализация ---
positions = np.random.rand(N_PARTICLES, 2) * BOX_SIZE
velocities = np.random.normal(0, np.sqrt(TEMPERATURE), (N_PARTICLES, 2))
collected_speeds = []

# --- Настройка фигуры ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Левый график: Ящик
ax1.set_xlim(0, BOX_SIZE)
ax1.set_ylim(0, BOX_SIZE)
ax1.set_aspect('equal')
ax1.set_title('Динамика частиц (Метод Монте-Карло)')
scat = ax1.scatter([], [], s=50, cmap='jet', vmin=0, vmax=np.sqrt(TEMPERATURE)*3)
plt.colorbar(scat, ax=ax1, label='Скорость')

# Правый график: Гистограмма (настройки осей)
# Мы не рисуем тут ничего статичного, всё будет в update
max_speed_plot = np.sqrt(TEMPERATURE) * 4 

def update(frame):
    global positions, velocities, collected_speeds
    
    # 1. Физика
    positions += velocities
    
    # Отражение от стенок
    # X
    mask_x = (positions[:, 0] < 0) | (positions[:, 0] > BOX_SIZE)
    velocities[mask_x, 0] *= -1
    positions[:, 0] = np.clip(positions[:, 0], 0, BOX_SIZE)
    # Y
    mask_y = (positions[:, 1] < 0) | (positions[:, 1] > BOX_SIZE)
    velocities[mask_y, 1] *= -1
    positions[:, 1] = np.clip(positions[:, 1], 0, BOX_SIZE)
    
    # 2. Сбор статистики
    current_speeds = np.linalg.norm(velocities, axis=1)
    collected_speeds.extend(current_speeds)
    
    # 3. Обновление ЛЕВОГО графика (Частицы)
    scat.set_offsets(positions)
    scat.set_array(current_speeds) # Цвет зависит от скорости
    
    # 4. Обновление ПРАВОГО графика (Гистограмма)
    ax2.clear() # ПОЛНАЯ ОЧИСТКА - ключевой момент исправления
    
    # Рисуем гистограмму заново
    ax2.hist(collected_speeds, bins=HIST_BINS, density=True, 
             alpha=0.6, color='blue', range=(0, max_speed_plot), label='Симуляция')
    
    # Рисуем теорию поверх
    v_theory = np.linspace(0, max_speed_plot, 100)
    pdf_theory = (v_theory / TEMPERATURE) * np.exp(-v_theory**2 / (2 * TEMPERATURE))
    ax2.plot(v_theory, pdf_theory, 'r-', lw=2, label='Теория (Максвелл)')
    
    # Оформление
    ax2.set_title(f'Распределение скоростей (N={len(collected_speeds)})')
    ax2.set_xlabel('|v|')
    ax2.set_ylabel('Плотность вероятности')
    ax2.set_xlim(0, max_speed_plot)
    ax2.set_ylim(0, 0.8) # Фиксируем Y, чтобы график не "дышал"
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    return scat,

print("Рендеринг исправленной анимации...")
# blit=False ОБЯЗАТЕЛЕН при использовании ax.clear()
ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=50, blit=False) 

ani.save('final_gas_report.gif', writer='pillow', fps=20)
print("Готово! Проверьте файл 'final_gas_report.gif'.")

plt.show()