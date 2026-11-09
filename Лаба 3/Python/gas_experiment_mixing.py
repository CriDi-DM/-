import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Параметры ---
N_PARTICLES_HALF = 100   # Частиц в каждой половине (всего 200)
BOX_SIZE = 100.0
T_HOT = 3.0              # Температура "горячего" газа (высокая скорость)
T_COLD = 0.5             # Температура "холодного" газа (низкая скорость)
N_FRAMES = 300           # Длительность
MIXING_FRAME = 100       # Кадр, на котором убираем перегородку

# --- Инициализация ---
# Левая часть (Горячая)
pos_hot = np.random.rand(N_PARTICLES_HALF, 2) * BOX_SIZE
pos_hot[:, 0] *= 0.5  # Сжимаем в левую половину (0..50)
vel_hot = np.random.normal(0, np.sqrt(T_HOT), (N_PARTICLES_HALF, 2))

# Правая часть (Холодная)
pos_cold = np.random.rand(N_PARTICLES_HALF, 2) * BOX_SIZE
pos_cold[:, 0] = pos_cold[:, 0] * 0.5 + 50 # Сдвигаем в правую половину (50..100)
vel_cold = np.random.normal(0, np.sqrt(T_COLD), (N_PARTICLES_HALF, 2))

# Объединяем
positions = np.vstack([pos_hot, pos_cold])
velocities = np.vstack([vel_hot, vel_cold])

# Метки для цветов (0 = был горячим, 1 = был холодным) - чтобы跟踪 происхождение
# Но для физики нам важна текущая скорость. 
# Давайте раскрашивать по ТЕКУЩЕЙ скорости, чтобы видеть выравнивание температуры.
# vmin/vmax зафиксируем по T_HOT, чтобы видеть изменения.

collected_speeds_hot = []
collected_speeds_cold = []

# --- Фигура ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Левый: Визуализация
ax1.set_xlim(0, BOX_SIZE)
ax1.set_ylim(0, BOX_SIZE)
ax1.set_aspect('equal')
ax1.set_title('Смешивание газов (Опыт Джоуля)')
scat = ax1.scatter([], [], s=50, cmap='coolwarm', vmin=0, vmax=np.sqrt(T_HOT)*2.5)
cbar = plt.colorbar(scat, ax=ax1)
cbar.set_label('Скорость (Температура)')

# Перегородка (линия)
partition_line = ax1.axvline(x=50, color='black', lw=3, linestyle='--')

# Правый: Графики
ax2.set_title('Эволюция распределения скоростей')
ax2.set_xlabel('|v|')
ax2.set_ylabel('Плотность вероятности')
ax2.set_xlim(0, np.sqrt(T_HOT) * 3)
ax2.set_ylim(0, 1.0)
ax2.grid(True, alpha=0.3)

# Теоретические кривые для справки
v_th = np.linspace(0, np.sqrt(T_HOT)*3, 100)
ax2.plot(v_th, (v_th/T_HOT)*np.exp(-v_th**2/(2*T_HOT)), 'r--', alpha=0.3, label='Нач. Горячий')
ax2.plot(v_th, (v_th/T_COLD)*np.exp(-v_th**2/(2*T_COLD)), 'b--', alpha=0.3, label='Нач. Холодный')

def update(frame):
    global positions, velocities
    
    # 1. Логика перегородки
    if frame < MIXING_FRAME:
        # Стенка посередине работает
        # Если частица слева летит вправо за 50
        mask_cross_right = (positions[:, 0] > 50) & (positions[:, 0] < 50 + 2) # Упрощенно
        # Просто жестко не пускаем за 50 для левой половины и за 50 для правой
        
        # Индексы
        left_part = positions[:, 0] < 50
        right_part = positions[:, 0] >= 50
        
        # Отражение от внутренней стенки
        # Для тех, кто был слева, но вылетел направо
        crossed_to_right = (~left_part) & (np.arange(len(positions)) < N_PARTICLES_HALF)
        velocities[crossed_to_right, 0] *= -1
        positions[crossed_to_right, 0] = 50 - (positions[crossed_to_right, 0] - 50)
        
        # Для тех, кто был справа, но вылетел налево
        crossed_to_left = (left_part) & (np.arange(len(positions)) >= N_PARTICLES_HALF)
        velocities[crossed_to_left, 0] *= -1
        positions[crossed_to_left, 0] = 50 + (50 - positions[crossed_to_left, 0])
        
        partition_line.set_visible(True)
        ax1.set_title(f'Изоляция систем (t < {MIXING_FRAME})')
    else:
        partition_line.set_visible(False)
        ax1.set_title(f'Свободное смешивание (t >= {MIXING_FRAME})')

    # 2. Обычное движение и внешние стенки
    positions += velocities
    
    # Внешние стенки X
    mask_x = (positions[:, 0] < 0) | (positions[:, 0] > BOX_SIZE)
    velocities[mask_x, 0] *= -1
    positions[:, 0] = np.clip(positions[:, 0], 0, BOX_SIZE)
    
    # Внешние стенки Y
    mask_y = (positions[:, 1] < 0) | (positions[:, 1] > BOX_SIZE)
    velocities[mask_y, 1] *= -1
    positions[:, 1] = np.clip(positions[:, 1], 0, BOX_SIZE)
    
    # 3. Обновление визуализации
    speeds = np.linalg.norm(velocities, axis=1)
    scat.set_offsets(positions)
    scat.set_array(speeds)
    
    # 4. Обновление статистики (разделяем на "бывших горячих" и "бывших холодных")
    # Первые N_PARTICLES_HALF были горячими
    speeds_hot = speeds[:N_PARTICLES_HALF]
    speeds_cold = speeds[N_PARTICLES_HALF:]
    
    collected_speeds_hot.extend(speeds_hot)
    collected_speeds_cold.extend(speeds_cold)
    
    # Рисуем гистограммы (только последние N данных, чтобы видеть ТЕКУЩЕЕ состояние, а не накопленное)
    # Это важно! Нам нужна "скользящая" статистика, чтобы видеть изменение во времени.
    window_size = 2000 # Берем последние 2000 замеров
    recent_hot = collected_speeds_hot[-window_size:]
    recent_cold = collected_speeds_cold[-window_size:]
    
    ax2.clear()
    # Перерисовываем оси и теорию
    ax2.set_title(f'Распределение скоростей (Кадр {frame})')
    ax2.set_xlabel('|v|')
    ax2.set_ylabel('Плотность вероятности')
    ax2.set_xlim(0, np.sqrt(T_HOT) * 3)
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, alpha=0.3)
    
    # Теория (статичная)
    ax2.plot(v_th, (v_th/T_HOT)*np.exp(-v_th**2/(2*T_HOT)), 'k:', alpha=0.5, label='T_hot init')
    ax2.plot(v_th, (v_th/T_COLD)*np.exp(-v_th**2/(2*T_COLD)), 'k:', alpha=0.5, label='T_cold init')
    
    if len(recent_hot) > 100:
        ax2.hist(recent_hot, bins=20, density=True, alpha=0.5, color='red', label='Hot Gas (Current)')
    if len(recent_cold) > 100:
        ax2.hist(recent_cold, bins=20, density=True, alpha=0.5, color='blue', label='Cold Gas (Current)')
        
    ax2.legend(loc='upper right')

    return scat, partition_line

print("Запуск эксперимента по смешиванию газов...")
ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=50, blit=False)
ani.save('gas_mixing_experiment.gif', writer='pillow', fps=20)
print("Готово! Файл: gas_mixing_experiment.gif")

plt.show()