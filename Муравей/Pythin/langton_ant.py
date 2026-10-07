import numpy as np
import matplotlib.pyplot as plt

# ===== Параметры =====
GRID_SIZE = 500      # Размер поля (берём с запасом, чтобы муравей не ушёл за край)
STEPS = 12000        # Число шагов (магистраль возникает около 10400)

# ===== Инициализация =====
# Поле: 0 = белый, 1 = черный
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

# Направления движения: 0=Север(вверх), 1=Восток(вправо), 2=Юг(вниз), 3=Запад(влево)
# В матрице строки идут сверху вниз, поэтому "вверх" = row - 1, "вниз" = row + 1
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Стартовая позиция (центр)
start_row, start_col = GRID_SIZE // 2, GRID_SIZE // 2
ant_row, ant_col = start_row, start_col
ant_dir = 0  # Изначально смотрит на Север

# Массивы для сбора метрик
distances = []
black_cells = []

print(f"Запускаем симуляцию на {STEPS} шагов...")

# ===== Эволюция =====
for step in range(1, STEPS + 1):
    current_color = grid[ant_row, ant_col]
    
    # Правило Лэнгтона
    if current_color == 0:
        # На белой: поворот НАЛЕВО, клетка чернеет
        ant_dir = (ant_dir - 1) % 4
        grid[ant_row, ant_col] = 1
    else:
        # На черной: поворот НАПРАВО, клетка белеет
        ant_dir = (ant_dir + 1) % 4
        grid[ant_row, ant_col] = 0
        
    # Шаг вперед
    dr, dc = directions[ant_dir]
    ant_row += dr
    ant_col += dc
    
    # Сохраняем метрики
    # Манхэттенское расстояние от старта
    dist = abs(ant_row - start_row) + abs(ant_col - start_col)
    distances.append(dist)
    black_cells.append(np.sum(grid))
    
    # Промежуточный вывод
    if step % 2000 == 0:
        print(f"Шаг {step:5d} | Дистанция: {dist:4d} | Черных клеток: {np.sum(grid):5d}")

print(f"\nСимуляция завершена.")
print(f"Финальная позиция: ({ant_row}, {ant_col})")
print(f"Максимальное удаление от старта: {max(distances)}")

# ===== Визуализация результатов =====
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 1. Финальное состояние поля (должна быть видна магистраль)
axes[0].matshow(grid, cmap='Greys')
axes[0].set_title(f"Поле после {STEPS} шагов (Ищите магистраль!)")
axes[0].axis('off')

# 2. График расстояния от старта (главный график исследования)
axes[1].plot(distances, color='blue', alpha=0.8, linewidth=0.5)
axes[1].set_title("Манхэттенское расстояние муравья от старта")
axes[1].set_xlabel("Шаг (время)")
axes[1].set_ylabel("Расстояние")
axes[1].axvline(x=10000, color='r', linestyle='--', alpha=0.5, label='~10k шагов')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()