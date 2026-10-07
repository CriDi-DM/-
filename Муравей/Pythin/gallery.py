import numpy as np
import matplotlib.pyplot as plt

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN = {'L': 3, 'R': 1, 'U': 2, 'F': 0}

def get_crop(rule, steps, grid_size=1500):
    """Симулирует муравья и возвращает обрезанное по границам поле"""
    q = len(rule)
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    r, c = grid_size // 2, grid_size // 2
    d = 0
    min_r, max_r, min_c, max_c = r, r, c, c
    
    for _ in range(steps):
        s = grid[r, c]
        d = (d + TURN[rule[s]]) % 4
        grid[r, c] = (s + 1) % q
        dr, dc = DIRS[d]
        r += dr; c += dc
        if r < min_r: min_r = r
        if r > max_r: max_r = r
        if c < min_c: min_c = c
        if c > max_c: max_c = c
        
    pad = 15
    return grid[max(0, min_r-pad):min(grid_size, max_r+pad), 
                max(0, min_c-pad):min(grid_size, max_c+pad)]

# Подборка правил на основе ваших данных
gallery = [
    ('LR',   12000, 'Класс 3 (q=2)\nКлассическая магистраль', 'Greys'),
    ('RR',   200,   'Класс 2 (q=2)\nПериодический цикл',    'Greys'),
    ('RLLR', 20000, 'Класс 3 (q=4)\nСложная магистраль',    'viridis'),
    ('RRUL', 15000, 'Класс 3 (q=4)\nДолгий хаос $\to$ магистраль', 'viridis'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, (rule, steps, title, cmap) in enumerate(gallery):
    print(f"Рисуем {rule} ({steps} шагов)...")
    crop = get_crop(rule, steps)
    axes[i].imshow(crop, cmap=cmap, interpolation='nearest')
    axes[i].set_title(f"Правило: {rule}\n{title}", fontsize=14, fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('ant_gallery.png', dpi=150)
print("Галерея успешно сохранена в ant_gallery.png!")
plt.show()