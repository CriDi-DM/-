import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from celluloid import Camera
from tqdm import tqdm
import os

# ==========================================
# 1. Модель DLA (Diffusion-Limited Aggregation)
# ==========================================
class DLACrystal:
    def __init__(self, size=300):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int32)
        self.time_grid = np.zeros((size, size), dtype=np.int32)
        self.center = size // 2
        self.grid[self.center, self.center] = 1
        self.time_grid[self.center, self.center] = 1
        self.particle_count = 1
        
    def grow(self, n_particles=3000):
        # Для ускорения не будем делать GIF внутри класса, вернем только данные
        radii = []
        
        pbar = tqdm(total=n_particles, desc="DLA Growth")
        for i in range(n_particles):
            ys, xs = np.where(self.grid > 0)
            current_radius = int(np.max(np.sqrt((ys - self.center)**2 + (xs - self.center)**2))) + 5
            
            angle = np.random.uniform(0, 2 * np.pi)
            start_r = min(current_radius + 10, self.size // 2 - 2)
            py = int(self.center + start_r * np.sin(angle))
            px = int(self.center + start_r * np.cos(angle))
            
            stuck = False
            steps = 0
            max_steps = self.size * self.size 
            
            while not stuck and steps < max_steps:
                if py <= 1 or py >= self.size-2 or px <= 1 or px >= self.size-2:
                    break
                
                # Проверка соседей (фон Нейман)
                if (self.grid[py-1, px] == 1 or self.grid[py+1, px] == 1 or 
                    self.grid[py, px-1] == 1 or self.grid[py, px+1] == 1):
                    self.grid[py, px] = 1
                    self.time_grid[py, px] = self.particle_count + 1
                    self.particle_count += 1
                    stuck = True
                else:
                    dy, dx = [(-1,0), (1,0), (0,-1), (0,1)][np.random.randint(4)]
                    py += dy
                    px += dx
                steps += 1
            
            if i % 50 == 0:
                radii.append(current_radius)
            pbar.update(1)
        pbar.close()
        return radii

# ==========================================
# 2. Модель Эдема (Eden Model)
# ==========================================
class EdenCrystal:
    def __init__(self, size=300):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int32)
        self.time_grid = np.zeros((size, size), dtype=np.int32)
        self.center = size // 2
        self.grid[self.center, self.center] = 1
        self.time_grid[self.center, self.center] = 1
        self.particle_count = 1
        # Список активных границ (клеток, куда можно расти)
        self.boundary = [(self.center-1, self.center), (self.center+1, self.center), 
                         (self.center, self.center-1), (self.center, self.center+1)]
        
    def grow(self, n_particles=3000):
        radii = []
        pbar = tqdm(total=n_particles, desc="Eden Growth")
        
        for i in range(n_particles):
            if not self.boundary:
                break
                
            # Выбираем случайную клетку из границы
            idx = np.random.randint(len(self.boundary))
            py, px = self.boundary.pop(idx)
            
            # Если клетка уже занята (могло случиться из-за дубликатов в списке), пропускаем
            if self.grid[py, px] == 1:
                continue
            
            # Занимаем клетку
            self.grid[py, px] = 1
            self.time_grid[py, px] = self.particle_count + 1
            self.particle_count += 1
            
            # Добавляем новых соседей в границу
            neighbors = [(py-1, px), (py+1, px), (py, px-1), (py, px+1)]
            for ny, nx in neighbors:
                if 0 <= ny < self.size and 0 <= nx < self.size and self.grid[ny, nx] == 0:
                    self.boundary.append((ny, nx))
            
            if i % 50 == 0:
                ys, xs = np.where(self.grid > 0)
                if len(ys) > 0:
                    r = np.max(np.sqrt((ys - self.center)**2 + (xs - self.center)**2))
                    radii.append(r)
            
            pbar.update(1)
        pbar.close()
        return radii

# ==========================================
# 3. Визуализация и Анализ
# ==========================================
def plot_comparison(dla_model, eden_model, dla_radii, eden_radii):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Цветовая карта
    cmap = LinearSegmentedColormap.from_list("crystal", ["#000033", "#0066ff", "#00ffff", "#ffffff"])
    
    # 1. DLA
    ax1 = axes[0]
    display_dla = np.ma.masked_where(dla_model.time_grid == 0, dla_model.time_grid)
    ax1.imshow(display_dla, cmap=cmap, origin='lower')
    ax1.set_title(f'DLA (Fractal ~1.7)\nParticles: {dla_model.particle_count}', fontsize=14)
    ax1.axis('off')
    
    # 2. Eden
    ax2 = axes[1]
    display_eden = np.ma.masked_where(eden_model.time_grid == 0, eden_model.time_grid)
    ax2.imshow(display_eden, cmap=cmap, origin='lower')
    ax2.set_title(f'Eden Model (Compact ~2.0)\nParticles: {eden_model.particle_count}', fontsize=14)
    ax2.axis('off')
    
    # 3. График Рост vs Время (Log-Log для фрактальной размерности)
    ax3 = axes[2]
    # Фильтруем нули и приводим к массивам
    dla_r = np.array(dla_radii)
    eden_r = np.array(eden_radii)
    dla_n = np.linspace(1, len(dla_r)*50, len(dla_r)) # Примерное число частиц
    eden_n = np.linspace(1, len(eden_r)*50, len(eden_r))
    
    ax3.loglog(dla_n, dla_r, label='DLA Radius', alpha=0.7)
    ax3.loglog(eden_n, eden_r, label='Eden Radius', alpha=0.7)
    ax3.set_xlabel('Number of Particles (log)')
    ax3.set_ylabel('Cluster Radius (log)')
    ax3.set_title('Growth Dynamics & Fractal Dimension')
    ax3.legend()
    ax3.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('comparison_final.png', dpi=150)
    plt.show()

def make_gif(model, filename, title_prefix):
    """Генерация GIF для конкретной модели"""
    fig, ax = plt.subplots(figsize=(6, 6))
    camera = Camera(fig)
    cmap = LinearSegmentedColormap.from_list("crystal", ["#000033", "#0066ff", "#00ffff", "#ffffff"])
    
    # Мы не можем легко восстановить промежуточные состояния без перезапуска,
    # поэтому для GIF лучше запустить отдельный быстрый прогон или использовать сохраненные кадры.
    # Для упрощения в этом скрипте мы сделаем GIF только для Эдема, перезапустив его с камерой.
    pass 

if __name__ == "__main__":
    SIZE = 300
    PARTICLES = 4000
    
    print("--- Запуск DLA ---")
    dla = DLACrystal(size=SIZE)
    radii_dla = dla.grow(n_particles=PARTICLES)
    
    print("\n--- Запуск Eden Model ---")
    eden = EdenCrystal(size=SIZE)
    radii_eden = eden.grow(n_particles=PARTICLES)
    
    print("\n--- Генерация отчетных графиков ---")
    plot_comparison(dla, eden, radii_dla, radii_eden)
    
    # Бонус: GIF для модели Эдема (чтобы показать разницу в динамике)
    print("\n--- Генерация GIF для модели Эдема ---")
    fig, ax = plt.subplots(figsize=(6, 6))
    camera = Camera(fig)
    cmap = LinearSegmentedColormap.from_list("crystal", ["#000033", "#0066ff", "#00ffff", "#ffffff"])
    
    # Перезапускаем Эдем для записи кадров
    eden_anim = EdenCrystal(size=SIZE)
    for i in tqdm(range(PARTICLES), desc="Rendering Eden GIF"):
        if not eden_anim.boundary: break
        idx = np.random.randint(len(eden_anim.boundary))
        py, px = eden_anim.boundary.pop(idx)
        if eden_anim.grid[py, px] == 1: continue
        eden_anim.grid[py, px] = 1
        eden_anim.time_grid[py, px] = eden_anim.particle_count + 1
        eden_anim.particle_count += 1
        neighbors = [(py-1, px), (py+1, px), (py, px-1), (py, px+1)]
        for ny, nx in neighbors:
            if 0 <= ny < SIZE and 0 <= nx < SIZE and eden_anim.grid[ny, nx] == 0:
                eden_anim.boundary.append((ny, nx))
        
        if i % 100 == 0:
            display = np.ma.masked_where(eden_anim.time_grid == 0, eden_anim.time_grid)
            ax.imshow(display, cmap=cmap, vmin=1, vmax=PARTICLES, origin='lower')
            ax.set_title(f"Eden Growth | Particles: {eden_anim.particle_count}")
            ax.axis('off')
            camera.snap()
            
    animation = camera.animate(interval=100, blit=True)
    animation.save('eden_crystal_growth.gif', writer='pillow', fps=15)
    print("Saved: eden_crystal_growth.gif")