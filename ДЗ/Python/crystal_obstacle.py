import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

class DLALabyrinth:
    def __init__(self, size=400):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int32) # 0 - пусто, 1 - кристалл, 2 - препятствие
        self.time_grid = np.zeros((size, size), dtype=np.int32)
        self.visit_count = np.zeros((size, size), dtype=np.float32) # Для тепловой карты
        
        self.center = size // 2
        
        # --- СОЗДАНИЕ ПРЕПЯТСТВИЯ ---
        # Вариант 1: Кольцо с разрывом (чтобы кристалл рос внутри и снаружи)
        radius = size // 6
        Y, X = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((X - self.center)**2 + (Y - self.center)**2)
        
        # Создаем кольцо
        ring_mask = (dist_from_center >= radius) & (dist_from_center <= radius + 2)
        # Делаем разрыв в кольце (сектор)
        angle = np.arctan2(Y - self.center, X - self.center)
        gap_mask = (angle > -0.2) & (angle < 0.2) 
        
        obstacle_mask = ring_mask & ~gap_mask
        self.grid[obstacle_mask] = 2 # 2 - это препятствие
        
        # Затравка в центре
        self.grid[self.center, self.center] = 1
        self.time_grid[self.center, self.center] = 1
        self.particle_count = 1
        
    def grow(self, n_particles=6000):
        pbar = tqdm(total=n_particles, desc="Growing in Labyrinth")
        
        for i in range(n_particles):
            # Оценка текущего радиуса для запуска
            ys, xs = np.where(self.grid == 1)
            if len(ys) == 0: current_radius = 1
            else: current_radius = int(np.max(np.sqrt((ys - self.center)**2 + (xs - self.center)**2))) + 5
            
            # Запуск частицы
            launch_r = min(current_radius + 15, self.size // 2 - 5)
            angle = np.random.uniform(0, 2 * np.pi)
            py = int(self.center + launch_r * np.sin(angle))
            px = int(self.center + launch_r * np.cos(angle))
            
            stuck = False
            steps = 0
            
            while not stuck and steps < 10000:
                # Границы
                if py <= 1 or py >= self.size-2 or px <= 1 or px >= self.size-2:
                    break
                
                # Считаем посещения для тепловой карты
                self.visit_count[py, px] += 1
                
                # Проверка соседей (фон Нейман)
                # Частица прилипает, если сосед - кристалл (1) ИЛИ препятствие (2)
                neighbors_sum = (self.grid[py-1, px] > 0) + \
                                (self.grid[py+1, px] > 0) + \
                                (self.grid[py, px-1] > 0) + \
                                (self.grid[py, px+1] > 0)
                
                if neighbors_sum > 0:
                    # Если текущая клетка - препятствие, мы не можем сюда встать (мы уже в стене?)
                    # Но логика блуждания не должна пускать нас внутрь стены.
                    # Если мы тут, значит мы свободны. Прилипаем.
                    if self.grid[py, px] == 0:
                        self.grid[py, px] = 1
                        self.time_grid[py, px] = self.particle_count + 1
                        self.particle_count += 1
                        stuck = True
                    else:
                        # Если мы внутри препятствия (не должно случаться), отталкиваемся
                        dy, dx = [(-1,0), (1,0), (0,-1), (0,1)][np.random.randint(4)]
                        py += dy; px += dx
                else:
                    # Случайный шаг
                    dy, dx = [(-1,0), (1,0), (0,-1), (0,1)][np.random.randint(4)]
                    py += dy
                    px += dx
                    
                    # Если попали в препятствие, отскакиваем (не даем зайти внутрь)
                    if self.grid[py, px] == 2:
                        py -= dy
                        px -= dx
                
                steps += 1
            pbar.update(1)
        pbar.close()

def visualize_experiment(model):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Структура кристалла + Препятствие
    ax1 = axes[0]
    # Создаем RGB изображение вручную для контроля цветов
    # Фон - черный
    img = np.zeros((model.size, model.size, 3))
    
    # Кристалл - раскраска по времени (голубой -> белый)
    crystal_mask = model.grid == 1
    times = model.time_grid[crystal_mask]
    norm_times = (times - times.min()) / (times.max() - times.min() + 1e-5)
    
    # Цветовая схема: темно-синий -> голубой -> белый
    img[crystal_mask, 0] = norm_times * 0.5       # R
    img[crystal_mask, 1] = 0.5 + norm_times * 0.5 # G
    img[crystal_mask, 2] = 1.0                    # B
    
    # Препятствие - красный
    obstacle_mask = model.grid == 2
    img[obstacle_mask] = [0.8, 0.1, 0.1] # Dark Red
    
    ax1.imshow(img, origin='lower')
    ax1.set_title(f"Crystal Growth with Obstacles\nParticles: {model.particle_count}", fontsize=14)
    ax1.axis('off')
    
    # 2. Тепловая карта посещений (Density Map)
    ax2 = axes[1]
    # Логарифмическая шкала для лучшей видимости редких посещений
    log_visits = np.log1p(model.visit_count)
    
    # Маскируем нули и препятствия для красоты
    display_map = np.ma.masked_where(log_visits == 0, log_visits)
    
    im = ax2.imshow(display_map, cmap='magma', origin='lower')
    # Накладываем препятствие поверх
    ax2.imshow(np.ma.masked_where(~obstacle_mask, np.ones_like(model.grid)), 
               cmap='Reds', alpha=0.5, origin='lower')
    
    plt.colorbar(im, ax=ax2, label='Log(Particle Visits)')
    ax2.set_title("Particle Density Map (Diffusion Field)", fontsize=14)
    ax2.axis('off')
    
    plt.suptitle("Experiment: Crystal in a Labyrinth", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig('labyrinth_experiment.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Увеличим размер поля для красоты
    sim = DLALabyrinth(size=400)
    sim.grow(n_particles=6000)
    visualize_experiment(sim)