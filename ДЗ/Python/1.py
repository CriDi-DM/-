import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from celluloid import Camera
from tqdm import tqdm
import os

class DLACrystal:
    """
    Модель Diffusion-Limited Aggregation (DLA) с оптимизацией и визуализацией.
    """
    def __init__(self, size=300, seed_pos=None):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int32)
        # Матрица времени присоединения для раскраски
        self.time_grid = np.zeros((size, size), dtype=np.int32)
        
        self.center = size // 2
        if seed_pos is None:
            self.grid[self.center, self.center] = 1
            self.time_grid[self.center, self.center] = 1
            
        self.particle_count = 1
        self.history = [] # Для хранения кадров
        
    def _get_neighbors(self, y, x):
        """Возвращает координаты соседей по фон Нейману (4 соседа)"""
        return [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
    
    def grow(self, n_particles=3000, save_every=50):
        """Запуск симуляции роста"""
        camera = None
        fig, ax = plt.subplots(figsize=(8, 8))
        camera = Camera(fig)
        
        # Настраиваем цветовую карту
        cmap = LinearSegmentedColormap.from_list("crystal", ["#000033", "#0066ff", "#00ffff", "#ffffff"])
        
        pbar = tqdm(total=n_particles, desc="Growing Crystal")
        frame_idx = 0
        
        for i in range(n_particles):
            # Запускаем частицу с границы круга, охватывающего текущий кластер
            # Оценка радиуса кластера для ускорения
            ys, xs = np.where(self.grid > 0)
            if len(ys) == 0: 
                current_radius = 1
            else:
                current_radius = int(np.max(np.sqrt((ys - self.center)**2 + (xs - self.center)**2))) + 5
            
            # Случайная позиция на окружности запуска
            angle = np.random.uniform(0, 2 * np.pi)
            start_r = min(current_radius + 10, self.size // 2 - 2)
            py = int(self.center + start_r * np.sin(angle))
            px = int(self.center + start_r * np.cos(angle))
            
            # Блуждание частицы
            stuck = False
            max_steps = self.size * self.size * 4 # Предохранитель
            steps = 0
            
            while not stuck and steps < max_steps:
                # Проверка выхода за границы (убиваем частицу)
                if py <= 1 or py >= self.size-2 or px <= 1 or px >= self.size-2:
                    break
                    
                # Проверка соседей
                neighbors = self._get_neighbors(py, px)
                has_neighbor = False
                for ny, nx in neighbors:
                    if self.grid[ny, nx] == 1:
                        has_neighbor = True
                        break
                
                if has_neighbor:
                    self.grid[py, px] = 1
                    self.time_grid[py, px] = self.particle_count + 1
                    self.particle_count += 1
                    stuck = True
                else:
                    # Случайный шаг
                    dy, dx = [(-1,0), (1,0), (0,-1), (0,1)][np.random.randint(4)]
                    py += dy
                    px += dx
                steps += 1
            
            pbar.update(1)
            
            # Сохранение кадра для анимации
            if i % save_every == 0 or i == n_particles - 1:
                # Маска для отображения только кристалла
                display = np.ma.masked_where(self.time_grid == 0, self.time_grid)
                ax.imshow(display, cmap=cmap, vmin=1, vmax=n_particles, origin='lower')
                ax.set_title(f"DLA Crystal | Particles: {self.particle_count}", fontsize=14)
                ax.axis('off')
                camera.snap()
                
        pbar.close()
        
        # Сохранение GIF
        print("Rendering animation...")
        animation = camera.animate(interval=100, blit=True)
        animation.save('dla_crystal_growth.gif', writer='pillow', fps=15)
        print("Saved: dla_crystal_growth.gif")
        
        # Финальное изображение высокого качества
        plt.figure(figsize=(10, 10))
        display = np.ma.masked_where(self.time_grid == 0, self.time_grid)
        plt.imshow(display, cmap=cmap, origin='lower')
        plt.colorbar(label='Attachment Time')
        plt.title('Final DLA Crystal Structure', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('dla_final.png', dpi=150)
        plt.show()

if __name__ == "__main__":
    model = DLACrystal(size=300)
    model.grow(n_particles=4000, save_every=80)