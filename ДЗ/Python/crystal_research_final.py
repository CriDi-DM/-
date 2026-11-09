import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

class EdenCrystalResearch:
    def __init__(self, size=300, neighborhood='neumann'):
        """
        neighborhood: 'neumann' (4 соседа) или 'moore' (8 соседей)
        """
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int32)
        self.time_grid = np.zeros((size, size), dtype=np.int32)
        self.center = size // 2
        self.grid[self.center, self.center] = 1
        self.time_grid[self.center, self.center] = 1
        self.particle_count = 1
        self.neighborhood = neighborhood
        
        # Инициализация границы
        self.boundary = []
        self._add_neighbors_to_boundary(self.center, self.center)
        
    def _get_neighbors_coords(self, y, x):
        if self.neighborhood == 'neumann':
            return [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
        elif self.neighborhood == 'moore':
            return [(y-1, x-1), (y-1, x), (y-1, x+1),
                    (y, x-1),           (y, x+1),
                    (y+1, x-1), (y+1, x), (y+1, x+1)]
        return []

    def _add_neighbors_to_boundary(self, y, x):
        neighbors = self._get_neighbors_coords(y, x)
        for ny, nx in neighbors:
            if 0 <= ny < self.size and 0 <= nx < self.size and self.grid[ny, nx] == 0:
                self.boundary.append((ny, nx))
                
    def grow(self, n_particles=4000):
        pbar = tqdm(total=n_particles, desc=f"Eden ({self.neighborhood})")
        for _ in range(n_particles):
            if not self.boundary:
                break
            
            # Оптимизация: удаляем дубликаты из границы периодически или проверяем при выборке
            # Для простоты и скорости просто пробуем выбрать, если занято - пропускаем
            # Но лучше очистить список, если он слишком большой (здесь оставим простую логику)
            
            idx = np.random.randint(len(self.boundary))
            py, px = self.boundary.pop(idx)
            
            if self.grid[py, px] == 1:
                continue
            
            self.grid[py, px] = 1
            self.time_grid[py, px] = self.particle_count + 1
            self.particle_count += 1
            
            self._add_neighbors_to_boundary(py, px)
            pbar.update(1)
        pbar.close()

def plot_neighborhood_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    cmap = LinearSegmentedColormap.from_list("crystal", ["#000033", "#0066ff", "#00ffff", "#ffffff"])
    
    # 1. Фон Нейман (4 соседа)
    print("Запуск модели с окрестностью фон Неймана (4 соседа)...")
    model_vn = EdenCrystalResearch(size=300, neighborhood='neumann')
    model_vn.grow(n_particles=5000)
    
    ax1 = axes[0]
    display_vn = np.ma.masked_where(model_vn.time_grid == 0, model_vn.time_grid)
    ax1.imshow(display_vn, cmap=cmap, origin='lower')
    ax1.set_title(f"Von Neumann Neighborhood (4 neighbors)\nParticles: {model_vn.particle_count}", fontsize=14)
    ax1.axis('off')
    
    # 2. Мур (8 соседей)
    print("Запуск модели с окрестностью Мура (8 соседей)...")
    model_moore = EdenCrystalResearch(size=300, neighborhood='moore')
    model_moore.grow(n_particles=5000)
    
    ax2 = axes[1]
    display_moore = np.ma.masked_where(model_moore.time_grid == 0, model_moore.time_grid)
    ax2.imshow(display_moore, cmap=cmap, origin='lower')
    ax2.set_title(f"Moore Neighborhood (8 neighbors)\nParticles: {model_moore.particle_count}", fontsize=14)
    ax2.axis('off')
    
    plt.suptitle("Influence of Neighborhood Type on Crystal Morphology (Eden Model)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('neighborhood_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_neighborhood_comparison()