import numpy as np
import time

# ==========================================
# ПАРАМЕТРЫ МОДЕЛИ
# ==========================================
G = 1.0
epsilon = 0.05
N = 500 # Увеличили N, чтобы лучше почувствовать разницу алгоритмов
M_total = 1.0
m = M_total / N

# ==========================================
# 1. ГЕНЕРАЦИЯ НАЧАЛЬНЫХ УСЛОВИЙ
# ==========================================
def initialize_particles(N, R=1.0):
    np.random.seed(42)
    r = R * np.sqrt(np.random.uniform(0, 1, N))
    theta = np.random.uniform(0, 2*np.pi, N)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    positions = np.column_stack((x, y))
    
    v_mag = 0.3 * np.sqrt(G * M_total * r / (r**2 + epsilon**2)**0.75) 
    vx = -v_mag * np.sin(theta)
    vy = v_mag * np.cos(theta)
    velocities = np.column_stack((vx, vy))
    
    masses = np.full(N, m)
    return positions, velocities, masses

# ==========================================
# 2. РАСЧЁТ СИЛ (МЕТОД PARTICLE-PARTICLE, O(N^2))
# ==========================================
def compute_forces_pp(positions, masses):
    dx = positions[:, 0].reshape(1, N) - positions[:, 0].reshape(N, 1)
    dy = positions[:, 1].reshape(1, N) - positions[:, 1].reshape(N, 1)
    
    r2 = dx**2 + dy**2 + epsilon**2
    r = np.sqrt(r2)
    
    inv_r3 = 1.0 / (r2 * r)
    np.fill_diagonal(inv_r3, 0.0)
    
    ax = G * np.sum(masses.reshape(1, N) * inv_r3 * dx, axis=1)
    ay = G * np.sum(masses.reshape(1, N) * inv_r3 * dy, axis=1)
    
    return np.column_stack((ax, ay))

# ==========================================
# 3. ДЕРЕВО БАРНСА-ХАТА (QUADTREE)
# ==========================================
class QuadTreeNode:
    __slots__ = ['x', 'y', 'w', 'mass', 'cx', 'cy', 'children', 'body', 'is_external']
    
    def __init__(self, x, y, w):
        self.x = x # минимальная координата X квадрата
        self.y = y # минимальная координата Y квадрата
        self.w = w # ширина квадрата
        self.mass = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.children = None
        self.body = -1
        self.is_external = True
        
    def insert(self, idx, positions, masses, depth=0):
        px, py = positions[idx]
        pm = masses[idx]
        
        # Обновляем центр масс и общую массу узла
        new_mass = self.mass + pm
        self.cx = (self.cx * self.mass + px * pm) / new_mass
        self.cy = (self.cy * self.mass + py * pm) / new_mass
        self.mass = new_mass
        
        # Защита от бесконечной рекурсии, если частицы идеально совпали
        if depth > 50:
            return
        
        if self.is_external:
            if self.body == -1:
                self.body = idx
            else:
                self.subdivide()
                old_idx = self.body
                self.body = -1
                self._insert_into_child(old_idx, positions, masses, depth+1)
                self._insert_into_child(idx, positions, masses, depth+1)
        else:
            self._insert_into_child(idx, positions, masses, depth+1)
            
    def _insert_into_child(self, idx, positions, masses, depth):
        px, py = positions[idx]
        mid_x = self.x + self.w / 2
        mid_y = self.y + self.w / 2
        
        # Определяем квадрант (0: SW, 1: SE, 2: NW, 3: NE)
        child_idx_x = 1 if px >= mid_x else 0
        child_idx_y = 1 if py >= mid_y else 0
        child_index = child_idx_x + 2 * child_idx_y
        
        self.children[child_index].insert(idx, positions, masses, depth)
        
    def subdivide(self):
        self.is_external = False
        self.children = [None] * 4
        half_w = self.w / 2
        self.children[0] = QuadTreeNode(self.x, self.y, half_w)
        self.children[1] = QuadTreeNode(self.x + half_w, self.y, half_w)
        self.children[2] = QuadTreeNode(self.x, self.y + half_w, half_w)
        self.children[3] = QuadTreeNode(self.x + half_w, self.y + half_w, half_w)
        
    def compute_force(self, idx, positions, masses, theta, G, epsilon):
        if self.mass == 0:
            return 0.0, 0.0
            
        dx = self.cx - positions[idx, 0]
        dy = self.cy - positions[idx, 1]
        dist_sq = dx*dx + dy*dy
        
        if self.is_external:
            if self.body == idx:
                return 0.0, 0.0
        else:
            dist = np.sqrt(dist_sq)
            # Критерий Барнса-Хата: s / d < theta
            if dist > 0 and (self.w / dist < theta):
                pass # Узел достаточно далеко, считаем его одной частицей
            else:
                # Узел слишком близко, раскрываем его
                ax, ay = 0.0, 0.0
                for child in self.children:
                    if child is not None and child.mass > 0:
                        cax, cay = child.compute_force(idx, positions, masses, theta, G, epsilon)
                        ax += cax
                        ay += cay
                return ax, ay
                
        # Расчёт силы от центра масс (как от одной частицы)
        r2 = dist_sq + epsilon**2
        r = np.sqrt(r2)
        inv_r3 = 1.0 / (r2 * r)
        ax = G * self.mass * dx * inv_r3
        ay = G * self.mass * dy * inv_r3
        return ax, ay

def compute_forces_bh(positions, masses, theta):
    # Определяем границы для корневого узла
    min_x, min_y = np.min(positions, axis=0)
    max_x, max_y = np.max(positions, axis=0)
    
    width = max(max_x - min_x, max_y - min_y) * 1.01
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    root = QuadTreeNode(center_x - width/2, center_y - width/2, width)
    
    for i in range(len(positions)):
        root.insert(i, positions, masses)
        
    accels = np.zeros((len(positions), 2))
    for i in range(len(positions)):
        ax, ay = root.compute_force(i, positions, masses, theta, G, epsilon)
        accels[i, 0] = ax
        accels[i, 1] = ay
        
    return accels

# ==========================================
# 4. СРАВНЕНИЕ МЕТОДОВ
# ==========================================
def run_comparison():
    print(f"--- Запуск сравнения методов: N={N} ---")
    positions, velocities, masses = initialize_particles(N)
    
    # 1. Прямой метод (PP)
    t0 = time.time()
    accels_pp = compute_forces_pp(positions, masses)
    t_pp = time.time() - t0
    print(f"Время Particle-Particle (PP): {t_pp:.4f} сек")
    
    # 2. Метод Барнса-Хата (BH) с theta = 0.5
    theta = 0.5
    t0 = time.time()
    accels_bh = compute_forces_bh(positions, masses, theta)
    t_bh = time.time() - t0
    print(f"Время Barnes-Hut (BH, theta={theta}): {t_bh:.4f} сек")
    
    # 3. Оценка ошибки BH относительно PP
    diff = accels_bh - accels_pp
    error = np.linalg.norm(diff) / np.linalg.norm(accels_pp)
    
    print(f"Относительная ошибка ускорений (BH vs PP): {error:.6f} ({error*100:.4f}%)")
    if t_bh > 0:
        print(f"Ускорение (Speedup): {t_pp / t_bh:.2f}x")

if __name__ == "__main__":
    run_comparison()