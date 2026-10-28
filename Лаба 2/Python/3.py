import numpy as np
import time
import matplotlib.pyplot as plt

# ==========================================
# ПАРАМЕТРЫ И ФУНКЦИИ (Из предыдущих шагов)
# ==========================================
G = 1.0
epsilon = 0.05
M_total = 1.0

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
    masses = np.full(N, M_total / N)
    return positions, velocities, masses

def compute_forces_pp(positions, masses):
    N = len(positions)
    dx = positions[:, 0].reshape(1, N) - positions[:, 0].reshape(N, 1)
    dy = positions[:, 1].reshape(1, N) - positions[:, 1].reshape(N, 1)
    r2 = dx**2 + dy**2 + epsilon**2
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    np.fill_diagonal(inv_r3, 0.0)
    ax = G * np.sum(masses.reshape(1, N) * inv_r3 * dx, axis=1)
    ay = G * np.sum(masses.reshape(1, N) * inv_r3 * dy, axis=1)
    return np.column_stack((ax, ay))

class QuadTreeNode:
    __slots__ = ['x', 'y', 'w', 'mass', 'cx', 'cy', 'children', 'body', 'is_external']
    def __init__(self, x, y, w):
        self.x, self.y, self.w = x, y, w
        self.mass, self.cx, self.cy = 0.0, 0.0, 0.0
        self.children, self.body, self.is_external = None, -1, True
        
    def insert(self, idx, positions, masses, depth=0):
        px, py = positions[idx]
        pm = masses[idx]
        new_mass = self.mass + pm
        self.cx = (self.cx * self.mass + px * pm) / new_mass
        self.cy = (self.cy * self.mass + py * pm) / new_mass
        self.mass = new_mass
        if depth > 50: return
        if self.is_external:
            if self.body == -1: self.body = idx
            else:
                self.subdivide()
                old_idx = self.body
                self.body = -1
                self._insert_into_child(old_idx, positions, masses, depth+1)
                self._insert_into_child(idx, positions, masses, depth+1)
        else: self._insert_into_child(idx, positions, masses, depth+1)
            
    def _insert_into_child(self, idx, positions, masses, depth):
        px, py = positions[idx]
        mid_x, mid_y = self.x + self.w / 2, self.y + self.w / 2
        child_idx_x = 1 if px >= mid_x else 0
        child_idx_y = 1 if py >= mid_y else 0
        self.children[child_idx_x + 2 * child_idx_y].insert(idx, positions, masses, depth)
        
    def subdivide(self):
        self.is_external = False
        self.children = [None] * 4
        half_w = self.w / 2
        self.children[0] = QuadTreeNode(self.x, self.y, half_w)
        self.children[1] = QuadTreeNode(self.x + half_w, self.y, half_w)
        self.children[2] = QuadTreeNode(self.x, self.y + half_w, half_w)
        self.children[3] = QuadTreeNode(self.x + half_w, self.y + half_w, half_w)
        
    def compute_force(self, idx, positions, masses, theta, G, epsilon):
        if self.mass == 0: return 0.0, 0.0
        dx, dy = self.cx - positions[idx, 0], self.cy - positions[idx, 1]
        dist_sq = dx*dx + dy*dy
        if self.is_external:
            if self.body == idx: return 0.0, 0.0
        else:
            dist = np.sqrt(dist_sq)
            if dist > 0 and (self.w / dist < theta): pass
            else:
                ax, ay = 0.0, 0.0
                for child in self.children:
                    if child is not None and child.mass > 0:
                        cax, cay = child.compute_force(idx, positions, masses, theta, G, epsilon)
                        ax += cax
                        ay += cay
                return ax, ay
        r2 = dist_sq + epsilon**2
        inv_r3 = 1.0 / (r2 * np.sqrt(r2))
        return G * self.mass * dx * inv_r3, G * self.mass * dy * inv_r3

def compute_forces_bh(positions, masses, theta):
    min_c, max_c = np.min(positions, axis=0), np.max(positions, axis=0)
    width = max(max_c[0] - min_c[0], max_c[1] - min_c[1]) * 1.01
    center = (min_c + max_c) / 2
    root = QuadTreeNode(center[0] - width/2, center[1] - width/2, width)
    for i in range(len(positions)): root.insert(i, positions, masses)
    accels = np.zeros((len(positions), 2))
    for i in range(len(positions)):
        accels[i, 0], accels[i, 1] = root.compute_force(i, positions, masses, theta, G, epsilon)
    return accels

def compute_energy(positions, velocities, masses):
    v2 = np.sum(velocities**2, axis=1)
    K = 0.5 * np.sum(masses * v2)
    dx = positions[:, 0].reshape(1, -1) - positions[:, 0].reshape(-1, 1)
    dy = positions[:, 1].reshape(1, -1) - positions[:, 1].reshape(-1, 1)
    r = np.sqrt(dx**2 + dy**2 + epsilon**2)
    np.fill_diagonal(r, np.inf)
    U = -0.5 * G * np.sum(np.outer(masses, masses) / r)
    return K, U, K + U

# ==========================================
# БЕНЧМАРК И ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ ОТЧЕТА
# ==========================================
def generate_report_data():
    print(">>> Генерация данных для отчета...")
    
    # 1. Зависимость времени от N
    N_list = [100, 200, 500, 1000, 2000]
    t_pp_list, t_bh_list = [], []
    for N in N_list:
        pos, _, mass = initialize_particles(N)
        t0 = time.time(); compute_forces_pp(pos, mass); t_pp_list.append(time.time() - t0)
        t0 = time.time(); compute_forces_bh(pos, mass, 0.5); t_bh_list.append(time.time() - t0)
        
    # 2. Зависимость ошибки от theta
    pos, _, mass = initialize_particles(1000)
    accels_pp = compute_forces_pp(pos, mass)
    theta_list = [0.1, 0.3, 0.5, 0.7, 1.0]
    errors, t_bhs = [], []
    for theta in theta_list:
        t0 = time.time()
        accels_bh = compute_forces_bh(pos, mass, theta)
        t_bhs.append(time.time() - t0)
        errors.append(np.linalg.norm(accels_bh - accels_pp) / np.linalg.norm(accels_pp) * 100)
        
    # 3. Сохранение энергии в симуляции
    pos, vel, mass = initialize_particles(500)
    acc = compute_forces_bh(pos, mass, 0.5)
    energies, steps, dt = [], 100, 0.005
    for step in range(steps):
        pos += vel * dt + 0.5 * acc * dt**2
        acc_new = compute_forces_bh(pos, mass, 0.5)
        vel += 0.5 * (acc + acc_new) * dt
        acc = acc_new
        if step % 10 == 0: energies.append(compute_energy(pos, vel, mass)[2])

    # --- Отрисовка графиков ---
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.loglog(N_list, t_pp_list, 'o-', label='Particle-Particle $O(N^2)$')
    plt.loglog(N_list, t_bh_list, 's-', label='Barnes-Hut $O(N \log N)$')
    plt.xlabel('Число частиц N'); plt.ylabel('Время (сек)')
    plt.title('Производительность (log-log шкала)'); plt.legend(); plt.grid(True, which="both", ls="--")
    
    plt.subplot(2, 2, 2)
    plt.plot(theta_list, errors, 'o-', color='red')
    plt.xlabel('Параметр $\\theta$'); plt.ylabel('Ошибка (%)')
    plt.title('Точность алгоритма Барнса-Хата'); plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(0, steps, 10), energies, 'o-', color='purple')
    plt.xlabel('Шаг времени'); plt.ylabel('Полная энергия E')
    plt.title('Сохранение энергии в системе'); plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(theta_list, t_bhs, 'o-', color='green')
    plt.xlabel('Параметр $\\theta$'); plt.ylabel('Время (сек)')
    plt.title('Затраты времени от точности'); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lab2_charts.png')
    print("Графики сохранены в 'lab2_charts.png'")
    
    # --- Визуализация Дерева (QuadTree) ---
    plt.figure(figsize=(8, 8))
    pos, _, mass = initialize_particles(200) # Меньше частиц, чтобы дерево было читаемым
    plt.scatter(pos[:, 0], pos[:, 1], s=15, c='blue', label='Частицы', zorder=5)
    
    def draw_tree(node):
        if node.mass == 0: return
        rect = plt.Rectangle((node.x, node.y), node.w, node.w, fill=False, color='gray', alpha=0.4, linewidth=1)
        plt.gca().add_patch(rect)
        if not node.is_external and node.children:
            for child in node.children:
                if child is not None: draw_tree(child)
                
    min_c, max_c = np.min(pos, axis=0), np.max(pos, axis=0)
    width = max(max_c[0] - min_c[0], max_c[1] - min_c[1]) * 1.01
    center = (min_c + max_c) / 2
    root = QuadTreeNode(center[0] - width/2, center[1] - width/2, width)
    for i in range(len(pos)): root.insert(i, pos, mass)
    draw_tree(root)
    
    plt.title('Структура Кваддерева (QuadTree) для N=200')
    plt.xlabel('X'); plt.ylabel('Y'); plt.axis('equal'); plt.grid(True, alpha=0.3)
    plt.savefig('lab2_tree.png')
    print("Визуализация дерева сохранена в 'lab2_tree.png'")
    plt.show()

if __name__ == "__main__":
    generate_report_data()