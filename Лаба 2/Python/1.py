import numpy as np
import time

# ==========================================
# ПАРАМЕТРЫ МОДЕЛИ
# ==========================================
G = 1.0            # Гравитационная постоянная
epsilon = 0.05     # Параметр сглаживания (softening), предотвращает бесконечные силы
N = 200            # Число частиц (для теста прямого метода 200 — это быстро)
M_total = 1.0      # Суммарная масса системы
m = M_total / N    # Масса одной частицы
dt = 0.005         # Шаг по времени
steps = 100        # Количество шагов для теста

# ==========================================
# 1. ГЕНЕРАЦИЯ НАЧАЛЬНЫХ УСЛОВИЙ
# ==========================================
def initialize_particles(N, R=1.0):
    np.random.seed(42)
    # Равномерное распределение в круге
    r = R * np.sqrt(np.random.uniform(0, 1, N))
    theta = np.random.uniform(0, 2*np.pi, N)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    positions = np.column_stack((x, y))
    
    # Начальные скорости: задаем легкое вращение
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
    N = len(positions)
    # dx[i, j] = x_j - x_i (вектор от i к j)
    dx = positions[:, 0].reshape(1, N) - positions[:, 0].reshape(N, 1)
    dy = positions[:, 1].reshape(1, N) - positions[:, 1].reshape(N, 1)
    
    r2 = dx**2 + dy**2 + epsilon**2
    r = np.sqrt(r2)
    
    # 1 / r^3
    inv_r3 = 1.0 / (r2 * r)
    np.fill_diagonal(inv_r3, 0.0) # Убираем взаимодействие частицы с самой собой
    
    # a_i = sum_j (G * m_j * (r_j - r_i) / r_ij^3)
    ax = G * np.sum(masses.reshape(1, N) * inv_r3 * dx, axis=1)
    ay = G * np.sum(masses.reshape(1, N) * inv_r3 * dy, axis=1)
    
    return np.column_stack((ax, ay))

# ==========================================
# 3. РАСЧЁТ ЭНЕРГИИ И ИМПУЛЬСА
# ==========================================
def compute_energy(positions, velocities, masses):
    # Кинетическая энергия
    v2 = np.sum(velocities**2, axis=1)
    K = 0.5 * np.sum(masses * v2)
    
    # Потенциальная энергия
    dx = positions[:, 0].reshape(1, -1) - positions[:, 0].reshape(-1, 1)
    dy = positions[:, 1].reshape(1, -1) - positions[:, 1].reshape(-1, 1)
    r = np.sqrt(dx**2 + dy**2 + epsilon**2)
    
    np.fill_diagonal(r, np.inf) # Исключаем самовоздействие
    U = -0.5 * G * np.sum(np.outer(masses, masses) / r)
    
    return K, U, K + U

def compute_momentum(velocities, masses):
    px = np.sum(masses * velocities[:, 0])
    py = np.sum(masses * velocities[:, 1])
    return px, py

# ==========================================
# 4. ОСНОВНОЙ ЦИКЛ СИМУЛЯЦИИ (VELOCITY VERLET)
# ==========================================
def run_simulation():
    print(f"--- Запуск симуляции: N={N}, dt={dt}, steps={steps} ---")
    positions, velocities, masses = initialize_particles(N)
    accelerations = compute_forces_pp(positions, masses)
    
    K, U, E = compute_energy(positions, velocities, masses)
    px, py = compute_momentum(velocities, masses)
    
    print(f"Step   0 | E={E:+.6f} (K={K:.6f}, U={U:.6f}) | P=({px:+.2e}, {py:+.2e}) | dE= 0.0000%")
    
    E0 = E
    start_time = time.time()
    
    for step in range(1, steps + 1):
        # 1. Обновляем координаты: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        positions += velocities * dt + 0.5 * accelerations * dt**2
        
        # 2. Вычисляем новые ускорения: a(t+dt)
        new_accelerations = compute_forces_pp(positions, masses)
        
        # 3. Обновляем скорости: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        velocities += 0.5 * (accelerations + new_accelerations) * dt
        
        # 4. Заменяем старые ускорения на новые
        accelerations = new_accelerations
        
        # Выводим статистику каждые 10 шагов
        if step % 10 == 0 or step == steps:
            K, U, E = compute_energy(positions, velocities, masses)
            px, py = compute_momentum(velocities, masses)
            dE = (E - E0) / abs(E0) * 100
            print(f"Step {step:3d} | E={E:+.6f} (K={K:.6f}, U={U:.6f}) | P=({px:+.2e}, {py:+.2e}) | dE={dE:+.4f}%")
            
    end_time = time.time()
    print(f"--- Симуляция завершена за {end_time - start_time:.4f} секунд ---")

if __name__ == "__main__":
    run_simulation()