import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell # Хотя для 2D лучше написать вручную, но для проверки сойдет и аналитика
from tqdm import tqdm # Для красивого прогресс-бара в консоли

# --- Параметры (можно увеличить N_STEPS для лучшей статистики) ---
N_PARTICLES = 500      # Увеличим число частиц для гладкости графика
BOX_SIZE = 100.0
TEMPERATURE = 1.5      # Зададим "температуру" (дисперсию скоростей)
N_STEPS = 1000         # Количество шагов для набора статистики
SAMPLE_EVERY = 10      # Как часто брать пробу (чтобы данные не были слишком коррелированы)

# --- Инициализация ---
positions = np.random.rand(N_PARTICLES, 2) * BOX_SIZE
# Важно: начальное распределение скоростей должно соответствовать T
velocities = np.random.normal(0, np.sqrt(TEMPERATURE), (N_PARTICLES, 2))

# Список для сохранения всех измеренных скоростей
all_speeds = []

print(f"Запуск симуляции: {N_PARTICLES} частиц, {N_STEPS} шагов...")

# --- Цикл симуляции ---
for step in tqdm(range(N_STEPS)):
    # 1. Движение
    positions += velocities 
    
    # 2. Стенки (упругое отражение)
    # X
    mask_x_out = (positions[:, 0] < 0) | (positions[:, 0] > BOX_SIZE)
    velocities[mask_x_out, 0] *= -1
    positions[:, 0] = np.clip(positions[:, 0], 0, BOX_SIZE) # Жестко возвращаем в границы
    
    # Y
    mask_y_out = (positions[:, 1] < 0) | (positions[:, 1] > BOX_SIZE)
    velocities[mask_y_out, 1] *= -1
    positions[:, 1] = np.clip(positions[:, 1], 0, BOX_SIZE)

    # 3. Сбор статистики (Метод Монте-Карло)
    if step % SAMPLE_EVERY == 0:
        current_speeds = np.linalg.norm(velocities, axis=1)
        all_speeds.extend(current_speeds)

# Превращаем в numpy array
all_speeds = np.array(all_speeds)

print(f"Собрано {len(all_speeds)} измерений скорости.")
print(f"Средняя скорость: {np.mean(all_speeds):.3f}")
print(f"Средняя кинетическая энергия: {0.5 * np.mean(all_speeds**2):.3f} (должна быть близка к T={TEMPERATURE})")

# --- Визуализация результатов ---
plt.figure(figsize=(10, 6))

# 1. Гистограмма из симуляции
# bins=50 для детализации, density=True для нормировки на вероятность
counts, bins, ignored = plt.hist(all_speeds, bins=50, density=True, alpha=0.6, 
                                 color='b', label='Симуляция (Монте-Карло)')

# 2. Теоретическая кривая (Распределение Максвелла-Больцмана для 2D)
# f(v) = (v / T) * exp(-v^2 / (2T))
v_theory = np.linspace(0, max(bins), 200)
pdf_theory = (v_theory / TEMPERATURE) * np.exp(-v_theory**2 / (2 * TEMPERATURE))

plt.plot(v_theory, pdf_theory, 'r-', lw=2, label=f'Теория (Максвелл 2D, T={TEMPERATURE})')

plt.title('Распределение скоростей частиц в газе')
plt.xlabel('Модуль скорости |v|')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.grid(True, alpha=0.3)

# Сохраняем график
plt.savefig('maxwell_distribution.png', dpi=150)
print("График сохранен как 'maxwell_distribution.png'")

plt.show()