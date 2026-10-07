import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ===== Параметры =====
M, N = 100, 100        # размер поля
STEPS = 50             # число шагов эволюции
SAVE_GIF = False       # True — сохранить анимацию в parity.gif

# ===== Инициализация поля =====
# Вариант А: одна живая клетка в центре (красивая симметричная эволюция)
field = np.zeros((M, N), dtype=np.uint8)
field[M // 2, N // 2] = 1

# Вариант Б (раскомментируйте, чтобы попробовать): случайное заполнение
# np.random.seed(42)
# field = (np.random.rand(M, N) > 0.8).astype(np.uint8)

frames = [field.copy()]  # сохраняем начальный кадр

# ===== Эволюция =====
for t in range(STEPS):
    new = np.zeros_like(field)

    # Сумма состояний 4-х соседей по фон Нейману (периодические границы — через np.roll)
    s = (
        np.roll(field,  1, axis=0) +   # сверху
        np.roll(field, -1, axis=0) +   # снизу
        np.roll(field,  1, axis=1) +   # слева
        np.roll(field, -1, axis=1)     # справа
    )

    # Новое состояние = 1, если сумма нечётная (сумма % 2 == 1)
    new = (s % 2).astype(np.uint8)

    field = new
    frames.append(field.copy())

    # Печатаем статистику каждые 10 шагов
    if (t + 1) % 10 == 0:
        print(f"Шаг {t+1:3d}: живых клеток = {field.sum()}, "
              f"плотность = {field.mean():.3f}")

# ===== Визуализация =====
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.matshow(frames[0], cmap='Greys', vmin=0, vmax=1)
ax.set_title("Правило чётности (шаг 0)")
ax.axis('off')

def update(i):
    im.set_data(frames[i])
    ax.set_title(f"Правило чётности (шаг {i})")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                              interval=150, blit=True)

if SAVE_GIF:
    print("Сохраняю анимацию в parity.gif...")
    ani.save("parity.gif", writer="pillow", fps=8)
    print("Готово!")

plt.show()