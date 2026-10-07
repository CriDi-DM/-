import numpy as np
import matplotlib.pyplot as plt

GRID, STEPS, R, EXTRA = 1500, 15000, 4, 3000

grid = np.zeros((GRID, GRID), dtype=np.uint8)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
r = c = GRID // 2
r0, c0 = r, c
d = 0
nblack = 0

def ant_step():
    global r, c, d, nblack
    s = grid[r, c]
    if s == 0:
        d = (d - 1) % 4; grid[r, c] = 1; nblack += 1
    else:
        d = (d + 1) % 4; grid[r, c] = 0; nblack -= 1
    dr, dc = DIRS[d]
    r += dr; c += dc

seen = {}
whist = [None]
dist_hist, ncol_hist = [], []
prev, conf, result = None, 0, None

for t in range(1, STEPS + 1):
    ant_step()
    w = (d, grid[r-R:r+R+1, c-R:c+R+1].tobytes())
    whist.append(w)
    dist_hist.append(abs(r - r0) + abs(c - c0))
    ncol_hist.append(nblack)
    if w in seen:
        t0, (p0r, p0c) = seen[w]
        tau, dd = t - t0, (r - p0r, c - p0c)
        conf = conf + 1 if prev == (tau, dd) else 1
        prev = (tau, dd)
        if conf >= 3:
            result = (t, tau, dd)
            break
    else:
        conf, prev = 0, None
    seen[w] = (t, (r, c))

if result is None:
    raise SystemExit("Магистраль не обнаружена за STEPS шагов")

t_det, tau, dd = result

# Обратный просмотр истории окон: корректная стартовая позиция
i = t_det - tau + 1
while i - 1 >= 1 and whist[i - 1] == whist[i - 1 + tau]:
    i -= 1
t_onset = i

# Досимулируем ленту для красивого рисунка
for _ in range(EXTRA):
    ant_step()
    dist_hist.append(abs(r - r0) + abs(c - c0))
    ncol_hist.append(nblack)

manh = abs(dd[0]) + abs(dd[1])
print(f"Момент детекции:             t_det   = {t_det}")
print(f"Точный момент возникновения: t_onset = {t_onset}")
print(f"Период цикла:                τ       = {tau}")
print(f"Сдвиг за период:             d       = {dd} (манхэттен {manh})")
print(f"Скорость: {manh/tau:.5f} кл/шаг (манх.), {np.hypot(*dd)/tau:.5f} кл/шаг (евкл.)")
print(f"Чёрных клеток за период:     {ncol_hist[t_det-1] - ncol_hist[t_det-1-tau]}")
print(f"Дистанция в момент возникновения: {dist_hist[t_onset-1]}")

with open('series1_metrics.csv', 'w') as fh:
    fh.write('t,dist,ncol\n')
    for k, (ds, nc) in enumerate(zip(dist_hist, ncol_hist), 1):
        fh.write(f'{k},{ds},{nc}\n')
print("Метрики сохранены в series1_metrics.csv")

# ===== Рисунки для отчёта =====
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

a = max(0, t_onset - 3000)
b = min(len(dist_hist), t_onset + 3000)
seg = dist_hist[a:b]                      # режем по фактической длине истории
axes[0].plot(range(a, a + len(seg)), seg, lw=0.7)
axes[0].axvline(t_onset, color='r', ls='--', label=f't_onset = {t_onset}')
axes[0].set_title('Переход хаос → магистраль (крупным планом)')
axes[0].set_xlabel('шаг'); axes[0].set_ylabel('манхэттенское расстояние')
axes[0].legend(); axes[0].grid(alpha=0.3)

r_lo, r_hi = min(r0, r) - 60, max(r0, r) + 60
c_lo, c_hi = min(c0, c) - 60, max(c0, c) + 60
axes[1].matshow(grid[r_lo:r_hi, c_lo:c_hi], cmap='Greys')
axes[1].set_title(f'Магистраль: τ={tau}, d={dd}, t_onset={t_onset}')
axes[1].axis('off')

plt.tight_layout()
plt.show()