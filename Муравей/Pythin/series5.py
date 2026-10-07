import hashlib
import numpy as np
import matplotlib.pyplot as plt

TURN4 = {'L': 3, 'R': 1, 'U': 2, 'F': 0}
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MIR_V = {0: 2, 2: 0, 1: 1, 3: 3}
MIR_H = {1: 3, 3: 1, 0: 0, 2: 2}

def box_run(m=50, boundary='reflect', T=200000, rule='LR'):
    OFF = 3; M = m + 2 * OFF
    grid = np.zeros((M, M), dtype=np.uint8)
    r = c = OFF + m // 2; d = 0
    r0, c0 = r, c
    seen = {}; t_wall = None; n_refl = 0; dist = []
    for t in range(1, T + 1):
        s = grid[r, c]
        d = (d + TURN4[rule[s]]) % 4
        grid[r, c] = (s + 1) % len(rule)
        nr, nc = r + DIRS4[d][0], c + DIRS4[d][1]
        if boundary == 'periodic':
            nr = OFF + (nr - OFF) % m; nc = OFF + (nc - OFF) % m
        else:
            if nr < OFF or nr >= OFF + m:
                d = MIR_V[d]; nr = r + DIRS4[d][0]; n_refl += 1
                if t_wall is None: t_wall = t
            if nc < OFF or nc >= OFF + m:
                d = MIR_H[d]; nc = c + DIRS4[d][1]; n_refl += 1
                if t_wall is None: t_wall = t
        r, c = nr, nc
        dist.append(abs(r - r0) + abs(c - c0))
        key = hashlib.md5(grid.tobytes() + bytes((r, c, d))).digest()
        if key in seen:
            return dict(m=m, boundary=boundary, tau=t - seen[key], t_start=seen[key],
                        t=t, t_wall=t_wall, n_refl=n_refl, dist=dist,
                        grid=grid[OFF:OFF+m, OFF:OFF+m].copy())
        seen[key] = t
    return dict(m=m, boundary=boundary, tau=None, t_start=None, t=T,
                t_wall=t_wall, n_refl=n_refl, dist=dist,
                grid=grid[OFF:OFF+m, OFF:OFF+m].copy())

DIRS6 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
TURN6 = {'R': 1, 'L': 5, 'U': 3, 'F': 0}
def hex_sim(rule, steps):
    q = len(rule); field = {}; pos = (0, 0); d = 0
    for _ in range(steps):
        s = field.get(pos, 0)
        d = (d + TURN6[rule[s]]) % 6
        field[pos] = (s + 1) % q
        if field[pos] == 0: field.pop(pos)
        dq, dr = DIRS6[d]; pos = (pos[0]+dq, pos[1]+dr)
    return field

if __name__ == '__main__':
    rows = []; pics = {}
    for m in (30, 50, 70):
        for bnd in ('reflect', 'periodic'):
            res = box_run(m=m, boundary=bnd)
            rows.append(res); pics[(m, bnd)] = res['grid']
            print(f"m={m:2d} {bnd:8s}: истинный цикл τ={res['tau']}, начало цикла t={res['t_start']}, "
                  f"детекция на t={res['t']}, первый удар t={res['t_wall']}, отражений={res['n_refl']}",
                  flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for j, m in enumerate((30, 50, 70)):
        axes[0, j].matshow(pics[(m, 'reflect')], cmap='Greys')
        axes[0, j].set_title(f'm={m}, отражающие'); axes[0, j].axis('off')
        axes[1, j].matshow(pics[(m, 'periodic')], cmap='Greys')
        axes[1, j].set_title(f'm={m}, тор'); axes[1, j].axis('off')
    plt.tight_layout(); plt.savefig('series5b_fields.png', dpi=150)

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
    for res in rows:
        if res['m'] in (30, 50):
            ax = axes2[0 if res['m'] == 30 else 1]
            ax.plot(res['dist'], lw=0.6, label=res['boundary'])
            if res['t_wall']: ax.axvline(res['t_wall'], color='r', ls=':', alpha=0.7)
    for ax, m in zip(axes2, (30, 50)):
        ax.set_title(f'Дистанция от старта, m={m}'); ax.set_xlabel('шаг')
        ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig('series5b_dist.png', dpi=150)

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 7))
    for ax, rule in zip(axes3, ('LR', 'RF')):
        fld = hex_sim(rule, 5000)
        ax.scatter([2*a+b for a, b in fld], [b for a, b in fld],
                   c=[fld[k] for k in fld], cmap='viridis', marker='h', s=30)
        ax.set_title(f'Гекс, правило {rule}, 5000 шагов (визуальная проверка магистрали)')
        ax.axis('off')
    plt.tight_layout(); plt.savefig('series5b_hex.png', dpi=150)
    print('Сохранены: series5b_fields.png, series5b_dist.png, series5b_hex.png')
    plt.show()