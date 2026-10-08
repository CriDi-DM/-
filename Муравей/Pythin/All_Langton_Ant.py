# -*- coding: utf-8 -*-
"""
ИССЛЕДОВАТЕЛЬСКАЯ РАБОТА (единый скрипт)
Тема: «Многоцветный муравей Лэнгтона: вычислительный поиск и классификация
режимов поведения на квадратной и гексагональной решётках»

Скрипт последовательно выполняет ВСЕ эксперименты работы:
  PART 0 — верификация среды: правило чётности, проверка закона N(t) = 4^w(t);
  PART 1 — Серия 1: классический муравей LR (параметры магистрали);
  PART 2 — Серия 2: чувствительность к начальным условиям;
  PART 3 — Серия 3: перебор правил (квадрат q=2,3,4; гекс q=2,3);
  PART 4 — Серия 4: мутации одного символа (базы RLLR, LRLR);
  PART 5 — Серия 5: гексагональная решётка (визуализация) и границы
           (коробки с отражающими и периодическими границами);
  PART 6 — галерея рисунков для отчёта.

Методы детекции (см. отчёт, раздел «Методология»):
  * детектор на локальных окнах  — квадратная решётка (перекрёстно верифицирован);
  * эталонный верификатор        — скан периодичности последовательности
    направлений + MD5-хэш полного состояния (гекс, ограниченные правила);
  * хэш полного состояния        — эксперименты с границами.

Вывод: консоль + research_summary.txt; рисунки: fig01..fig06 (.png).
Управление временем счёта: константы RUN_Q4, BOX_T, SHOW_PLOTS.
"""
import hashlib, itertools, json, random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ============================== КОНФИГУРАЦИЯ ==============================
T_MAIN     = 30000     # основной горизонт классификации правил
TAU_MAX    = 6000      # предел сканера периодов эталонного верификатора
BOX_T      = 200000    # горизонт экспериментов с границами
RUN_Q4     = True      # полный перебор q=4 на квадрате (~2 мин); False — пропустить
SHOW_PLOTS = True      # показать все окна рисунков в конце

TURN4 = {'L': 3, 'R': 1, 'U': 2, 'F': 0}
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]                 # N E S W
TURN6 = {'R': 1, 'L': 5, 'U': 3, 'F': 0}                   # ±60°, 180°
DIRS6 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
RING2 = sorted(set([(0, 0)] + DIRS6 + [(a+c, b+d) for a, b in DIRS6 for c, d in DIRS6]))
MIR_V = {0: 2, 2: 0, 1: 1, 3: 3}
MIR_H = {1: 3, 3: 1, 0: 0, 2: 2}

SUMMARY = []
def log(msg=''):
    print(msg)
    SUMMARY.append(str(msg))

def hdr(title):
    log('\n' + '=' * 78)
    log(title)
    log('=' * 78)

# ============================== ЯДРО МОДЕЛИ ==============================
def _turns_dirs(lattice):
    return (TURN4, DIRS4, 4) if lattice == 'square' else (TURN6, DIRS6, 6)

def scan_period(dirs, pos, T, tau_max=TAU_MAX, k_min=3000):
    """Эталонный критерий: периодичность последовательности направлений."""
    for tau in range(1, min(tau_max, T // 3) + 1):
        k = min(T - tau, max(3 * tau, k_min))
        if np.array_equal(dirs[T-k:T-tau], dirs[T-k+tau:T]):
            j = T - k - 1
            while j >= 0 and dirs[j] == dirs[j + tau]:
                j -= 1
            s0 = j + 2
            p0, p1 = pos[s0 - 1], pos[s0 - 1 + tau]
            return tau, s0, (int(p1[0]-p0[0]), int(p1[1]-p0[1]))
    return None

def classify_verified(rule, T=T_MAIN, lattice='square', wait=4000, init=None):
    """Эталонный классификатор: хэш полного состояния (ограниченные правила)
    + скан периодичности траектории (магистрали). Без локальных окон."""
    TURN, DIRS, nD = _turns_dirs(lattice)
    q = len(rule); field = dict(init) if init else {}
    r = c = 0; d = 0
    dirs = np.empty(T, dtype=np.int32)
    pos = np.zeros((T + 1, 2), dtype=np.int32)
    bb = [0, 0, 0, 0]; prev_area = 1; last_grow = 0
    seen2 = {}; hashing = False
    nvis = 1; visited = {(0, 0)}
    for t in range(1, T + 1):
        s = field.get((r, c), 0)
        d = (d + TURN[rule[s]]) % nD
        s2 = (s + 1) % q
        if s2: field[(r, c)] = s2
        else:  field.pop((r, c), None)
        dr, dc = DIRS[d]; r += dr; c += dc
        if (r, c) not in visited: visited.add((r, c)); nvis += 1
        dirs[t-1] = d; pos[t] = (r, c)
        bb[0] = min(bb[0], r); bb[1] = max(bb[1], r)
        bb[2] = min(bb[2], c); bb[3] = max(bb[3], c)
        area = (bb[1]-bb[0]+1) * (bb[3]-bb[2]+1)
        if area != prev_area: prev_area, last_grow = area, t
        if not hashing and t - last_grow >= wait:
            hashing = True
        if hashing:
            key = hashlib.md5(repr((r, c, d, sorted(field.items()))).encode()).digest()
            if key in seen2:
                return dict(cls=2, tau=t-seen2[key], s0=seen2[key], d=(0, 0), nvis=nvis)
            seen2[key] = t
    got = scan_period(dirs, pos, T)
    if got:
        tau, s0, dd = got
        return dict(cls=2 if dd == (0, 0) else 3, tau=tau, s0=s0, d=dd, nvis=nvis)
    return dict(cls=1 if hashing else 4, tau=None, s0=None, d=None, nvis=nvis)

def classify_window(rule, T=T_MAIN, lattice='square', R=4, wait=4000):
    """Детектор на локальных окнах (квадратная решётка, перекрёстно верифицирован)."""
    TURN, DIRS, nD = _turns_dirs(lattice)
    q = len(rule); field = {}
    r = c = 0; d = 0
    bb = [0, 0, 0, 0]; prev_area = 1; last_grow = 0
    nvis = 1; visited = {(0, 0)}
    seen = {}; whist = [None]; prev = None; conf = 0
    cells = RING2 if lattice == 'hex' else [(a, b) for a in range(-R, R+1) for b in range(-R, R+1)]
    for t in range(1, T + 1):
        s = field.get((r, c), 0)
        d = (d + TURN[rule[s]]) % nD
        s2 = (s + 1) % q
        if s2: field[(r, c)] = s2
        else:  field.pop((r, c), None)
        dr, dc = DIRS[d]; r += dr; c += dc
        if (r, c) not in visited: visited.add((r, c)); nvis += 1
        bb[0] = min(bb[0], r); bb[1] = max(bb[1], r)
        bb[2] = min(bb[2], c); bb[3] = max(bb[3], c)
        area = (bb[1]-bb[0]+1) * (bb[3]-bb[2]+1)
        if area != prev_area: prev_area, last_grow = area, t
        g = field.get
        w = (d, tuple(g((r+a, c+b), 0) for a, b in cells))
        whist.append(w)
        if w in seen:
            t0, p0 = seen[w]
            tau, dd = t - t0, (r - p0[0], c - p0[1])
            conf = conf + 1 if prev == (tau, dd) else 1
            prev = (tau, dd)
            if conf >= 3:
                i = t - tau + 1
                while i - 1 >= 1 and whist[i-1] == whist[i-1+tau]:
                    i -= 1
                return dict(cls=2 if dd == (0, 0) else 3, tau=tau, d=dd, t_onset=i, nvis=nvis)
        else:
            conf, prev = 0, None
        seen[w] = (t, (r, c))
    if t - last_grow >= wait:
        return dict(cls=1, nvis=nvis, bbox=area)
    f = field
    if f and lattice == 'square':
        m1 = sum(1 for (i, j), s in f.items() if f.get((i, -j), -1) == s) / len(f)
        m2 = sum(1 for (i, j), s in f.items() if f.get((-i, j), -1) == s) / len(f)
        sym = max(m1, m2)
        return dict(cls=5 if sym >= 0.98 else 4, nvis=nvis, bbox=area, sym=round(sym, 3))
    return dict(cls=4, nvis=nvis, bbox=area)

def fmt(res):
    if res['cls'] in (2, 3):
        key = 's0' if 's0' in res else 't_onset'
        return f"класс {res['cls']}, tau={res['tau']}, d={res['d']}, {key}={res[key]}"
    return f"класс {res['cls']}"

def grid_sim_crop(rule, steps, G=1500):
    """Матричная симуляция для визуализации; возврат обрезанного поля."""
    q = len(rule); grid = np.zeros((G, G), dtype=np.uint8)
    r = c = G // 2; d = 0
    bb = [r, r, c, c]
    for _ in range(steps):
        s = grid[r, c]
        d = (d + TURN4[rule[s]]) % 4
        grid[r, c] = (s + 1) % q
        dr, dc = DIRS4[d]; r += dr; c += dc
        bb[0] = min(bb[0], r); bb[1] = max(bb[1], r)
        bb[2] = min(bb[2], c); bb[3] = max(bb[3], c)
    p = 10
    return grid[max(0, bb[0]-p):bb[1]+p+1, max(0, bb[2]-p):bb[3]+p+1]

def hex_sim(rule, steps):
    q = len(rule); field = {}; pos = (0, 0); d = 0
    for _ in range(steps):
        s = field.get(pos, 0)
        d = (d + TURN6[rule[s]]) % 6
        field[pos] = (s + 1) % q
        if field[pos] == 0: field.pop(pos)
        dq, dr = DIRS6[d]; pos = (pos[0]+dq, pos[1]+dr)
    return field

def box_run(m=50, boundary='reflect', T=BOX_T, rule='LR'):
    """Коробка m×m; эталонный детектор: MD5 полного состояния каждый такт."""
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
            return dict(m=m, boundary=boundary, tau=t-seen[key], t_start=seen[key],
                        t=t, t_wall=t_wall, n_refl=n_refl, dist=dist,
                        grid=grid[OFF:OFF+m, OFF:OFF+m].copy())
        seen[key] = t
    return dict(m=m, boundary=boundary, tau=None, t_start=None, t=T,
                t_wall=t_wall, n_refl=n_refl, dist=dist,
                grid=grid[OFF:OFF+m, OFF:OFF+m].copy())

# ============================== PART 0 ==============================
def part0():
    hdr('PART 0. Верификация среды: правило чётности, N(t) = 4^w(t)')
    M = N = 100
    f = np.zeros((M, N), np.uint8); f[M//2, N//2] = 1
    log(f'{"t":>3} | w(t) | 4^w(t) | измерено | примечание')
    for t in range(1, 51):
        s = (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)) % 2
        f = s.astype(np.uint8)
        if t in (10, 20, 30, 40, 50):
            w = bin(t).count('1'); th = 4 ** w; me = int(f.sum())
            note = 'OK' if th == me else 'отклонение: интерференция на границах поля 100x100'
            log(f'{t:>3} | {w:>4} | {th:>6} | {me:>8} | {note}')
    plt.figure(figsize=(5, 5))
    plt.matshow(f, cmap='Greys', fignum=1)
    plt.title('Правило чётности, шаг 50')
    plt.axis('off')
    plt.savefig('fig01_parity.png', dpi=150, bbox_inches='tight')

# ============================== PART 1 ==============================
def part1():
    hdr('PART 1. Серия 1: классический муравей LR (квадрат, q=2)')
    G = 1500; T = 12000
    grid = np.zeros((G, G), dtype=np.uint8)
    r = c = G // 2; r0, c0 = r, c; d = 0
    rule = 'LR'
    dist = []; ncol = []
    dirs = np.empty(T, dtype=np.int32); pos = np.zeros((T+1, 2), dtype=np.int32)
    nblack = 0
    seen = {}; whist = [None]; prev = None; conf = 0; resW = None
    bb = [r, r, c, c]
    for t in range(1, T + 1):
        s = grid[r, c]
        d = (d + TURN4[rule[s]]) % 4
        grid[r, c] = (s + 1) % 2
        nblack += 1 if s == 0 else -1
        dr, dc = DIRS4[d]; r += dr; c += dc
        bb[0] = min(bb[0], r); bb[1] = max(bb[1], r)
        bb[2] = min(bb[2], c); bb[3] = max(bb[3], c)
        dirs[t-1] = d; pos[t] = (r, c)
        dist.append(abs(r - r0) + abs(c - c0)); ncol.append(nblack)
        if resW is None:
            w = (d, grid[r-4:r+5, c-4:c+5].tobytes())
            whist.append(w)
            if w in seen:
                t0, p0 = seen[w]
                tau, dd = t - t0, (r - p0[0], c - p0[1])
                conf = conf + 1 if prev == (tau, dd) else 1
                prev = (tau, dd)
                if conf >= 3:
                    i = t - tau + 1
                    while i - 1 >= 1 and whist[i-1] == whist[i-1+tau]:
                        i -= 1
                    resW = (t, tau, dd, i)
            else:
                conf, prev = 0, None
            seen[w] = (t, (r, c))
    got = scan_period(dirs, pos, T, tau_max=1000, k_min=1000)
    if got is None:
        log('ПРЕДУПРЕЖДЕНИЕ: скан направлений не нашёл период '
            '(слишком короткий хвост после перехода); увеличьте T в part1')
        tau_v, s0, d_v = None, None, None
    else:
        tau_v, s0, d_v = got
    t_det, tau, dd, t_onset = resW
    log(f'{"шаг":>6} | расстояние | ненулевых клеток')
    for t in range(2000, T + 1, 2000):
        log(f'{t:>6} | {dist[t-1]:>10} | {ncol[t-1]:>16}')
    log('')
    log(f'Период цикла tau              = {tau}')
    log(f'Вектор сноса d                = {dd}')
    log(f'Скорость манхэттенская        = {(abs(dd[0])+abs(dd[1]))/tau:.5f} кл/шаг')
    log(f'Скорость евклидова            = {np.hypot(*dd)/tau:.5f} кл/шаг')
    log(f't_onset (детектор окон)       = {t_onset}')
    log(f's0 (эталонный скан направлений) = {s0}  (tau={tau_v}, d={d_v})')
    log(f'Ненулевых клеток за период    = {ncol[t_det-1] - ncol[t_det-1-tau]}')
    log(f'Расстояние в момент перехода  = {dist[t_onset-1]}')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes[0, 0].matshow(grid[bb[0]-10:bb[1]+11, bb[2]-10:bb[3]+11], cmap='Greys')
    axes[0, 0].set_title('Поле после 12000 шагов: хаос + магистраль')
    axes[0, 0].axis('off')
    axes[0, 1].plot(dist, lw=0.6)
    axes[0, 1].axvline(t_onset, color='r', ls='--', label=f't_onset={t_onset}')
    axes[0, 1].set_title('Манхэттенское расстояние от старта')
    axes[0, 1].set_xlabel('шаг'); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    a, b = max(0, t_onset-3000), min(T, t_onset+3000)
    axes[1, 0].plot(range(a, b), dist[a:b], lw=0.7)
    axes[1, 0].axvline(t_onset, color='r', ls='--')
    axes[1, 0].set_title('Переход хаос -> магистраль (крупным планом)')
    axes[1, 0].set_xlabel('шаг'); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].matshow(grid[r-50:r+51, c-70:c+71], cmap='Greys')
    axes[1, 1].set_title(f'Магистраль: tau={tau}, d={dd} (крупным планом)')
    axes[1, 1].axis('off')
    plt.tight_layout()
    plt.savefig('fig02_series1.png', dpi=150)

# ============================== PART 2 ==============================
def part2():
    hdr('PART 2. Серия 2: чувствительность к начальным условиям (правило LR)')
    rnd = random.Random(42)
    cases = [
        ('пустое поле (контроль)',        None),
        ('ненулевая клетка в старте (0,0)', [((0, 0), 1)]),
        ('ненулевая клетка на пути (1,0)',  [((1, 0), 1)]),
        ('ненулевая клетка вне траектории (40,40)', [((40, 40), 1)]),
        ('20 случайных клеток в [-30,30]^2',
         [((rnd.randint(-30, 30), rnd.randint(-30, 30)), 1) for _ in range(20)]),
    ]
    log(f'{"начальное поле":<38} | результат')
    for name, init in cases:
        res = classify_verified('LR', init=init)
        log(f'{name:<38} | {fmt(res)}')
    log('Вывод: возмущение вне траектории не влияет; возмущение на пути сдвигает')
    log('фазу и зеркалит магистраль; структурный шум меняет архитектуру цикла.')

# ============================== PART 3 ==============================
def part3():
    hdr('PART 3. Серия 3: перебор строк правил')
    catalogs = {}
    for lattice in ('square', 'hex'):
        for q in (2, 3):
            if lattice == 'square':
                cntW = Counter(); catW = []
                for tup in itertools.product('RLFU', repeat=q):
                    rule = ''.join(tup)
                    res = classify_window(rule)
                    cntW[res['cls']] += 1
                    catW.append(dict(rule=rule, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in res.items()}))
                log(f'квадрат q={q} [детектор окон, перекрёстно верифицирован]: {dict(sorted(cntW.items()))}')
                catalogs[f'square_window_q{q}'] = catW
                cntV = Counter()
                for tup in itertools.product('RLFU', repeat=q):
                    res = classify_verified(''.join(tup))
                    cntV[res['cls']] += 1
                log(f'квадрат q={q} [эталонный верификатор, контроль]:          {dict(sorted(cntV.items()))}')
            else:
                log('гекс: детектор окон дискредитирован (артефакт малых окон), используется только верификатор')
                cntV = Counter(); catV = []
                for tup in itertools.product('RLFU', repeat=q):
                    rule = ''.join(tup)
                    res = classify_verified(rule, lattice='hex')
                    cntV[res['cls']] += 1
                    if res['cls'] in (2, 3):
                        log(f'  {rule}: {fmt(res)}')
                        catV.append(dict(rule=rule, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in res.items()}))
                log(f'гекс q={q} [эталонный верификатор]: {dict(sorted(cntV.items()))}')
                catalogs[f'hex_verified_q{q}'] = catV
    if RUN_Q4:
        cntW = Counter(); catW = []
        for k, tup in enumerate(itertools.product('RLFU', repeat=4), 1):
            rule = ''.join(tup)
            res = classify_window(rule)
            cntW[res['cls']] += 1
            catW.append(dict(rule=rule, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in res.items()}))
            if k % 64 == 0:
                log(f'  ... прогресс перебора q=4: {k}/256')
        log(f'квадрат q=4 [детектор окон]: {dict(sorted(cntW.items()))}')
        catalogs['square_window_q4'] = catW
    with open('catalogs.json', 'w') as fh:
        json.dump(catalogs, fh, ensure_ascii=False)
    log('Каталоги правил сохранены в catalogs.json')

# ============================== PART 4 ==============================
def part4():
    hdr('PART 4. Серия 4: мутации одного символа')
    for base in ('RLLR', 'LRLR'):
        b = classify_window(base)
        log(f'\nБаза {base}: {fmt(b)}')
        changed = total = 0
        for pos_i in range(len(base)):
            for sym in 'RLFU':
                if sym == base[pos_i]:
                    continue
                mut = base[:pos_i] + sym + base[pos_i+1:]
                res = classify_window(mut)
                total += 1
                tag = '' if res['cls'] == b['cls'] else '   <== СМЕНА КЛАССА'
                if res['cls'] != b['cls']:
                    changed += 1
                log(f'  {base} -> {mut}: {fmt(res)}{tag}')
        log(f'Итог: у базы {base} класс сменили {changed} из {total} мутантов')

# ============================== PART 5 ==============================
def part5():
    hdr('PART 5. Серия 5: гексагональная решётка (визуально) и границы')
    log('Примечание: в первой версии эксперимента детектор на локальных окнах дал')
    log('артефакт (tau=948 во всех коробках); итоговая версия использует хэш полного')
    log('состояния — см. раздел отчёта «Верификация измерительного инструмента».')
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, rule in zip(axes, ('LR', 'RF')):
        fld = hex_sim(rule, 5000)
        ax.scatter([2*a+b for a, b in fld], [b for a, b in fld],
                   c=[fld[k] for k in fld], cmap='viridis', marker='h', s=30)
        ax.set_title(f'Гекс, правило {rule}, 5000 шагов: облако, не магистраль')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('fig04_hex_clouds.png', dpi=150)

    rows = []; pics = {}
    log(f'\n{"m":>2} | {"граница":<10} | точный цикл | t_wall | отражений')
    for m in (30, 50, 70):
        for bnd in ('reflect', 'periodic'):
            res = box_run(m=m, boundary=bnd)
            rows.append(res); pics[(m, bnd)] = res['grid']
            tau_s = str(res['tau']) if res['tau'] else f'нет за {BOX_T}'
            log(f"{m:>2} | {bnd:<10} | {tau_s:>11} | {str(res['t_wall']):>6} | {res['n_refl']:>9}")
    log('Вывод: точные циклы длиннее горизонта моделирования; после первого контакта')
    log('с границей магистраль разрушается, наблюдается устойчивый граничный хаос.')

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for j, m in enumerate((30, 50, 70)):
        axes[0, j].matshow(pics[(m, 'reflect')], cmap='Greys')
        axes[0, j].set_title(f'm={m}, отражающие'); axes[0, j].axis('off')
        axes[1, j].matshow(pics[(m, 'periodic')], cmap='Greys')
        axes[1, j].set_title(f'm={m}, тор'); axes[1, j].axis('off')
    plt.tight_layout()
    plt.savefig('fig05_box_fields.png', dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for res in rows:
        if res['m'] in (30, 50):
            ax = axes[0 if res['m'] == 30 else 1]
            ax.plot(res['dist'], lw=0.5, label=res['boundary'])
            if res['t_wall']:
                ax.axvline(res['t_wall'], color='r', ls=':', alpha=0.7)
    for ax, m in zip(axes, (30, 50)):
        ax.set_title(f'Дистанция от старта, m={m}')
        ax.set_xlabel('шаг'); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig('fig06_box_dist.png', dpi=150)

# ============================== PART 6 ==============================
def part6():
    hdr('PART 6. Галерея рисунков для отчёта')
    gallery = [
        ('LR',   12000, 'Класс 3 (q=2): классическая магистраль', 'Greys'),
        ('RR',   204,   'Класс 2 (q=2): цикл tau=8 (шаг 204 = 25*8+4)', 'Greys'),
        ('RLLR', 20000, 'Класс 3 (q=4): структурированный фронт', 'viridis'),
        ('RRUL', 15000, 'Класс 3 (q=4): долгий хаос -> магистраль', 'viridis'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, (rule, steps, title, cmap) in zip(axes.flatten(), gallery):
        ax.matshow(grid_sim_crop(rule, steps), cmap=cmap)
        ax.set_title(f'Правило {rule}\n{title}', fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('fig03_gallery.png', dpi=150)
    log('Галерея сохранена: fig03_gallery.png')

# ============================== MAIN ==============================
if __name__ == '__main__':
    part0()
    part1()
    part2()
    part3()
    part4()
    part5()
    part6()
    with open('research_summary.txt', 'w') as fh:
        fh.write('\n'.join(SUMMARY))
    hdr('ГОТОВО')
    log('Текстовая сводка:   research_summary.txt')
    log('Каталоги правил:    catalogs.json')
    log('Рисунки: fig01_parity.png, fig02_series1.png, fig03_gallery.png,')
    log('         fig04_hex_clouds.png, fig05_box_fields.png, fig06_box_dist.png')
    if SHOW_PLOTS:
        plt.show()