import itertools, json, sys
from collections import Counter

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN = {'L': 3, 'R': 1, 'U': 2, 'F': 0}

def classify(rule, T=30000, R=4, wait=4000, sym_thr=0.98):
    q = len(rule)
    field = {}
    r = c = 0; d = 0
    bb = [0, 0, 0, 0]; last_grow, last_area = 0, 1
    nvis = 1; visited = {(0, 0)}
    seen = {}; whist = [None]; prev = None; conf = 0

    for t in range(1, T + 1):
        s = field.get((r, c), 0)
        d = (d + TURN[rule[s]]) % 4
        s2 = (s + 1) % q
        if s2: field[(r, c)] = s2
        else:  field.pop((r, c), None)
        dr, dc = DIRS[d]; r += dr; c += dc
        if (r, c) not in visited: visited.add((r, c)); nvis += 1
        bb[0] = min(bb[0], r); bb[1] = max(bb[1], r)
        bb[2] = min(bb[2], c); bb[3] = max(bb[3], c)
        area = (bb[1]-bb[0]+1) * (bb[3]-bb[2]+1)
        if area != last_area: last_area, last_grow = area, t

        g = field.get
        w = (d, tuple(g((r+a, c+b), 0) for a in range(-R, R+1) for b in range(-R, R+1)))
        whist.append(w)
        if w in seen:
            t0, (p0r, p0c) = seen[w]
            tau, dd = t - t0, (r - p0r, c - p0c)
            conf = conf + 1 if prev == (tau, dd) else 1
            prev = (tau, dd)
            if conf >= 3:
                i = t - tau + 1
                while i - 1 >= 1 and whist[i-1] == whist[i-1+tau]:
                    i -= 1
                return dict(rule=rule, cls=2 if dd == (0, 0) else 3,
                            tau=tau, d=dd, t_onset=i, nvis=nvis)
        else:
            conf, prev = 0, None
        seen[w] = (t, (r, c))

    if t - last_grow >= wait:
        return dict(rule=rule, cls=1, nvis=nvis, bbox=area)
    f = field
    if f:
        m1 = sum(1 for (i, j), s in f.items() if f.get((i, -j), -1) == s) / len(f)
        m2 = sum(1 for (i, j), s in f.items() if f.get((-i, j), -1) == s) / len(f)
        sym = max(m1, m2)
    else:
        sym = 1.0
    return dict(rule=rule, cls=5 if sym >= sym_thr else 4,
                nvis=nvis, bbox=area, sym=round(sym, 3))

def check():
    res = classify('LR', T=15000)
    ok = (res['cls'], res['tau'], res['d'], res['t_onset']) == (3, 104, (2, 2), 10112)
    print(res); print('check:', 'OK — реализации согласованы' if ok else 'MISMATCH!')

def enum(q, T):
    total = 4 ** q
    cnt = Counter(); catalog = []
    for k, tup in enumerate(itertools.product('RLFU', repeat=q), 1):
        res = classify(''.join(tup), T=T)
        cnt[res['cls']] += 1
        if res['cls'] in (2, 3, 5):
            catalog.append(res); print(res, flush=True)
        if k % 64 == 0:
            print(f'... прогресс {k}/{total}', flush=True)
    print(f'q={q}, T={T}, всего правил={total}')
    print('Распределение классов {1:огран., 2:период., 3:магистраль, 4:неогр.рост, 5:симметр.}:')
    print(dict(sorted(cnt.items())))
    with open(f'enum_q{q}.json', 'w') as fh:
        json.dump(catalog, fh, ensure_ascii=False, indent=1)
    print(f'Каталог сохранён в enum_q{q}.json')

if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if a == 'check':
        check()
    else:
        enum(int(a), int(sys.argv[2]) if len(sys.argv) > 2 else 30000)