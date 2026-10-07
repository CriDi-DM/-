import hashlib
import itertools
import numpy as np
from collections import Counter

TURN4 = {'L': 3, 'R': 1, 'U': 2, 'F': 0}
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN6 = {'R': 1, 'L': 5, 'U': 3, 'F': 0}
DIRS6 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

def classify_verified(rule, T=30000, lattice='square', tau_max=6000, wait=4000):
    TURN, DIRS = (TURN4, DIRS4) if lattice == 'square' else (TURN6, DIRS6)
    nD = 4 if lattice == 'square' else 6
    q = len(rule)
    field = {}
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
        dr, dc = DIRS[d]
        r += dr; c += dc
        if (r, c) not in visited: visited.add((r, c)); nvis += 1
        dirs[t - 1] = d
        pos[t] = (r, c)
        bb[0] = min(bb[0], r); bb[1] = max(bb[1], r)
        bb[2] = min(bb[2], c); bb[3] = max(bb[3], c)
        area = (bb[1]-bb[0]+1) * (bb[3]-bb[2]+1)
        if area != prev_area: prev_area, last_grow = area, t
        if not hashing and t - last_grow >= wait:
            hashing = True
        if hashing:
            key = hashlib.md5(repr((r, c, d, sorted(field.items()))).encode()).digest()
            if key in seen2:
                return dict(cls=2, tau=t - seen2[key], s0=seen2[key], d=(0, 0), nvis=nvis)
            seen2[key] = t
    # скан периодов траектории (без локальных окон)
    for tau in range(1, min(tau_max, T // 3) + 1):
        k = min(T - tau, max(3 * tau, 3000))
        if np.array_equal(dirs[T - k:T - tau], dirs[T - k + tau:T]):
            j = T - k - 1
            while j >= 0 and dirs[j] == dirs[j + tau]:
                j -= 1
            s0 = j + 2
            p0, p1 = pos[s0 - 1], pos[s0 - 1 + tau]
            dd = (int(p1[0] - p0[0]), int(p1[1] - p0[1]))
            return dict(cls=2 if dd == (0, 0) else 3, tau=tau, s0=s0, d=dd, nvis=nvis)
    return dict(cls=1 if hashing else 4, tau=None, s0=None, d=None, nvis=nvis)

if __name__ == '__main__':
    res = classify_verified('LR', lattice='square')
    print('SANITY square LR:', res)
    print('(ожидаем cls=3, tau=104, s0=10112, d=(2, 2))')
    for lattice in ('square', 'hex'):
        for q in (2, 3):
            cnt = Counter()
            print(f'\n===== {lattice}, q={q}: верифицированная классификация =====')
            for tup in itertools.product('RLFU', repeat=q):
                rule = ''.join(tup)
                r = classify_verified(rule, lattice=lattice)
                cnt[r['cls']] += 1
                if r['cls'] in (2, 3):
                    print(rule, r, flush=True)
            print(f'ИТОГ {lattice} q={q}: {dict(sorted(cnt.items()))}')