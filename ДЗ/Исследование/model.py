"""Стохастический рост на решётке; время = число реально занятых клеток."""
from dataclasses import dataclass
import numpy as np
from numba import njit

NEUMANN = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
MOORE = np.array([(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x], dtype=np.int64)


@dataclass(frozen=True)
class Experiment:
    key: str
    title: str
    kind: str = 'dla'
    neighbors: int = 4
    sticking: float = 1.0
    drift: float = 0.0
    wall: str = 'none'


EXPERIMENTS = [
    Experiment('eden4', 'Эдем · 4 соседа', 'eden'),
    Experiment('eden8', 'Эдем · 8 соседей', 'eden', 8),
    Experiment('dla', 'DLA · p = 1'),
    Experiment('slow', 'DLA · p = 0.15', sticking=0.15),
    Experiment('drift', 'DLA · дрейф вправо', drift=0.35),
    Experiment('reflect', 'Кольцо · отражение', wall='reflect'),
    Experiment('catalytic', 'Кольцо · нуклеация', wall='catalytic'),
]


def obstacle(size, wall):
    y, x = np.indices((size, size)) - size // 2
    r = np.hypot(y, x)
    # Разрыв 60 градусов направлен вправо; радиус и толщина в клетках.
    return (r >= 24) & (r <= 26) & (np.abs(np.arctan2(y, x)) >= np.pi / 6) if wall != 'none' else np.zeros((size, size), bool)


@njit(cache=True)
def _eden(size, n, seed, offsets):
    np.random.seed(seed)
    grid = np.zeros((size, size), np.int32)
    active = np.zeros((size, size), np.bool_)
    frontier = np.empty((size * size, 2), np.int64)
    c = size // 2
    grid[c, c] = 1
    count = 0
    for k in range(len(offsets)):
        y, x = c + offsets[k, 0], c + offsets[k, 1]
        frontier[count, 0], frontier[count, 1] = y, x
        active[y, x] = True
        count += 1
    for mass in range(2, n + 1):
        if count == 0:
            raise RuntimeError('Граница Эдема исчерпана')
        i = np.random.randint(count)
        y, x = frontier[i, 0], frontier[i, 1]
        count -= 1
        frontier[i] = frontier[count]
        active[y, x] = False
        grid[y, x] = mass
        for k in range(len(offsets)):
            yy, xx = y + offsets[k, 0], x + offsets[k, 1]
            if 0 <= yy < size and 0 <= xx < size and grid[yy, xx] == 0 and not active[yy, xx]:
                frontier[count, 0], frontier[count, 1] = yy, xx
                count += 1
                active[yy, xx] = True
    return grid


@njit(cache=True)
def _dla(size, n, seed, p, drift, wall, catalytic, max_steps, max_attempts):
    np.random.seed(seed)
    grid = np.zeros((size, size), np.int32)
    visits = np.zeros((size, size), np.int64)
    c = size // 2
    grid[c, c] = 1
    radius = 0.0
    mass, attempts, escaped, timed_out, steps = 1, 0, 0, 0, 0
    has_wall = np.any(wall)
    while mass < n:
        if attempts >= max_attempts:
            raise RuntimeError('Превышен лимит запусков DLA; увеличьте лимит или расширьте щель')
        launch = max(radius + 10, 36.0 if has_wall else 10.0)
        if launch >= c - 3:
            raise RuntimeError('Кластер слишком близко к границе; увеличьте размер поля')
        kill = min(2 * launch, c - 2.0)
        angle = np.random.uniform(0, 2 * np.pi)
        y = int(round(c + launch * np.sin(angle)))
        x = int(round(c + launch * np.cos(angle)))
        attempts += 1
        for _ in range(max_steps):
            if (y - c)**2 + (x - c)**2 > kill**2:
                escaped += 1
                break
            visits[y, x] += 1
            steps += 1
            touching = False
            for k in range(4):
                yy, xx = y + NEUMANN[k, 0], x + NEUMANN[k, 1]
                if grid[yy, xx] > 0 or (catalytic and wall[yy, xx]):
                    touching = True
                    break
            if touching and np.random.random() < p:
                mass += 1
                grid[y, x] = mass
                radius = max(radius, np.sqrt((y - c)**2 + (x - c)**2))
                break
            u = np.random.random()
            dy, dx = 0, 0
            if u < 0.25:
                dy = -1
            elif u < 0.5:
                dy = 1
            elif u < 0.5 + (1 + drift) / 4:
                dx = 1
            else:
                dx = -1
            yy, xx = y + dy, x + dx
            # Отклонённый шаг остаётся на месте: стенки не проницаемы,
            # а отказ прилипания не разрешает ход сквозь кристалл.
            if not wall[yy, xx] and grid[yy, xx] == 0:
                y, x = yy, xx
        else:
            timed_out += 1
    return grid, visits, np.array([attempts, escaped, timed_out, steps], dtype=np.int64)


def simulate(exp, n=1800, size=321, seed=2026, max_steps=100000, max_attempts=None):
    if size < 81 or size % 2 != 1 or not 2 <= n < size * size:
        raise ValueError('Нужно нечётное поле >=81 и 2 <= n < size²')
    if exp.kind not in ('eden', 'dla') or exp.neighbors not in (4, 8):
        raise ValueError('Неизвестная модель или окрестность')
    if not 0 < exp.sticking <= 1 or not -1 <= exp.drift <= 1:
        raise ValueError('Нужно 0 < p <= 1, -1 <= drift <= 1')
    if exp.wall not in ('none', 'reflect', 'catalytic'):
        raise ValueError('Неизвестный тип стенки')
    if not 0 <= seed < 2**32 or max_steps < 1 or (max_attempts is not None and max_attempts < 1):
        raise ValueError('Недопустимый seed или лимит')
    if exp.kind == 'eden' and exp.wall != 'none':
        raise ValueError('Стенка реализована только для DLA')
    if exp.kind == 'dla' and exp.neighbors != 4:
        raise ValueError('В этой реализации DLA использует 4 соседей')
    wall = obstacle(size, exp.wall)
    if exp.kind == 'eden':
        grid = _eden(size, n, seed, NEUMANN if exp.neighbors == 4 else MOORE)
        visits = np.zeros_like(grid, dtype=np.int64)
        counters = np.array([n - 1, 0, 0, 0], dtype=np.int64)
    else:
        grid, visits, counters = _dla(size, n, seed, exp.sticking, exp.drift, wall,
                                     exp.wall == 'catalytic', max_steps, max_attempts or n * 3000)
    return grid, wall, visits, counters


def growth_curve(grid, interval=100):
    n = int(grid.max())
    positions = np.empty((n, 2), float)
    y, x = np.nonzero(grid)
    positions[grid[y, x] - 1] = np.column_stack((y, x))
    mass = np.unique(np.r_[np.arange(100, n + 1, interval), n])
    radii = []
    for m in mass:
        points = positions[:m]
        radii.append(np.sqrt(np.mean(np.sum((points - points.mean(axis=0))**2, axis=1))))
    return mass, np.array(radii)


def metrics(grid):
    y, x = np.nonzero(grid)
    points = np.column_stack((y, x)).astype(float)
    centered = points - points.mean(axis=0)
    eig = np.linalg.eigvalsh(centered.T @ centered / len(points))
    rg = np.sqrt(eig.sum())
    rmax = np.hypot(y - grid.shape[0] // 2, x - grid.shape[1] // 2).max()
    mass, radii = growth_curve(grid)
    fit = mass >= max(200, len(points) // 4)
    if fit.sum() >= 3:
        a, b = np.polyfit(np.log(radii[fit]), np.log(mass[fit]), 1)
        residual = np.log(mass[fit]) - (a * np.log(radii[fit]) + b)
        r2 = 1 - np.sum(residual**2) / np.sum((np.log(mass[fit]) - np.log(mass[fit]).mean())**2)
    else:
        a, r2 = float('nan'), float('nan')
    return dict(n=len(points), rg=float(rg), rmax=float(rmax),
                filling=float(len(points) / (np.pi * rmax**2)),
                anisotropy=float((eig[1] - eig[0]) / eig.sum()),
                shift_x=float(x.mean() - grid.shape[1] // 2),
                d_eff=float(a), fit_r2=float(r2),
                outside=float(np.mean(np.hypot(y - grid.shape[0] // 2, x - grid.shape[1] // 2) > 26)))
