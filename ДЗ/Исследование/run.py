"""python run.py — все эксперименты, исходные данные, графики и GIF."""
import csv
import json
import platform
import time
from pathlib import Path
import importlib.metadata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import t

from model import EXPERIMENTS, simulate, metrics, growth_curve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
SIZE, N = 501, 4000
SEEDS = list(range(2026, 2036))
BG, FG, MUTED = '#0b1220', '#e5edf7', '#9aacc3'
COLORS = ['#4fd1c5', '#56a8ff', '#b99bff', '#ffce73', '#ff839c', '#61d5a0', '#f29567']
CMAP = LinearSegmentedColormap.from_list('birth', ['#2552a5', '#36c6d4', '#ffe39b'])
CMAP.set_bad(BG)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'figure.facecolor': BG, 'axes.facecolor': BG, 'axes.edgecolor': MUTED,
    'axes.labelcolor': FG, 'text.color': FG, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': MUTED, 'savefig.facecolor': BG,
    'legend.facecolor': BG, 'legend.edgecolor': MUTED})


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches='tight')
    plt.close(fig)


def cluster(ax, grid, wall, extent=None):
    ax.imshow(np.ma.masked_equal(grid, 0), origin='lower', cmap=CMAP, vmin=1, vmax=N,
              interpolation='nearest')
    rgba = np.zeros((*wall.shape, 4))
    rgba[wall] = [0.94, 0.35, 0.45, 1]
    ax.imshow(rgba, origin='lower', interpolation='nearest')
    if extent is not None:
        c = grid.shape[0] // 2
        ax.set_xlim(c-extent, c+extent)
        ax.set_ylim(c-extent, c+extent)
    ax.axis('off')


def interval(values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    ci = float(t.ppf(0.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {'mean': mean, 'sd': float(values.std(ddof=1)), 'ci95': ci}


def make_gif(models):
    fontpath = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(fontpath), 21) if fontpath.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(fontpath), 15) if fontpath.exists() else font
    chosen = ['eden4', 'dla', 'slow', 'drift']
    titles = {e.key: e.title for e in EXPERIMENTS}
    frames = []
    # Общий пространственный масштаб; номер кадра — масса, не физическое время.
    c, radius, tile = SIZE // 2, 165, 370
    for mass in np.unique(np.r_[1, np.linspace(40, N, 79).astype(int)]):
        frame = Image.new('RGB', (tile * 2, tile * 2 + 90), BG)
        draw = ImageDraw.Draw(frame)
        draw.text((20, 12), 'РОСТ ИЗ ЛОКАЛЬНЫХ ПРАВИЛ', fill=FG, font=font)
        draw.text((20, 43), f'N = {mass:04d} / {N}  |  одинаковая масса и масштаб  |  seed = {SEEDS[0]}', fill=MUTED, font=small)
        for i, key in enumerate(chosen):
            grid = models[key][0][c-radius:c+radius+1, c-radius:c+radius+1]
            rgba = CMAP(np.clip((grid - 1) / (N - 1), 0, 1), bytes=True)[:, :, :3]
            rgba[(grid == 0) | (grid > mass)] = [11, 18, 32]
            image = Image.fromarray(np.flipud(rgba)).resize((tile-35, tile-35), Image.Resampling.NEAREST)
            x, y = (i % 2) * tile, 85 + (i // 2) * tile
            frame.paste(image, (x+15, y + 24))
            draw.text((x+12, y), titles[key], fill=FG, font=small)
        frames.append(frame)
    frames[0].save(OUT / 'growth.gif', save_all=True, append_images=frames[1:],
                   duration=[80] * (len(frames)-1) + [1800], loop=0, optimize=False)


def main():
    OUT.mkdir(exist_ok=True)
    started = time.perf_counter()
    records, models, curves = [], {}, {}
    for exp in EXPERIMENTS:
        for seed in SEEDS:
            grid, wall, visits, counts = simulate(exp, n=N, size=SIZE, seed=seed)
            row = {'experiment': exp.key, 'seed': seed, **metrics(grid),
                   'attempts': int(counts[0]), 'escaped': int(counts[1]),
                   'timeouts': int(counts[2]), 'walk_steps': int(counts[3]),
                   'capture': (N-1)/int(counts[0])}
            records.append(row)
            mass, rg = growth_curve(grid)
            # Сохранены все реализации, а не только эффектная картинка.
            np.savez_compressed(OUT / f'{exp.key}_{seed}.npz', birth=grid, wall=wall,
                                visits=visits, counters=counts, mass=mass, rg=rg)
            curves.setdefault(exp.key, []).append(rg)
            if seed == SEEDS[0]:
                models[exp.key] = (grid, wall, visits)
            print(f'{exp.key:10s} seed={seed}: N={row["n"]}, Rg={row["rg"]:.2f}', flush=True)
    with (OUT / 'runs.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    keys = ['rg', 'filling', 'anisotropy', 'shift_x', 'd_eff', 'fit_r2', 'capture', 'outside']
    summary = {exp.key: {key: interval([r[key] for r in records if r['experiment'] == exp.key])
                         for key in keys} for exp in EXPERIMENTS}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for i, exp in enumerate(EXPERIMENTS):
        cluster(axes.flat[i], *models[exp.key][:2], extent=170)
        axes.flat[i].set_title(exp.title, color=COLORS[i], weight='bold', fontsize=12)
        m = metrics(models[exp.key][0])
        axes.flat[i].text(.03, .02, f'Rg = {m["rg"]:.1f}   A = {m["anisotropy"]:.2f}',
                          transform=axes.flat[i].transAxes, color=MUTED, fontsize=10)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.05, .85, '7 ПРАВИЛ\n10 ПОВТОРОВ\n4 000 ЧАСТИЦ', fontsize=24, weight='bold', va='top', linespacing=1.5)
    axes.flat[-1].text(.05, .15, 'Синий → бирюзовый → золотой:\nпорядок присоединения.\nКоралловый: неподвижная стенка.\nМасштаб всех панелей одинаков.', color=MUTED, linespacing=1.6)
    fig.suptitle('АТЛАС РОСТА  /  ОДНА РЕШЁТКА — РАЗНАЯ МОРФОЛОГИЯ', fontsize=21, weight='bold')
    fig.text(.5, .025, f'Иллюстративный запуск: seed {SEEDS[0]} · N = {N} · поле {SIZE} × {SIZE}', ha='center', color=MUTED)
    save(fig, 'atlas.png')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for i, exp in enumerate(EXPERIMENTS[:5]):
        array = np.array(curves[exp.key])
        mean = array.mean(axis=0)
        ci = t.ppf(.975, len(SEEDS)-1) * array.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
        axes[0].plot(mass, mean, color=COLORS[i], label=exp.title, lw=2)
        axes[0].fill_between(mass, mean-ci, mean+ci, color=COLORS[i], alpha=.13)
    axes[0].set(xscale='log', yscale='log', xlabel='Число занятых клеток N', ylabel='Радиус инерции Rg', title='Рост: среднее и 95% ДИ')
    axes[0].legend(fontsize=8)
    for ax, key, title in [(axes[1], 'filling', 'Заполнение охватывающего диска'), (axes[2], 'anisotropy', 'Анизотропия тензора инерции')]:
        for i, exp in enumerate(EXPERIMENTS):
            values = [r[key] for r in records if r['experiment'] == exp.key]
            s = summary[exp.key][key]
            ax.scatter(np.full(len(values), i)+np.linspace(-.14, .14, len(values)), values, s=18, color=COLORS[i], alpha=.6)
            ax.errorbar(i, s['mean'], yerr=s['ci95'], color=FG, capsize=4, marker='_', ms=14)
        ax.set_xticks(range(len(EXPERIMENTS)), [e.key for e in EXPERIMENTS], rotation=35)
        ax.set_title(title)
    for ax in axes:
        ax.grid(alpha=.13)
    fig.tight_layout()
    save(fig, 'statistics.png')

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, key in enumerate(['reflect', 'catalytic']):
        grid, wall, visits = models[key]
        cluster(axes[row, 0], grid, wall, extent=125)
        axes[row, 0].set_title('Отражающая стенка' if row == 0 else 'Стенка с нуклеацией')
        # Посещения на запущенную частицу — эмпирическая статистика траекторий, не концентрация.
        attempts = next(r['attempts'] for r in records if r['experiment'] == key and r['seed'] == SEEDS[0])
        im = axes[row, 1].imshow(np.log1p(visits / attempts), origin='lower', cmap='magma')
        axes[row, 1].contour(wall, levels=[.5], colors=['#ff839c'], linewidths=.7)
        c = SIZE // 2
        axes[row, 1].set(xlim=(c-125, c+125), ylim=(c-125, c+125), title='log(1 + посещения / запуски)')
        axes[row, 1].axis('off')
        fig.colorbar(im, ax=axes[row, 1], fraction=.04, pad=.02)
        cluster(axes[row, 2], grid, wall, extent=35)
        axes[row, 2].set_title('Камера и щель · увеличение')
    fig.suptitle('ПРЕПЯТСТВИЕ ≠ ЦЕНТР КРИСТАЛЛИЗАЦИИ', fontsize=21, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, 'walls.png')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, exp in enumerate(EXPERIMENTS[2:]):
        s = summary[exp.key]['capture']
        axes[0].bar(i, s['mean'], yerr=s['ci95'], color=COLORS[i+2], capsize=4)
    axes[0].set_xticks(range(5), [e.key for e in EXPERIMENTS[2:]], rotation=25)
    axes[0].set(title='Доля захваченных частиц', ylabel='(N − 1) / число запусков')
    for i, exp in enumerate(EXPERIMENTS[:5]):
        s = summary[exp.key]['shift_x']
        axes[1].errorbar(i, s['mean'], yerr=s['ci95'], color=COLORS[i], capsize=5, marker='o', ms=8)
    axes[1].axhline(0, color=MUTED, lw=1, ls='--')
    axes[1].set_xticks(range(5), [e.key for e in EXPERIMENTS[:5]], rotation=25)
    axes[1].set(title='Смещение центра массы', ylabel='x̄ − x₀, клетки')
    for ax in axes:
        ax.grid(axis='y', alpha=.13)
    fig.tight_layout()
    save(fig, 'transport.png')
    make_gif(models)
    tables = ['# Численные результаты', '', 'Все значения: среднее ± 95% t-интервал среднего, 10 независимых seed.', '',
              '| Модель | Rg | Заполнение | Анизотропия | Δx | D_eff |', '|---|---:|---:|---:|---:|---:|']
    for exp in EXPERIMENTS:
        cells = [f'{summary[exp.key][k]["mean"]:.3f} ± {summary[exp.key][k]["ci95"]:.3f}' for k in ['rg', 'filling', 'anisotropy', 'shift_x', 'd_eff']]
        tables.append('| ' + exp.title + ' | ' + ' | '.join(cells) + ' |')
    (OUT / 'tables.md').write_text('\n'.join(tables) + '\n', encoding='utf-8')
    meta = {'size': SIZE, 'particles_including_seed': N, 'seeds': SEEDS, 'runs': len(records),
            'max_steps_per_walker': 100000, 'max_attempts_per_run': N*3000,
            'python': platform.python_version(), 'platform': platform.platform(),
            'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'numba', 'matplotlib', 'Pillow', 'scipy']},
            'elapsed_seconds': round(time.perf_counter()-started, 2)}
    (OUT / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Done: {len(records)} runs; {meta["elapsed_seconds"]} seconds. Results: {OUT}', flush=True)


if __name__ == '__main__':
    main()
