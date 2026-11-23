"""Независимая проверка сохранённого комплекта; python verify_results.py."""
import csv
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.stats import t
from model import EXPERIMENTS, metrics, growth_curve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'


def main():
    meta = json.loads((OUT / 'metadata.json').read_text(encoding='utf-8'))
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    with (OUT / 'runs.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    expected = {(e.key, seed) for e in EXPERIMENTS for seed in meta['seeds']}
    assert len(rows) == len(expected) == meta['runs']
    assert {(r['experiment'], int(r['seed'])) for r in rows} == expected
    assert len(list(OUT.glob('*.npz'))) == len(expected)
    rules = {e.key: e for e in EXPERIMENTS}
    for row in rows:
        key, seed = row['experiment'], int(row['seed'])
        with np.load(OUT / f'{key}_{seed}.npz', allow_pickle=False) as z:
            grid, wall, visits, counters = (z[k] for k in ['birth', 'wall', 'visits', 'counters'])
            n = meta['particles_including_seed']
            assert grid.shape == (meta['size'], meta['size'])
            np.testing.assert_array_equal(np.sort(grid[grid > 0]), np.arange(1, n+1))
            assert np.all(grid[wall] == 0) and np.all(visits[wall] == 0)
            if rules[key].kind == 'dla':
                assert counters[0] == n - 1 + counters[1] + counters[2]
                assert counters[3] == visits.sum()
            if rules[key].neighbors == 4 and rules[key].wall != 'catalytic':
                assert label(grid > 0)[1] == 1
            for name, value in metrics(grid).items():
                np.testing.assert_allclose(value, float(row[name]), rtol=1e-10)
            mass, rg = growth_curve(grid)
            np.testing.assert_array_equal(z['mass'], mass)
            np.testing.assert_allclose(z['rg'], rg)
            for i, name in enumerate(['attempts', 'escaped', 'timeouts', 'walk_steps']):
                assert counters[i] == int(row[name])
    for exp in EXPERIMENTS:
        for key, saved in summary[exp.key].items():
            values = np.array([float(r[key]) for r in rows if r['experiment'] == exp.key])
            np.testing.assert_allclose(saved['mean'], values.mean())
            np.testing.assert_allclose(saved['sd'], values.std(ddof=1))
            np.testing.assert_allclose(saved['ci95'], t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    for name in ['atlas.png', 'statistics.png', 'transport.png', 'walls.png']:
        with Image.open(OUT / name) as image:
            assert min(image.size) > 500
            image.verify()
    with Image.open(OUT / 'growth.gif') as gif:
        assert gif.n_frames == 80
        for i in range(gif.n_frames):
            gif.seek(i)
            gif.load()
    report = (ROOT / 'Отчёт.md').read_text(encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)', report):
        if not link.startswith(('#', 'https://', 'http://')):
            assert (ROOT / link).is_file(), link
    print(f'OK: {len(rows)} runs, all metrics, confidence intervals, 4 PNG, 80 GIF frames and report links.')


if __name__ == '__main__':
    main()
