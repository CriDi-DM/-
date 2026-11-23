import unittest
from dataclasses import replace
import numpy as np
from scipy.ndimage import label
from model import EXPERIMENTS, simulate, metrics, growth_curve, NEUMANN


class GrowthTests(unittest.TestCase):
    def test_all_rules_preserve_mass_and_walls(self):
        for exp in EXPERIMENTS:
            with self.subTest(exp=exp.key):
                grid, wall, visits, counts = simulate(exp, n=120, size=161, seed=41)
                np.testing.assert_array_equal(np.sort(grid[grid > 0]), np.arange(1, 121))
                self.assertTrue(np.all(grid[wall] == 0))
                self.assertTrue(np.all(visits[wall] == 0))
                if exp.kind == 'dla':
                    self.assertEqual(counts[0], 119 + counts[1] + counts[2])
                    self.assertEqual(counts[3], visits.sum())
                # Каждое присоединение должно иметь допустимого предшественника.
                offsets = NEUMANN if exp.neighbors == 4 else np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b])
                for y, x in zip(*np.where(grid > 1)):
                    self.assertTrue(any((0 < grid[y+a, x+b] < grid[y, x]) or
                                        (exp.wall == 'catalytic' and wall[y+a, x+b]) for a, b in offsets))
                if exp.neighbors == 4 and exp.wall != 'catalytic':
                    self.assertEqual(label(grid > 0)[1], 1)

    def test_reproducibility(self):
        for exp in (EXPERIMENTS[0], EXPERIMENTS[3]):
            a = simulate(exp, n=150, seed=10)
            b = simulate(exp, n=150, seed=10)
            for aa, bb in zip(a, b):
                np.testing.assert_array_equal(aa, bb)
            self.assertFalse(np.array_equal(a[0], simulate(exp, n=150, seed=11)[0]))

    def test_metrics_on_symmetric_square(self):
        grid = np.zeros((81, 81), np.int32)
        grid[35:46, 35:46] = np.arange(1, 122).reshape(11, 11)
        m = metrics(grid)
        self.assertEqual(m['n'], 121)
        self.assertAlmostEqual(m['rg'], np.sqrt(20))
        self.assertAlmostEqual(m['anisotropy'], 0)
        self.assertAlmostEqual(m['shift_x'], 0)
        mass, _ = growth_curve(grid)
        self.assertEqual(mass[-1], 121)

    def test_reject_invalid_inputs(self):
        for kwargs in ({'n': 1}, {'size': 80}, {'seed': -1}, {'max_steps': 0}):
            with self.assertRaises(ValueError):
                simulate(EXPERIMENTS[2], **kwargs)
        for exp in (replace(EXPERIMENTS[2], sticking=0), replace(EXPERIMENTS[2], drift=2),
                    replace(EXPERIMENTS[2], wall='unknown'), replace(EXPERIMENTS[2], neighbors=8)):
            with self.assertRaises(ValueError):
                simulate(exp)

    def test_limits_fail_explicitly(self):
        with self.assertRaises(RuntimeError):
            simulate(EXPERIMENTS[2], n=120, max_steps=1, max_attempts=1)


if __name__ == '__main__':
    unittest.main()
