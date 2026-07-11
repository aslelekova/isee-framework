import unittest

import numpy as np

from isee import data, normalise


class TestData(unittest.TestCase):
    def setUp(self):
        self.ms = data.load()

    def test_shape_and_flagships(self):
        self.assertEqual(self.ms.X.shape, (len(self.ms), 10))
        self.assertEqual(int(self.ms.flagship.sum()), 4)

    def test_gap_indicators_are_inverted(self):
        j = self.ms.ids.index("AUS-Li")
        self.assertAlmostEqual(self.ms.X[j, 8], 100.0 - 91.0)

    def test_subset_preserves_alignment(self):
        sub = self.ms.subset(self.ms.flagship)
        self.assertEqual(len(sub), 4)
        self.assertEqual(sub.X.shape, (4, 10))


class TestNormalise(unittest.TestCase):
    def test_minmax_bounds(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 0.0]])
        N = normalise.minmax(X)
        self.assertTrue((N >= 0).all() and (N <= 1).all())
        self.assertAlmostEqual(N[:, 0].min(), 0.0)
        self.assertAlmostEqual(N[:, 0].max(), 1.0)

    def test_minmax_constant_column(self):
        N = normalise.minmax(np.array([[5.0], [5.0], [5.0]]))
        self.assertFalse(np.isnan(N).any())

    def test_dimension_scores_shape(self):
        ms = data.load()
        S = normalise.dimension_scores(normalise.minmax(ms.X))
        self.assertEqual(S.shape, (len(ms), 5))
        self.assertTrue((S >= 0).all() and (S <= 1).all())


if __name__ == "__main__":
    unittest.main()
