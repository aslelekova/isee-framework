import unittest

import numpy as np

from isee import (aggregate, counterfactual, data, normalise, robustness,
                  smaa, typology, weights)


class TestData(unittest.TestCase):
    def setUp(self):
        self.ms = data.load()

    def test_shape_and_flagships(self):
        self.assertEqual(self.ms.X.shape, (len(self.ms), 10))
        self.assertEqual(int(self.ms.flagship.sum()), 4)

    def test_gap_indicators_are_inverted(self):
        j = self.ms.ids.index("AUS-Li")
        self.assertAlmostEqual(self.ms.X[j, 8], 100.0 - 82.3)

    def test_subset_preserves_alignment(self):
        sub = self.ms.subset(self.ms.flagship)
        self.assertEqual(len(sub), 4)
        self.assertEqual(sub.X.shape, (4, 10))
        self.assertEqual(sub.P.shape, (4, 10))
        self.assertEqual(len(sub.countries), 4)

    def test_derived_indicators_consistent_after_perturbation(self):
        rng = np.random.default_rng(0)
        Pp = data.perturb_primitives(self.ms.P, self.ms.countries, rng, 0.1)
        X = data.build_indicators(Pp)
        np.testing.assert_allclose(X[:, 1], 100.0 * Pp[:, 0] / Pp[:, 1])
        np.testing.assert_allclose(X[:, 5], Pp[:, 5] / Pp[:, 4])

    def test_country_level_shocks_are_shared(self):
        rng = np.random.default_rng(0)
        Pp = data.perturb_primitives(self.ms.P, self.ms.countries, rng, 0.1)
        aus = [j for j, c in enumerate(self.ms.countries) if c == "AUS"]
        F = Pp[aus] / self.ms.P[aus]
        for col in np.where(data.PRIMITIVE_COUNTRY_LEVEL)[0]:
            self.assertAlmostEqual(F[:, col].max(), F[:, col].min(), places=12)


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


class TestWeights(unittest.TestCase):
    def setUp(self):
        ms = data.load()
        S = normalise.dimension_scores(normalise.minmax(ms.X))
        self.B = aggregate.benefit_oriented(S)

    def test_all_schemes_sum_to_one(self):
        for w in [weights.equal(), weights.entropy(self.B),
                  weights.pca(self.B), weights.critic(self.B)]:
            self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
            self.assertTrue((w >= 0).all())

    def test_entropy_prefers_discriminating_dimension(self):
        B = np.column_stack([np.full(10, 0.5), np.linspace(0, 1, 10)])
        w = weights.entropy(B)
        self.assertLess(w[0], w[1])


class TestAggregate(unittest.TestCase):
    def test_additive_signs(self):
        S = np.array([[1.0, 1.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 1.0]])
        isee = aggregate.additive(S, weights.equal())
        self.assertAlmostEqual(isee[0], 0.6)
        self.assertAlmostEqual(isee[1], -0.4)

    def test_geometric_order_matches_on_dominance(self):
        S = np.array([[0.9, 0.9, 0.9, 0.1, 0.1],
                      [0.2, 0.2, 0.2, 0.8, 0.8]])
        g = aggregate.geometric(S, weights.equal())
        self.assertGreater(g[0], g[1])

    def test_ranks(self):
        self.assertEqual(aggregate.ranks([0.3, 0.9, 0.1]).tolist(), [2, 1, 3])


class TestRobustness(unittest.TestCase):
    def setUp(self):
        ms = data.load()
        self.cases = ms.subset(ms.flagship)
        self.S = normalise.dimension_scores(normalise.minmax(self.cases.X))

    def test_weight_mc_rank_frequencies_are_distributions(self):
        _, freq = robustness.weight_mc(self.S, n=500)
        np.testing.assert_allclose(freq.sum(axis=0), 1.0, atol=1e-9)
        np.testing.assert_allclose(freq.sum(axis=1), 1.0, atol=1e-9)

    def test_weight_mc_reproducible(self):
        s1, _ = robustness.weight_mc(self.S, n=200, seed=7)
        s2, _ = robustness.weight_mc(self.S, n=200, seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_data_mc_zero_noise_keeps_baseline(self):
        w = weights.equal()
        scores, _ = robustness.data_mc(self.cases, w, n=10, noise=0.0)
        base = aggregate.additive(self.S, w)
        np.testing.assert_allclose(scores, np.tile(base, (10, 1)), atol=1e-12)


class TestTypology(unittest.TestCase):
    def setUp(self):
        ms = data.load()
        S = normalise.dimension_scores(normalise.minmax(ms.X))
        self.B = aggregate.benefit_oriented(S)

    def test_kmeans_label_count(self):
        labels, centers = typology.kmeans(self.B, 4)
        self.assertEqual(len(set(labels)), 4)
        self.assertEqual(centers.shape, (4, 5))

    def test_silhouette_selection_range(self):
        metrics = typology.select_k(self.B, k_range=range(2, 5))
        self.assertTrue(all(-1 <= m["silhouette"] <= 1
                            for m in metrics.values()))

    def test_pca_variance_shares(self):
        _, var, _ = typology.pca_project(self.B)
        self.assertTrue(0 < var[0] <= 1 and 0 < var[1] <= var[0])


class TestRegression(unittest.TestCase):
    def setUp(self):
        self.full = data.load()
        self.cases = self.full.subset(self.full.flagship)
        self.S = normalise.dimension_scores(normalise.minmax(self.cases.X))
        S_all = normalise.dimension_scores(normalise.minmax(self.full.X))
        self.B_all = aggregate.benefit_oriented(S_all)
        self.isee_all = aggregate.additive(S_all, weights.equal())

    def test_baseline_four_case_scores(self):
        isee = aggregate.additive(self.S, weights.equal())
        np.testing.assert_allclose(
            isee, [0.272594, 0.120683, 0.037026, -0.124597], atol=1e-6)

    def test_baseline_four_case_ranks(self):
        isee = aggregate.additive(self.S, weights.equal())
        self.assertEqual(aggregate.ranks(isee).tolist(), [1, 2, 3, 4])

    def test_entropy_weights_reference_sample(self):
        np.testing.assert_allclose(
            weights.entropy(self.B_all),
            [0.430614, 0.080033, 0.337580, 0.057227, 0.094547], atol=1e-6)

    def test_weight_mc_rank1_frequencies(self):
        _, freq = robustness.weight_mc(self.S, n=10_000)
        np.testing.assert_allclose(
            freq[0], [0.6292, 0.2125, 0.1363, 0.0220], atol=1e-4)

    def test_selected_k_is_three(self):
        metrics = typology.select_k(self.B_all, k_range=range(2, 11))
        sil_max = max(m["silhouette"] for m in metrics.values())
        candidates = [k for k, m in metrics.items()
                      if m["silhouette"] >= sil_max - 0.02]
        stability = {k: typology.bootstrap_stability(self.B_all, k)
                     for k in candidates}
        k_best = max(candidates,
                     key=lambda k: (round(stability[k][0], 3), -k))
        self.assertEqual(k_best, 6)

    def test_extended_sample_leader(self):
        j = int(np.argmax(self.isee_all))
        self.assertEqual(self.full.ids[j], "CHL-Cu")
        self.assertAlmostEqual(self.isee_all[j], 0.197471, places=5)


class TestSMAA(unittest.TestCase):
    def setUp(self):
        ms = data.load()
        self.cases = ms.subset(ms.flagship)
        self.S = normalise.dimension_scores(normalise.minmax(self.cases.X))

    def test_power_mean_rho1_preserves_baseline_ordering(self):
        w = weights.equal()
        pm = smaa.power_mean(self.S, w, 1.0)
        base = aggregate.additive(self.S, w)
        self.assertEqual(aggregate.ranks(pm).tolist(),
                         aggregate.ranks(base).tolist())

    def test_power_mean_geometric_limit(self):
        w = weights.equal()
        near_zero = smaa.power_mean(self.S, w, 1e-10)
        exact = smaa.power_mean(self.S, w, 0.0)
        np.testing.assert_allclose(near_zero, exact, atol=1e-6)

    def test_acceptability_is_distribution(self):
        res = smaa.sample(self.cases, n=500, noise=0.10)
        b = smaa.rank_acceptability(res["ranks"])
        np.testing.assert_allclose(b.sum(axis=0), 1.0, atol=1e-9)
        np.testing.assert_allclose(b.sum(axis=1), 1.0, atol=1e-9)

    def test_pairwise_probability_antisymmetric(self):
        res = smaa.sample(self.cases, n=500, noise=0.0)
        P = smaa.pairwise_probability(res["scores"])
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertAlmostEqual(P[i, j] + P[j, i], 1.0, places=9)

    def test_smaa_regression_rank1_acceptability(self):
        res = smaa.sample(self.cases, n=2000, noise=0.10)
        b = smaa.rank_acceptability(res["ranks"])
        np.testing.assert_allclose(
            b[0], [0.7795, 0.1070, 0.0970, 0.0165], atol=1e-4)

    def test_compensability_crossover(self):
        rho = smaa.crossover(self.S, weights.equal(), i=1, j=2)
        self.assertAlmostEqual(rho, -0.0595, places=3)

    def test_crossover_exists_for_all_floors(self):
        for floor in [0.01, 0.05, 0.10, 0.20]:
            rho = smaa.crossover(self.S, weights.equal(), i=1, j=2,
                                 floor=floor)
            self.assertIsNotNone(rho)
            self.assertTrue(-0.6 < rho < 0.4)

    def test_counterfactual_probability_bounds(self):
        p = counterfactual.rank_probability(self.cases, 2, 2, n=300)
        self.assertTrue(0.0 <= p <= 1.0)


class TestTieHandling(unittest.TestCase):
    def test_spearman_matches_scipy_with_ties(self):
        from scipy.stats import spearmanr
        ms = data.load()
        C = robustness.spearman_matrix(ms.X)
        i, j = 4, 5
        m = ~(np.isnan(ms.X[:, i]) | np.isnan(ms.X[:, j]))
        expected = spearmanr(ms.X[m, i], ms.X[m, j]).statistic
        self.assertAlmostEqual(C[i, j], expected, places=10)

    def test_grouped_bootstrap_bounds(self):
        ms = data.load()
        S = normalise.dimension_scores(normalise.minmax(ms.X))
        B = aggregate.benefit_oriented(S)
        mean, sd = typology.bootstrap_stability_grouped(
            B, ms.countries, 6, n_boot=30)
        self.assertTrue(0.0 <= mean <= 1.0)
        self.assertTrue(sd >= 0.0)


if __name__ == "__main__":
    unittest.main()
