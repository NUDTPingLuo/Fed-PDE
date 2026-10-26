import unittest

import numpy as np

from util.partitioning import (
    cluster_solution_responses,
    emd_histogram_distance,
    mean_pairwise_transport,
    pairwise_euclidean,
    partition_clients,
    partition_clients_iid,
)


class PartitioningTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        conditions = rng.randn(80, 3)
        self.responses = np.c_[
            conditions[:, 0] + conditions[:, 1],
            conditions[:, 1] ** 2,
        ]

    def test_solution_clustering_is_complete_and_reproducible(self):
        first = cluster_solution_responses(self.responses, 4, seed=2)
        second = cluster_solution_responses(self.responses, 4, seed=2)
        self.assertTrue(np.array_equal(first.labels, second.labels))
        self.assertEqual(set(first.labels), set(range(4)))
        self.assertEqual(first.response_cost.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(first.response_cost), 0.0))
        self.assertGreater(first.response_normalizer, 0.0)
        self.assertGreaterEqual(first.quantization_error, 0.0)

    def test_partition_is_disjoint_exhaustive_and_quantity_heterogeneous(self):
        bins = cluster_solution_responses(self.responses, 4, seed=2)
        indices, target, realized = partition_clients(
            bins.labels, 4, alpha=0.1, minimum=3, seed=3)
        combined = np.concatenate(indices)
        sizes = np.asarray([len(index) for index in indices])
        self.assertTrue(np.array_equal(np.sort(combined), np.arange(len(bins.labels))))
        self.assertEqual(len(np.unique(combined)), len(bins.labels))
        self.assertTrue(np.all(sizes >= 3))
        self.assertGreater(np.std(sizes), 0.0)
        self.assertTrue(np.allclose(target.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(realized.sum(axis=1), 1.0))
        self.assertGreaterEqual(
            mean_pairwise_transport(realized, bins.response_cost), 0.0)

    def test_extreme_dirichlet_repairs_empty_clients_without_duplicates(self):
        labels = np.repeat(np.arange(4), 20)
        indices, _, _ = partition_clients(
            labels, 10, alpha=0.01, minimum=2, seed=4)
        combined = np.concatenate(indices)
        self.assertTrue(all(len(index) >= 2 for index in indices))
        self.assertTrue(np.array_equal(np.sort(combined), np.arange(80)))

    def test_exact_iid_is_stratified_balanced_and_without_replacement(self):
        labels = np.concatenate([
            np.full(23, 0), np.full(17, 1),
            np.full(29, 2), np.full(14, 3)])
        indices, target, realized = partition_clients_iid(
            labels, 4, minimum=16, seed=9)
        combined = np.concatenate(indices)
        sizes = np.asarray([len(index) for index in indices])
        counts = np.stack([
            np.bincount(labels[index], minlength=4) for index in indices])
        self.assertTrue(np.array_equal(np.sort(combined), np.arange(len(labels))))
        self.assertEqual(len(np.unique(combined)), len(labels))
        self.assertLessEqual(int(sizes.max() - sizes.min()), 1)
        self.assertTrue(np.all(sizes >= 16))
        self.assertTrue(np.all(np.ptp(counts, axis=0) <= 1))
        expected = np.bincount(labels) / len(labels)
        self.assertTrue(np.allclose(target, np.tile(expected, (4, 1))))
        self.assertTrue(np.allclose(realized.sum(axis=1), 1.0))

    def test_transport_matches_known_solution_and_does_not_mutate_inputs(self):
        left = np.array([2.0, 0.0, 0.0])
        right = np.array([0.0, 1.0, 3.0])
        originals = left.copy(), right.copy()
        cost = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :])
        self.assertAlmostEqual(emd_histogram_distance(left, right, cost), 1.75)
        np.testing.assert_array_equal(left, originals[0])
        np.testing.assert_array_equal(right, originals[1])
        left.flags.writeable = False
        right.flags.writeable = False
        self.assertAlmostEqual(emd_histogram_distance(left, right, cost), 1.75)

    def test_transport_handles_highs_presolve_infeasibility_regression(self):
        # The original full-constraint/default-presolve LP reports HiGHS
        # Status 8 on SciPy 1.17.1, although outer(left, right) is feasible.
        rng = np.random.RandomState(123)
        left = 10.0 ** rng.uniform(-10, -6, 10)
        left[rng.randint(10)] = 1.0
        right = 10.0 ** rng.uniform(-10, -6, 10)
        right[rng.randint(10)] = 1.0
        left /= left.sum()
        right /= right.sum()
        cost = pairwise_euclidean(rng.randn(10, 4))
        distance = emd_histogram_distance(left, right, cost)
        self.assertTrue(np.isfinite(distance))
        self.assertGreaterEqual(distance, 0.0)
        self.assertLessEqual(distance, float(np.sum(
            np.outer(left, right) * cost)) + 1e-8)
        self.assertAlmostEqual(
            distance, emd_histogram_distance(right, left, cost.T), places=7)

    def test_transport_preserves_small_positive_masses(self):
        left = np.array([1.0 - 1e-8, 1e-8, 0.0])
        right = np.array([1.0 - 2e-8, 0.0, 2e-8])
        cost = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :])
        # W1 on the unit-spaced line is the L1 distance between the CDFs.
        expected = float(np.sum(np.abs(np.cumsum(left - right)[:-1])))
        self.assertAlmostEqual(
            emd_histogram_distance(left, right, cost), expected, places=10)

    def test_transport_on_sparse_dirichlet_targets_and_realized_histograms(self):
        # Shallow-water bin sizes from the default 800-case training split.
        labels = np.repeat(np.arange(10), [76, 87, 85, 91, 117, 48, 110, 77, 19, 90])
        cost = np.abs(np.arange(10)[:, None] - np.arange(10)[None, :]) / 9
        for seed in (0, 1, 42, 999, 2026):
            with self.subTest(seed=seed):
                _, target, realized = partition_clients(
                    labels, 10, alpha=0.01, minimum=16, seed=30000 + seed)
                for histograms in (target, realized):
                    original = histograms.copy()
                    distance = mean_pairwise_transport(histograms, cost)
                    self.assertTrue(np.isfinite(distance))
                    self.assertGreaterEqual(distance, 0.0)
                    np.testing.assert_array_equal(histograms, original)

    def test_transport_rejects_invalid_histograms_and_costs(self):
        cost = np.eye(2)
        for histogram in ([0.0, 0.0], [-1.0, 2.0], [np.nan, 1.0],
                          [np.inf, 1.0], [], [[0.5, 0.5]]):
            with self.subTest(histogram=histogram):
                with self.assertRaises(ValueError):
                    emd_histogram_distance(histogram, [0.5, 0.5], cost)
                with self.assertRaises(ValueError):
                    emd_histogram_distance([0.5, 0.5], histogram, cost)
        for invalid_cost in (np.ones((3, 3)), [[0, -1], [1, 0]],
                             [[0, np.nan], [1, 0]]):
            with self.subTest(cost=invalid_cost):
                with self.assertRaises(ValueError):
                    emd_histogram_distance([0.5, 0.5], [0.5, 0.5], invalid_cost)


if __name__ == '__main__':
    unittest.main()
