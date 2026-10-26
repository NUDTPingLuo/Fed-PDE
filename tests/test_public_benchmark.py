import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from real_world.datasets import PublicPDEData

try:
    from real_world import benchmark
except (ImportError, RuntimeError):
    benchmark = None


@unittest.skipIf(benchmark is None, 'PyTorch is not installed')
class PublicBenchmarkTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(3)
        self.data = PublicPDEData(
            rng.randn(48, 1, 8, 8).astype(np.float32),
            rng.randn(48, 2, 8, 8).astype(np.float32),
            rng.randn(4, 1, 8, 8).astype(np.float32),
            rng.randn(4, 2, 8, 8).astype(np.float32), {})

    @staticmethod
    def config(directory):
        return dict(
            task='shallow_water', dataset=dict(type='pdebench_shallow_water'),
            model=dict(type='fno2d', width=4, modes=2, layers=2),
            partition=dict(n_bins=3, alpha_values=[1.0, 0.01],
                           min_client_samples=2, clustering_seed=42),
            federation=dict(n_clients=4),
            training=dict(device='cpu', batch_size=2, optimizer='sgd',
                          lr=0.01, local_steps=1, n_rounds=1),
            evaluation=dict(seeds=[0], round_checkpoints=[0, 1]),
            output=dict(dir=directory, prefix='test_public_benchmark'))

    def test_invalid_transport_is_detected_before_training(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(benchmark, 'load_public_dataset', return_value=self.data), \
                patch.object(benchmark, 'mean_pairwise_transport',
                             side_effect=ValueError('invalid transport')), \
                patch.object(benchmark, 'fedavg_train') as training:
            with self.assertRaisesRegex(ValueError, 'invalid transport'):
                benchmark.run_public_benchmark(self.config(directory))
            training.assert_not_called()

    def test_sparse_public_run_caches_distances_and_iid_reference(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(benchmark, 'load_public_dataset', return_value=self.data), \
                patch.object(benchmark, 'save_results') as save, \
                patch.object(benchmark, 'mean_pairwise_transport',
                             wraps=benchmark.mean_pairwise_transport) as distance, \
                patch.object(benchmark, 'fedavg_train',
                             wraps=benchmark.fedavg_train) as training:
            rows, seed_rows = benchmark.run_public_benchmark(self.config(directory))
            self.assertEqual(training.call_count, 3)  # IID once + two alphas.
            self.assertEqual(distance.call_count, 6)  # Two distances per split.
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(seed_rows), 2)
            self.assertTrue(all(np.isfinite(row['distance']) for row in seed_rows))
            self.assertTrue(all(np.isfinite(row['drift_mean']) for row in rows))
            _, _, metadata, _, round_rows, iid_rows = save.call_args[0]
            self.assertTrue(metadata['complete'])
            self.assertEqual(len(round_rows), 4)
            self.assertEqual(len(iid_rows), 2)
            self.assertTrue(all(row['partition'] == 'iid' for row in iid_rows))


if __name__ == '__main__':
    unittest.main()
