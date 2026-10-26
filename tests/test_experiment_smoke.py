import unittest
import numpy as np
from unittest.mock import patch

import experiment


class ExperimentSmokeTests(unittest.TestCase):
    @staticmethod
    def config(task):
        return dict(
            experiment_profile='test', task=task, n_sensors=6, output_grid=5,
            n_modes=3, condition_scale=0.2, diffusion=0.01, reaction=0.01,
            viscosity=0.1, n_bins=3, sampling_pool_size=50,
            alpha_values=[0.2],
            transport_samples=6,
            min_client_samples=2, n_clients=2,
            model_width=4, model_depth=2, model_activation='relu',
            optimizer='sgd', momentum=0.9, lr=0.01,
            local_steps=1, n_rounds=1,
            seeds=[0], n_seeds=1, n_test=6, seed_base=7,
            round_checkpoints=[0, 1],
            output_dir='results', output_prefix='smoke')

    def test_all_operator_tasks_end_to_end(self):
        with patch.object(experiment, 'save_results'):
            for task in ('antiderivative', 'diffusion_reaction', 'burgers'):
                rows = experiment.run_experiment(self.config(task))
                self.assertEqual(len(rows), 1)
                self.assertEqual(
                    {row['method'] for row in rows}, {'solution_dirichlet'})
                self.assertTrue(all(row['distance_mean'] >= 0 for row in rows))
                self.assertTrue(all(np.isfinite(
                    row['response_quantization_error_mean']) for row in rows))
                self.assertTrue(all('drift_ci95_low' in row for row in rows))

    def test_iid_fedavg_reference_is_cached_once_per_seed(self):
        cfg = self.config('burgers')
        original = experiment.fedavg_train
        calls = []
        saved_seed_rows = []
        saved_round_seed_rows = []
        saved_iid_rows = []

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        def capture(_rows, _cfg, _metadata, seed_rows=None,
                    round_seed_rows=None, iid_rows=None):
            saved_seed_rows[:] = list(seed_rows or [])
            saved_round_seed_rows[:] = list(round_seed_rows or [])
            saved_iid_rows[:] = list(iid_rows or [])

        with patch.object(experiment, 'fedavg_train', side_effect=counted), \
                patch.object(experiment, 'save_results', side_effect=capture):
            experiment.run_experiment(cfg)
        self.assertEqual(
            len(calls), (len(cfg['alpha_values']) + 1) * cfg['n_seeds'])
        self.assertEqual(
            len(saved_seed_rows), len(cfg['alpha_values']) * cfg['n_seeds'])
        self.assertEqual({row['seed'] for row in saved_seed_rows}, {0})
        self.assertTrue(all(row['momentum'] == 0.9 for row in saved_seed_rows))
        self.assertTrue(all(
            len(row['client_bin_counts'].split('|')) == cfg['n_clients']
            for row in saved_seed_rows))
        self.assertTrue(all(
            len(row['iid_client_bin_counts'].split('|')) == cfg['n_clients']
            for row in saved_seed_rows))
        self.assertTrue(all('central_error' not in row for row in saved_seed_rows))
        self.assertTrue(all('iid_weight_divergence' in row
                            for row in saved_seed_rows))
        expected_round_rows = (len(cfg['alpha_values']) * cfg['n_seeds']
                               * len(cfg['round_checkpoints']))
        self.assertEqual(len(saved_round_seed_rows), expected_round_rows)
        self.assertEqual(
            {int(row['round']) for row in saved_round_seed_rows}, {0, 1})
        final_rows = [row for row in saved_round_seed_rows
                      if int(row['round']) == cfg['n_rounds']]
        self.assertEqual(len({row['iid_error'] for row in final_rows}), 1)
        self.assertNotIn(100.0, {
            float(row['severity_value']) for row in saved_round_seed_rows})
        self.assertEqual(
            len(saved_iid_rows), cfg['n_seeds'] * len(cfg['round_checkpoints']))
        self.assertTrue(all(row['partition'] == 'iid' for row in saved_iid_rows))
        self.assertTrue(all('test_error' in row for row in saved_iid_rows))

    def test_metadata_keeps_compact_solution_bin_diagnostics(self):
        cfg = self.config('burgers')
        task = experiment.make_task(cfg)
        data = {0: experiment.prepare_seed_data(cfg, task, 0)}
        metadata = experiment._metadata(cfg, data)
        diagnostics = metadata['solution_binning']['0']
        self.assertEqual(len(diagnostics['bin_sizes']), cfg['n_bins'])
        self.assertNotIn('centroids', diagnostics)
        self.assertNotIn('labels', diagnostics)

if __name__ == '__main__':
    unittest.main()
