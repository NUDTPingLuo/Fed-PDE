import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sb_cfo_main import build_plan, parse_args


class SBCFOPlanTests(unittest.TestCase):
    def test_adam_defaults_and_explicit_adam_agree(self):
        plans = build_plan(parse_args(['--task', 'all', '--dry-run']))
        self.assertEqual(len(plans), 2)
        self.assertEqual([p['viscosity'] for p in plans], [0.005, None])
        self.assertTrue(all(p['alpha_values'] == [.01] for p in plans))
        self.assertTrue(all(p['methods'] == ['fedavg', 'sb_cfo'] for p in plans))
        self.assertTrue(all(p['training']['min_lr'] == .0001 for p in plans))
        self.assertTrue(all(p['training']['lr_schedule'] == 'cosine' for p in plans))
        self.assertTrue(all(p['training']['local_steps'] == 5 for p in plans))
        self.assertTrue(all(p['training']['participation_rate'] == 1. for p in plans))
        self.assertTrue(all(p['training']['n_rounds'] == 1000 for p in plans))
        self.assertTrue(all(p['training']['momentum'] == 0 for p in plans))
        self.assertTrue(all(p['training']['optimizer'] == 'adam' for p in plans))
        self.assertTrue(all(p['training']['lr'] == 0.001 for p in plans))
        self.assertTrue(all(Path(p['output_dir']).name == 'adam' for p in plans))
        adam = build_plan(parse_args(['--task', 'darcy', '--optimizer', 'adam']))[0]
        self.assertEqual(adam, plans[-1])
        self.assertIn('sb_cfo', adam['output_dir'])

    def test_sgd_ablation_overrides_adam_learning_rate_and_output(self):
        plans = build_plan(parse_args(['--task', 'all', '--optimizer', 'sgd']))
        self.assertTrue(all(p['training']['optimizer'] == 'sgd' for p in plans))
        self.assertTrue(all(p['training']['lr'] == 0.01 for p in plans))
        self.assertTrue(all(p['training']['momentum'] == 0.9 for p in plans))
        self.assertTrue(all(p['training']['nesterov'] for p in plans))
        self.assertTrue(all(p['training']['min_lr'] == 0.001 for p in plans))
        self.assertTrue(all(Path(p['output_dir']).name ==
                            'sgd_nesterov_lr0p01_min0p001_m0p9' for p in plans))

    def test_explicit_learning_rate_and_sgd_momentum_are_preserved(self):
        for optimizer in ('adam', 'sgd'):
            cfg = build_plan(parse_args([
                '--task', 'darcy', '--optimizer', optimizer, '--lr', '0.002']))[0]
            self.assertEqual(cfg['training']['lr'], 0.002)
        cfg = build_plan(parse_args([
            '--task', 'darcy', '--optimizer', 'sgd', '--momentum', '0.9']))[0]
        self.assertEqual(cfg['training']['momentum'], 0.9)
        self.assertEqual(cfg['training']['lr'], 0.01)

    def test_smoke_is_isolated_and_has_multiple_local_steps(self):
        cfg = build_plan(parse_args(['--task', 'darcy', '--smoke']))[0]
        self.assertEqual(cfg['training']['local_steps'], 2)
        self.assertIn('smoke', Path(cfg['output_dir']).parts)
        self.assertEqual(cfg['round_checkpoints'], [0, 1, 2])

    def test_invalid_arguments(self):
        for args in (['--alphas', '0'], ['--seeds', '1', '1'],
                     ['--lr', 'nan'], ['--refresh-every', '0'],
                     ['--samples-per-client-bin', '-1'], ['--rounds', '0'],
                     ['--viscosities', '0.1'],
                     ['--optimizer', 'adam', '--momentum', '0.9']):
            with self.assertRaises(ValueError):
                build_plan(parse_args(['--task', 'burgers'] + args))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class SBCFOCoreTests(unittest.TestCase):
    def setUp(self):
        import torch
        from sb_cfo.core import ClientShard
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.model = torch.nn.Linear(2, 1)
        rng = np.random.RandomState(4)
        self.clients = []
        for labels in (np.array([0, 0, 0, 1]), np.array([0, 1, 1, 1, 1, 1])):
            x = rng.randn(len(labels), 2).astype(np.float32)
            y = (x[:, :1] + labels[:, None]).astype(np.float32)
            self.clients.append(ClientShard(x, y, labels))
        self.training = dict(device='cpu', batch_size=99, optimizer='sgd', lr=0.01,
                             momentum=0., local_steps=2, n_rounds=2, log_every=10)
        self.control = dict(n_bins=2, strength=1., refresh_every=1, samples_per_client_bin=0)

    def snapshot(self, **kwargs):
        from sb_cfo.core import bin_gradient_snapshot
        return bin_gradient_snapshot(self.model, self.clients, 2, 2, 'cpu', **kwargs)

    def train(self, method, training=None, control=None):
        from sb_cfo.core import train
        return train(copy.deepcopy(self.model), self.clients,
                     self.clients[0].inputs, self.clients[0].targets,
                     training or self.training, control or self.control,
                     12, method, [0, 1, 2])

    def test_exact_prototypes_use_bin_sample_weights(self):
        import torch
        from sb_cfo.core import ClientShard, mean_gradient, correction_vectors
        snapshot = self.snapshot()
        pooled = ClientShard(np.concatenate([c.inputs for c in self.clients]),
                             np.concatenate([c.targets for c in self.clients]),
                             np.concatenate([c.labels for c in self.clients]))
        for b in range(2):
            expected = mean_gradient(self.model, pooled, np.flatnonzero(pooled.labels == b), 3, 'cpu')
            torch.testing.assert_close(snapshot['prototypes'][b], expected)
        q = correction_vectors(snapshot['counts'], snapshot['prototypes'])
        p = torch.tensor([.4, .6])
        torch.testing.assert_close(p @ q, torch.zeros(q.shape[1]), atol=1e-6, rtol=0)
        self.assertEqual(snapshot['examples'], 10)

    def test_ideal_bin_means_cancel_client_bias_and_identical_histograms_give_zero(self):
        import torch
        from sb_cfo.core import correction_vectors
        counts = np.array([[9, 1], [1, 9]])
        prototypes = torch.tensor([[2., -1.], [-4., 3.]])
        q = correction_vectors(counts, prototypes)
        gradients = torch.tensor(counts / 10, dtype=torch.float32) @ prototypes
        expected = prototypes.mean(0).expand(2, -1)
        torch.testing.assert_close(gradients + q, expected)
        torch.testing.assert_close(correction_vectors(np.array([[2, 4], [3, 6]]), prototypes),
                                   torch.zeros(2, 2))

    def test_capped_sampling_retains_actual_counts_and_is_reproducible(self):
        import torch
        a = self.snapshot(samples_per_client_bin=1, seed=101)
        b = self.snapshot(samples_per_client_bin=1, seed=101)
        self.assertEqual(a['examples'], 4)
        np.testing.assert_array_equal(a['counts'], [[3, 1], [1, 5]])
        torch.testing.assert_close(a['prototypes'], b['prototypes'])

    def test_zero_strength_is_identical_to_fedavg_with_persistent_adam(self):
        import torch
        training = dict(self.training, optimizer='adam', lr=0.001, batch_size=2)
        base, weights = self.train('fedavg', training)
        controlled, other = self.train('sb_cfo', training, dict(self.control, strength=0.))
        for name in weights:
            torch.testing.assert_close(weights[name], other[name], rtol=0, atol=0)
        self.assertEqual([r['test_error'] for r in base], [r['test_error'] for r in controlled])
        self.assertGreater(controlled[-1]['prototype_examples'], 0)
        self.assertEqual(base[-1]['prototype_examples'], 0)
        self.assertEqual(base[-1]['optimizer_examples'], controlled[-1]['optimizer_examples'])

    def test_one_full_batch_sgd_step_preserves_aggregated_first_order_update(self):
        import torch
        training = dict(self.training, local_steps=1, n_rounds=1)
        a, weights = self.train('fedavg', training)
        b, other = self.train('sb_cfo', training)
        for name in weights:
            torch.testing.assert_close(weights[name], other[name], atol=2e-7, rtol=1e-6)
        self.assertEqual(b[-1]['prototype_age_rounds'], 1)

    def test_fno_complex_gradients_and_corrections(self):
        import torch
        from real_world.models import FNO2d
        from sb_cfo.core import real_vector, add_correction, mean_gradient, ClientShard
        model = FNO2d(1, 1, width=2, modes=2, layers=1)
        rng = np.random.RandomState(1)
        shard = ClientShard(rng.randn(2, 1, 8, 8).astype(np.float32),
                            rng.randn(2, 1, 8, 8).astype(np.float32), np.array([0, 1]))
        gradient = mean_gradient(model, shard, np.array([0, 1]), 2, 'cpu')
        add_correction(model, -gradient)
        residual = real_vector(p.grad for p in model.parameters())
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-7, rtol=0)

    def test_refresh_cadence_and_independent_full_diagnostics(self):
        records, _ = self.train('sb_cfo', control=dict(
            self.control, refresh_every=2, samples_per_client_bin=1))
        self.assertEqual(records[-1]['prototype_refreshes'], 1)
        self.assertEqual(records[-1]['prototype_examples'], 4)
        self.assertEqual(records[-1]['diagnostic_examples'], 30)

    def test_fedavg_matches_existing_public_trainer(self):
        import torch
        from real_world.models import make_model
        from real_world.training import fedavg_train
        from sb_cfo.core import ClientShard, train
        rng = np.random.RandomState(15)
        clients = [ClientShard(rng.randn(n, 1, 8, 8).astype(np.float32),
                               rng.randn(n, 1, 8, 8).astype(np.float32),
                               np.arange(n) % 2) for n in (4, 6)]
        model_cfg = dict(type='fno2d', width=2, modes=2, layers=1)
        # Both entry points must use the same decaying round schedule.
        training = dict(self.training, optimizer='adam', lr=0.001, min_lr=0.0001, batch_size=2)
        old = fedavg_train([c.inputs for c in clients], [c.targets for c in clients],
                          dict(training, model=model_cfg), 12)
        torch.manual_seed(12)
        model = make_model(model_cfg, (1, 8, 8), (1, 8, 8))
        _, new = train(model, clients, clients[0].inputs, clients[0].targets,
                       training, self.control, 12, 'fedavg', [0, 2])
        for name in old:
            torch.testing.assert_close(new[name], old[name], rtol=1e-5, atol=2e-7)

    def test_fedavg_matches_existing_deeponet_trainer(self):
        import torch
        from util.torch_training import fedavg_train, TorchCartesianDeepONet
        from sb_cfo.core import train
        spec = dict(coordinates=np.array([[0.]], dtype=np.float32),
                    width=4, depth=2, activation='relu', device='cpu', batch_size=2)
        old = fedavg_train([c.inputs for c in self.clients],
                          [c.targets for c in self.clients],
                          2, 2, 0.001, 'adam', spec, 12)
        model = TorchCartesianDeepONet(2, spec, 12)
        training = dict(self.training, optimizer='adam', lr=0.001, min_lr=0.0001, batch_size=2)
        _, new = train(model, self.clients, self.clients[0].inputs,
                       self.clients[0].targets, training, self.control,
                       12, 'fedavg', [0, 2])
        for name in old:
            torch.testing.assert_close(new[name], torch.from_numpy(old[name]),
                                       rtol=1e-5, atol=2e-7)

    def test_paired_metrics_and_single_seed_ci(self):
        from sb_cfo.runner import paired_metrics, summarize
        rows = [dict(method=m, condition=c, seed=0, round=2, test_error=e)
                for m, c, e in [('fedavg', 'iid', .1), ('sb_cfo', 'iid', .11),
                                ('fedavg', 'alpha0.01', .2), ('sb_cfo', 'alpha0.01', .15)]]
        paired_metrics(rows)
        self.assertAlmostEqual(rows[-1]['improvement_vs_fedavg_pp'], 5)
        self.assertAlmostEqual(rows[-1]['excess_vs_fedavg_iid_pp'], 5)
        self.assertAlmostEqual(rows[-1]['excess_vs_own_iid_pp'], 4)
        summary = summarize(rows, 2)
        self.assertTrue(np.isnan(summary[-1]['test_error_ci95_low']))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class SBCFORunnerTests(unittest.TestCase):
    def test_burgers_smoke_resume_and_config_mismatch(self):
        from sb_cfo import runner
        with tempfile.TemporaryDirectory() as directory:
            cfg = build_plan(parse_args([
                '--task', 'burgers', '--viscosities', '0.01', '--smoke',
                '--device', 'cpu', '--output-dir', directory]))[0]
            rows = runner.run_experiment(cfg)
            self.assertEqual(len(rows), 12)  # 2 methods x 2 conditions x 3 checkpoints
            directory = Path(cfg['output_dir'])
            self.assertEqual({p.name for p in directory.iterdir()},
                             {'rounds.csv', 'summary.csv', 'metadata.json'})
            metadata = json.loads((directory / 'metadata.json').read_text())
            self.assertTrue(metadata['complete'])
            self.assertEqual(metadata['protocol']['learning_rate_schedule']['name'], 'cosine')
            self.assertTrue(all(r['learning_rate'] == cfg['training']['min_lr']
                                for r in rows if r['round'] == 2))
            with patch.object(runner, 'train', side_effect=AssertionError('should skip')):
                self.assertEqual(len(runner.run_experiment(cfg)), 12)
            changed = copy.deepcopy(cfg)
            changed['training']['lr'] *= 2
            with self.assertRaisesRegex(ValueError, 'different settings'):
                runner.run_experiment(changed)
            changed = copy.deepcopy(cfg)
            changed['training']['min_lr'] *= 2
            with self.assertRaisesRegex(ValueError, 'different settings'):
                runner.run_experiment(changed)
            changed = copy.deepcopy(cfg)
            changed['training']['lr_schedule'] = 'constant'
            with self.assertRaisesRegex(ValueError, 'lr_schedule must be cosine'):
                runner.run_experiment(changed)
            # Simulate interruption between completed method runs.
            partial = runner.read_rows(directory / 'rounds.csv')
            partial = [r for r in partial if not (
                r['method'] == 'sb_cfo' and r['condition'] == 'alpha0.1')]
            runner.write_rows(directory / 'rounds.csv', partial)
            with patch.object(runner, 'train', wraps=runner.train) as training:
                self.assertEqual(len(runner.run_experiment(cfg)), 12)
                self.assertEqual(training.call_count, 1)


if __name__ == '__main__':
    unittest.main()
