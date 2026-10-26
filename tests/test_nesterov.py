"""Nesterov updates, optimizer propagation and isolated result contracts."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import experiment
import main
from util.model import CartesianDeepONet
from util.torch_training import _make_optimizer
from real_world.training import _optimizer
from sb_cfo.core import make_optimizer
from util.result_contract import matching_metadata
from sb_cfo_main import build_plan, parse_args


class NesterovTests(unittest.TestCase):
    def test_sb_cfo_explicit_plain_sgd_and_invalid_momentum(self):
        cfg = build_plan(parse_args(['--task', 'burgers', '--optimizer', 'sgd',
                                    '--no-nesterov', '--momentum', '0']))[0]
        self.assertFalse(cfg['training']['nesterov'])
        self.assertEqual(cfg['training']['momentum'], 0)
        with self.assertRaises(ValueError):
            build_plan(parse_args(['--task', 'burgers', '--optimizer', 'sgd',
                                   '--momentum', '0']))

    def test_numpy_updates_match_torch_with_persistent_momentum(self):
        model = CartesianDeepONet(1, np.array([[0.]]), 2, depth=2, seed=0)
        params = {k: torch.nn.Parameter(torch.tensor(v.copy(), dtype=torch.float64))
                  for k, v in model.params.items()}
        optimizer = torch.optim.SGD(params.values(), lr=.01, momentum=.9,
                                    nesterov=True)
        for step, lr in enumerate((.01, .0055, .001)):
            grads = {k: np.full_like(v, .2 * (step + 1)) for k, v in model.params.items()}
            with patch.object(model, '_grads', return_value=(0., grads)):
                model.step(None, None, lr, 'sgd', .9, True)
            optimizer.param_groups[0]['lr'] = lr
            for k, p in params.items():
                p.grad = torch.tensor(grads[k], dtype=torch.float64)
            optimizer.step()
            for k, p in params.items():
                np.testing.assert_allclose(model.params[k], p.detach().numpy(), atol=1e-12)

    def test_all_torch_factories_enable_nesterov(self):
        cfg = dict(optimizer='sgd', lr=.01, momentum=.9, nesterov=True)
        for factory in (lambda m: _make_optimizer(m, 'sgd', .01, .9, True),
                        lambda m: _optimizer(m, cfg), lambda m: make_optimizer(m, cfg)):
            opt = factory(torch.nn.Linear(1, 1))
            self.assertTrue(opt.param_groups[0]['nesterov'])
            self.assertEqual(opt.param_groups[0]['momentum'], .9)

    def test_all_ablation_tasks_train_iid_and_alpha_with_new_schedule(self):
        torch.set_num_threads(1)
        settings = main.load_experiment_config('config/paper_experiments.yaml')
        for item in main.study_plan(settings, 'optimizer'):
            with self.subTest(task=item['task']), tempfile.TemporaryDirectory() as directory:
                cfg = main.prepare_run_config(settings, item, smoke=True, device='cpu')
                cfg.update(n_rounds=3, round_checkpoints=[0, 1, 2, 3],
                           alpha_values=[.1], output_dir=directory, checkpoint=True)
                calls = []
                original = torch.optim.SGD.step
                def checked_step(opt, *args, **kwargs):
                    calls.append((opt.param_groups[0]['lr'], opt.param_groups[0]['nesterov']))
                    return original(opt, *args, **kwargs)
                with patch.object(torch.optim.SGD, 'step', checked_step):
                    experiment.run_experiment(cfg)
                self.assertTrue(calls)
                self.assertTrue(all(flag for _, flag in calls))
                np.testing.assert_allclose(sorted(set(rate for rate, _ in calls)), [.001, .0055, .01])
                for suffix in ('_iid.csv', '_alpha_round_seed_raw.csv'):
                    with (Path(directory) / (cfg['output_prefix'] + suffix)).open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(rows[0]['current_learning_rate'], '')
                    np.testing.assert_allclose([float(r['current_learning_rate']) for r in rows[1:]],
                                               [.01, .0055, .001])
                meta = Path(directory) / (cfg['output_prefix'] + '_metadata.json')
                self.assertTrue(json.loads(meta.read_text())['config']['nesterov'])
                self.assertTrue(matching_metadata(meta, cfg))
                self.assertFalse(matching_metadata(meta, dict(cfg, nesterov=False)))


if __name__ == '__main__':
    unittest.main()
