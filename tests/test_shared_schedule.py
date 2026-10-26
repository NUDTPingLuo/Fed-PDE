"""Check actual optimizer updates, not just YAML values."""
import unittest
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from util import training, torch_training
from real_world import training as public_training
from util.schedules import checkpoint_learning_rate
from util.io_utils import load_config
from real_world_main import load_task_config


class SharedScheduleTests(unittest.TestCase):
    def test_formal_torch_pipeline_writes_round_learning_rates(self):
        import main
        import experiment
        settings = main.load_experiment_config('config/paper_experiments.yaml')
        torch.set_num_threads(1)
        for item in main.study_plan(settings, 'formal'):
            with self.subTest(task=item['task']), tempfile.TemporaryDirectory() as directory:
                cfg = main.prepare_run_config(settings, item, smoke=True, device='cpu')
                cfg.update(n_rounds=3, round_checkpoints=[0, 1, 2, 3],
                           alpha_values=[.1], output_dir=directory, checkpoint=True)
                experiment.run_experiment(cfg)
                for suffix in ('_iid.csv', '_alpha_round_seed_raw.csv'):
                    path = Path(directory) / (cfg['output_prefix'] + suffix)
                    with path.open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(rows[0]['current_learning_rate'], '')
                    self.assertEqual(float(rows[1]['current_learning_rate']), .001)
                    self.assertAlmostEqual(float(rows[2]['current_learning_rate']), .00055)
                    self.assertEqual(float(rows[3]['current_learning_rate']), .0001)

    def test_all_configured_defaults(self):
        for task in ('antiderivative', 'diffusion_reaction', 'burgers'):
            cfg = load_config('config/config_' + task + '.yaml')
            self.assertEqual(cfg['training_batch_size'], 16)
            self.assertEqual(cfg['optimizer'], 'adam')
            self.assertEqual(checkpoint_learning_rate(cfg, 1), .001)
            self.assertEqual(checkpoint_learning_rate(cfg, 1000), .0001)
        for task in ('darcy', 'shallow_water', 'cfd_cylinder', 'openfwi'):
            cfg = load_task_config('config/real_' + task + '.yaml')['training']
            self.assertEqual(cfg['optimizer'], 'adam')
            self.assertEqual(checkpoint_learning_rate(cfg, 1), .001)
            self.assertEqual(checkpoint_learning_rate(cfg, 1000), .0001)
        self.assertIsNone(checkpoint_learning_rate(cfg, 0))

    def test_every_actual_step_uses_round_rate(self):
        torch.set_num_threads(1)
        x = [np.array([[.1], [.4], [.8]], dtype=np.float32)] * 2
        y = [v * 2 for v in x]
        spec = dict(coordinates=np.array([[0.]], dtype=np.float32), width=4,
                    depth=2, activation='relu', device='cpu', batch_size=2,
                    lr_schedule='cosine', min_lr=.0001, momentum=.9)
        expected = np.repeat([.001, .00055, .0001], 4)
        for optimizer in ('adam', 'sgd'):
            for backend in ('numpy', 'torch', 'public'):
                with self.subTest(optimizer=optimizer, backend=backend):
                    rates = []
                    if backend == 'numpy':
                        original = training.CartesianDeepONet.step
                        def step(model, inputs, targets, lr, *args):
                            rates.append(lr)
                            return original(model, inputs, targets, lr, *args)
                        with patch.object(training.CartesianDeepONet, 'step', step):
                            training.fedavg_train(x, y, 3, 2, .001, optimizer,
                                                 dict(spec, backend='numpy'), 0)
                    else:
                        cls = torch.optim.Adam if optimizer == 'adam' else torch.optim.SGD
                        original = cls.step
                        def step(opt, *args, **kwargs):
                            rates.append(opt.param_groups[0]['lr'])
                            return original(opt, *args, **kwargs)
                        with patch.object(cls, 'step', step):
                            if backend == 'torch':
                                torch_training.fedavg_train(x, y, 3, 2, .001,
                                                            optimizer, spec, 0)
                            else:
                                cfg = dict(spec, optimizer=optimizer, lr=.001,
                                           local_steps=2, n_rounds=3)
                                with patch.object(public_training, '_model',
                                                  return_value=torch.nn.Linear(1, 1)):
                                    public_training.fedavg_train(x, y, cfg, 0)
                    np.testing.assert_allclose(rates, expected, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
