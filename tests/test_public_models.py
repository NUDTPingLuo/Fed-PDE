import unittest

import numpy as np


try:
    import torch
    from real_world.models import make_model
except (ImportError, RuntimeError):
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class PublicModelTests(unittest.TestCase):
    def test_fno_preserves_grid_shape(self):
        model = make_model(
            dict(type='fno2d', width=8, modes=4, layers=2),
            (2, 16, 16), (3, 16, 16))
        self.assertEqual(tuple(model(torch.randn(2, 2, 16, 16)).shape),
                         (2, 3, 16, 16))

    def test_inversionnet_maps_waveform_to_velocity_grid(self):
        model = make_model(
            dict(type='inversionnet_lite', width=8),
            (5, 100, 70), (1, 70, 70))
        self.assertEqual(tuple(model(torch.randn(2, 5, 100, 70)).shape),
                         (2, 1, 70, 70))

    def test_public_training_returns_selected_checkpoints(self):
        from real_world.training import fedavg_train

        rng = np.random.RandomState(4)
        inputs = rng.randn(6, 1, 8, 8).astype(np.float32)
        targets = rng.randn(6, 1, 8, 8).astype(np.float32)
        cfg = dict(
            device='cpu', batch_size=2, optimizer='sgd', momentum=0.9,
            lr=1e-3,
            local_steps=1, n_rounds=2,
            model=dict(type='fno2d', width=4, modes=2, layers=2))
        _, fed_history = fedavg_train(
            [inputs[:3], inputs[3:]], [targets[:3], targets[3:]],
            cfg, seed=3, checkpoint_rounds=[0, 1, 2])
        self.assertEqual(set(fed_history), {0, 1, 2})


if __name__ == '__main__':
    unittest.main()
