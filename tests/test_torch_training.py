import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class TorchTrainingTests(unittest.TestCase):
    def test_tiny_torch_backend_end_to_end(self):
        from util.training import (
            eval_params, fedavg_train, gradient_dissimilarity,
            relative_parameter_divergence,
        )

        rng = np.random.RandomState(3)
        client_x = [rng.randn(3, 4), rng.randn(5, 4)]
        client_u = [rng.randn(3, 3), rng.randn(5, 3)]
        spec = dict(
            coordinates=np.linspace(0, 1, 3)[:, None], width=4, depth=2,
            activation='tanh', backend='torch', device='cpu', batch_size=2,
            mixed_precision=False)
        gradient = gradient_dissimilarity(client_x, client_u, spec, 11)
        federated = fedavg_train(
            client_x, client_u, 2, 1, 1e-3, 'adam', spec, 11)
        _, history = fedavg_train(
            client_x, client_u, 2, 1, 1e-3, 'adam', spec, 11,
            checkpoint_rounds=[0, 1, 2])
        error = eval_params(federated, client_x[0], client_u[0], spec)

        self.assertTrue(np.isfinite(gradient))
        self.assertTrue(np.isfinite(error))
        self.assertEqual(set(history), {0, 1, 2})
        self.assertTrue(np.isfinite(
            relative_parameter_divergence(federated, federated)))
        self.assertEqual(
            relative_parameter_divergence(federated, federated), 0.0)


if __name__ == '__main__':
    unittest.main()
