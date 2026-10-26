import unittest
import numpy as np
from util.model import CartesianDeepONet, average_params
from util.training import _step_batches, fedavg_train, gradient_dissimilarity


class FedAvgWeightingTests(unittest.TestCase):
    @staticmethod
    def spec():
        return dict(
            coordinates=np.linspace(0, 1, 4)[:, None],
            width=3, depth=2, activation='relu')

    def test_identical_clients_have_zero_gradient_dissimilarity(self):
        x = np.array([[0.0], [1.0]])
        y = np.array([[0.0, 0.2, 0.5, 1.0], [0.0, 0.1, 0.4, 0.8]])
        value = gradient_dissimilarity([x, x], [y, y], self.spec(), init_seed=4)
        self.assertAlmostEqual(value, 0.0)

    def test_sample_weighted_aggregation(self):
        params = [{'x': np.array([1.0])}, {'x': np.array([5.0])}]
        self.assertAlmostEqual(float(average_params(params, [1, 3])['x'][0]), 4.0)

    def test_unequal_client_sizes_train(self):
        x = [np.zeros((3, 1)), np.ones((9, 1))]
        y = [np.zeros((3, 4)), np.ones((9, 4))]
        params = fedavg_train(
            x, y, 1, 1, 0.01, 'sgd', self.spec(), init_seed=0)
        self.assertEqual(params['branch_W0'].shape, (1, 3))
        self.assertEqual(params['trunk_W0'].shape, (1, 3))

    def test_fedavg_can_return_selected_round_states(self):
        x = [np.zeros((3, 1)), np.ones((5, 1))]
        y = [np.zeros((3, 4)), np.ones((5, 4))]
        final, history = fedavg_train(
            x, y, 3, 1, 0.01, 'sgd', self.spec(), init_seed=0,
            checkpoint_rounds=[0, 1, 3])
        self.assertEqual(set(history), {0, 1, 3})
        self.assertEqual(final['branch_W0'].shape, history[3]['branch_W0'].shape)

    def test_local_steps_draw_exactly_one_batch_per_step(self):
        batches = list(_step_batches(
            n_samples=7, batch_size=3, steps=5,
            rng=np.random.RandomState(5)))
        self.assertEqual(len(batches), 5)
        self.assertTrue(all(len(batch) == 3 for batch in batches))
        self.assertTrue(all(len(np.unique(batch)) == 3 for batch in batches))

    def test_deeponet_gradient_matches_finite_difference(self):
        model = CartesianDeepONet(
            2, np.array([[0.0], [0.5], [1.0]]), width=2, depth=2,
            activation='tanh', seed=3)
        x = np.array([[0.2, -0.1], [0.4, 0.3]])
        y = np.array([[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]])
        _, gradients = model._grads(x, y)
        key, index, epsilon = 'branch_W0', (0, 0), 1e-6
        original = model.params[key][index]
        model.params[key][index] = original + epsilon
        plus = np.mean((model.forward(x)[0] - y) ** 2)
        model.params[key][index] = original - epsilon
        minus = np.mean((model.forward(x)[0] - y) ** 2)
        model.params[key][index] = original
        numerical = (plus - minus) / (2.0 * epsilon)
        self.assertAlmostEqual(float(gradients[key][index]), float(numerical), places=6)


if __name__ == '__main__':
    unittest.main()
