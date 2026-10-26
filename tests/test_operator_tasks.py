import unittest
import numpy as np
from operator_tasks import OperatorTask


class OperatorTaskTests(unittest.TestCase):
    def test_all_tasks_generate_finite_aligned_data(self):
        for name in ('antiderivative', 'diffusion_reaction', 'burgers'):
            task = OperatorTask(name, n_sensors=8, output_grid=6,
                                condition_scale=0.2, n_modes=4)
            conditions = task.sample_conditions(5, seed=1)
            responses = task.solve_batch(conditions)
            expected = 6 if name == 'antiderivative' else 36
            self.assertEqual(conditions.shape, (5, 8))
            self.assertEqual(responses.shape, (5, expected))
            self.assertTrue(np.all(np.isfinite(responses)))

    def test_low_viscosity_burgers_is_finite(self):
        task = OperatorTask(
            'burgers', n_sensors=21, output_grid=21,
            condition_scale=1.0, n_modes=5, viscosity=0.005)
        conditions = task.sample_conditions(4, seed=19)
        responses = task.solve_batch(conditions)
        self.assertTrue(np.all(np.isfinite(responses)))


if __name__ == '__main__':
    unittest.main()
