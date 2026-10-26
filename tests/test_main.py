import unittest
import os

import main


class PaperExperimentCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = main.load_experiment_config(
            'config/paper_experiments.yaml')

    def test_default_plan_sizes(self):
        self.assertEqual(
            len(main.study_plan(self.settings, 'formal')), 3)
        self.assertEqual(
            len(main.study_plan(self.settings, 'clients')), 3)
        self.assertEqual(
            len(main.study_plan(self.settings, 'viscosity')), 4)
        self.assertEqual(
            len(main.study_plan(self.settings, 'optimizer')), 3)
        with self.assertRaises(ValueError):
            main.study_plan(self.settings, 'clients', values=[10])
        with self.assertRaises(ValueError):
            main.study_plan(self.settings, 'viscosity', values=[0.02])

    def test_publication_rounds_seeds_and_client_override(self):
        item = main.study_plan(
            self.settings, 'clients', ['burgers'], [20])[0]
        cfg = main.prepare_run_config(self.settings, item)
        self.assertEqual(cfg['n_rounds'], 1000)
        self.assertEqual(cfg['local_steps'], 5)
        self.assertEqual(cfg['n_seeds'], 5)
        self.assertEqual(cfg['seeds'], [0, 1, 42, 999, 2026])
        self.assertEqual(cfg['n_clients'], 20)
        self.assertEqual(cfg['min_client_samples'], 16)
        self.assertEqual(
            cfg['output_dir'],
            os.path.join('results', 'ablation_clients', 'burgers'))
        self.assertEqual(
            cfg['round_checkpoints'],
            [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000])

    def test_viscosity_overrides(self):
        viscosity_item = main.study_plan(
            self.settings, 'viscosity', values=[0.05])[0]
        viscosity = main.prepare_run_config(
            self.settings, viscosity_item)
        self.assertAlmostEqual(viscosity['viscosity'], 0.05)
        self.assertEqual(viscosity['n_clients'], 10)
        self.assertEqual(
            viscosity['output_dir'],
            os.path.join('results', 'ablation_viscosity', 'burgers'))
        self.assertEqual(
            viscosity['alpha_values'], [10.0, 1.0, 0.1, 0.01])
        self.assertNotIn('iid_reference_alpha', viscosity)

    def test_federated_output_name_is_compact(self):
        item = main.study_plan(
            self.settings, 'formal', ['antiderivative'])[0]
        cfg = main.prepare_run_config(self.settings, item, smoke=True)
        self.assertEqual(
            cfg['output_prefix'],
            'antiderivative_formal_k10_s1_r1_smoke')
        self.assertNotIn('dirichlet_wor', cfg['output_prefix'])

    def test_publication_profiles_use_torch_cuda_auto(self):
        item = main.study_plan(
            self.settings, 'formal', ['diffusion_reaction'])[0]
        cfg = main.prepare_run_config(self.settings, item)
        self.assertEqual(cfg['training_backend'], 'torch')
        self.assertEqual(cfg['training_device'], 'auto')
        self.assertEqual(cfg['training_batch_size'], 16)
        forced = main.prepare_run_config(
            self.settings, item, device='cuda:0')
        self.assertEqual(forced['training_device'], 'cuda:0')

    def test_expected_rows_distinguish_complete_checkpoints(self):
        formal_item = main.study_plan(
            self.settings, 'formal', ['antiderivative'])[0]
        formal = main.prepare_run_config(self.settings, formal_item)
        expected = len(formal['alpha_values'])
        self.assertEqual(formal['n_clients'], 10)
        self.assertIn('_formal_k10_', formal['output_prefix'])
        self.assertEqual(
            formal['output_dir'],
            os.path.join('results', 'formal', 'antiderivative'))
        self.assertEqual(main.expected_result_rows(formal), expected)
        client_item = main.study_plan(
            self.settings, 'clients', ['burgers'], [20])[0]
        clients = main.prepare_run_config(self.settings, client_item)
        self.assertEqual(
            main.expected_result_rows(clients),
            len(clients['alpha_values']))

    def test_sgd_optimizer_ablation_uses_nesterov(self):
        item = main.study_plan(
            self.settings, 'optimizer', ['burgers'], ['sgd'])[0]
        cfg = main.prepare_run_config(self.settings, item)
        self.assertEqual(cfg['optimizer'], 'sgd')
        self.assertEqual(cfg['momentum'], 0.9)
        self.assertTrue(cfg['nesterov'])
        self.assertEqual(cfg['lr'], 0.01)
        self.assertEqual(cfg['lr_schedule'], 'cosine')
        self.assertEqual(cfg['min_lr'], 0.001)
        self.assertEqual(cfg['n_clients'], 10)
        self.assertIn(
            '_optimizer_sgd_lr0p01_m0p9_nesterov_k10_', cfg['output_prefix'])
        self.assertEqual(
            cfg['output_dir'],
            os.path.join('results', 'ablation_optimizer', 'burgers'))

    def test_complete_result_requires_seed_level_rows(self):
        import csv
        import os
        import tempfile

        item = main.study_plan(
            self.settings, 'formal', ['antiderivative'])[0]
        cfg = main.prepare_run_config(self.settings, item, smoke=True)
        expected = main.expected_result_rows(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'result_alpha.csv')
            stem = path[:-len('_alpha.csv')]
            for target, count in ((path, expected),
                                  (stem + '_alpha_seed_raw.csv',
                                   expected * cfg['n_seeds']),
                                  (stem + '_alpha_round_seed_raw.csv',
                                   expected * 2)):
                with open(target, 'w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['row'])
                    writer.writeheader()
                    writer.writerows({'row': index} for index in range(count))
            self.assertFalse(main.completed_result(path, cfg))
            seed_path = stem + '_alpha_seed_raw.csv'
            with open(seed_path, 'w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=[
                        'row', 'client_bin_counts', 'iid_client_bin_counts'])
                writer.writeheader()
                writer.writerows({
                    'row': index, 'client_bin_counts': '1;0|0;1',
                    'iid_client_bin_counts': '1;0|0;1'
                } for index in range(expected * cfg['n_seeds']))
            round_path = stem + '_alpha_round_seed_raw.csv'
            with open(round_path, 'w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=['row', 'iid_error'])
                writer.writeheader()
                writer.writerows({
                    'row': index, 'iid_error': 0.1
                } for index in range(expected * 2))
            iid_path = stem + '_iid.csv'
            with open(iid_path, 'w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=['partition', 'test_error'])
                writer.writeheader()
                writer.writerows({
                    'partition': 'iid', 'test_error': 0.1
                } for _ in range(cfg['n_seeds'] * 2))
            import json
            with open(stem + '_metadata.json', 'w') as stream:
                json.dump({'config': cfg}, stream)
            self.assertTrue(main.completed_result(path, cfg))
            self.assertFalse(main.completed_result(path, dict(cfg, min_lr=0.0002)))


if __name__ == '__main__':
    unittest.main()
