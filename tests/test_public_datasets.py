import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from real_world.datasets import load_public_dataset


class PublicDatasetAdapterTests(unittest.TestCase):
    def test_cfdbench_cases_become_conditioned_multihorizon_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'cylinder'
            for index in range(12):
                case = root / 'bc' / f'case{index:04d}'
                case.mkdir(parents=True)
                values = np.full((12, 8, 8), index, dtype=np.float32)
                np.save(case / 'u.npy', values)
                np.save(case / 'v.npy', values + 1)
                (case / 'case.json').write_text(json.dumps({
                    'vel_in': 0.1 + index * 0.01,
                    'density': 10.0,
                    'viscosity': 0.001,
                    'radius': 0.01,
                    'x_min': -0.06,
                    'x_max': 0.16,
                    'y_min': -0.06,
                    'y_max': 0.06,
                }), encoding='utf-8')
            data = load_public_dataset(dict(
                type='cfdbench_cylinder', path=str(root), subsets=['bc'],
                n_train=8, n_test=2, split_seed=0,
                history_frames=2, history_stride=1,
                target_offsets=[1, 3], window_stride=4,
                max_windows_per_case=2, spatial_stride=2))
            self.assertEqual(data.train_inputs.shape, (16, 9, 4, 4))
            self.assertEqual(data.test_targets.shape, (4, 4, 4, 4))
            self.assertEqual(data.metadata['train_cases'], 8)
            self.assertEqual(data.metadata['target_offsets'], [1, 3])

    def test_openfwi_annotations_pair_data_and_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_lines, test_lines = [], []
            for split, lines in (('train', train_lines), ('test', test_lines)):
                data_path = root / f'data_{split}.npy'
                model_path = root / f'model_{split}.npy'
                np.save(data_path, np.ones((4, 5, 20, 7), dtype=np.float32))
                np.save(model_path, np.ones((4, 1, 7, 7), dtype=np.float32))
                lines.append(str(data_path))
            train_annotation = root / 'train.txt'
            test_annotation = root / 'test.txt'
            train_annotation.write_text('\n'.join(train_lines), encoding='utf-8')
            test_annotation.write_text('\n'.join(test_lines), encoding='utf-8')
            data = load_public_dataset(dict(
                type='openfwi_flatvel_a',
                train_annotation=str(train_annotation),
                test_annotation=str(test_annotation),
                n_train=3, n_test=2, time_stride=2, receiver_stride=1))
            self.assertEqual(data.train_inputs.shape, (3, 5, 10, 7))
            self.assertEqual(data.train_targets.shape, (3, 1, 7, 7))

    def test_openfwi_curvefault_seis_vel_names_are_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seis_path = root / 'seis1.npy'
            vel_path = root / 'vel1.npy'
            np.save(seis_path, np.ones((4, 5, 20, 7), dtype=np.float32))
            np.save(vel_path, np.ones((4, 1, 7, 7), dtype=np.float32))
            train_annotation = root / 'train.txt'
            test_annotation = root / 'val.txt'
            train_annotation.write_text(str(seis_path), encoding='utf-8')
            test_annotation.write_text(str(seis_path), encoding='utf-8')
            data = load_public_dataset(dict(
                type='openfwi', dataset_name='OpenFWI CurveFault-B',
                train_annotation=str(train_annotation),
                test_annotation=str(test_annotation), n_train=3, n_test=2,
                time_stride=2, receiver_stride=1))
            self.assertEqual(data.train_inputs.shape, (3, 5, 10, 7))
            self.assertEqual(data.test_targets.shape, (2, 1, 7, 7))

    def test_openfwi_official_two_column_manifest_maps_to_local_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'CurveFault_B'
            data_root.mkdir()
            seis_path = data_root / 'seis6_1_0.npy'
            vel_path = data_root / 'vel6_1_0.npy'
            np.save(seis_path, np.ones((4, 5, 20, 7), dtype=np.float32))
            np.save(vel_path, np.ones((4, 1, 7, 7), dtype=np.float32))
            official_line = (
                '/projects/openfwi/CurveFault_B/seis6_1_0.npy '
                '/projects/openfwi/CurveFault_B/vel6_1_0.npy')
            train_annotation = root / 'train.txt'
            test_annotation = root / 'val.txt'
            train_annotation.write_text(official_line, encoding='utf-8')
            test_annotation.write_text(official_line, encoding='utf-8')
            data = load_public_dataset(dict(
                type='openfwi', dataset_name='OpenFWI CurveFault-B',
                data_root=str(data_root),
                train_annotation=str(train_annotation),
                test_annotation=str(test_annotation), n_train=3, n_test=2,
                time_stride=2, receiver_stride=1))
            self.assertEqual(data.train_inputs.shape, (3, 5, 10, 7))
            self.assertEqual(data.test_targets.shape, (2, 1, 7, 7))

    def test_pdebench_darcy_official_keys(self):
        try:
            import h5py
        except ImportError:
            self.skipTest('h5py is not installed')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'darcy.hdf5'
            with h5py.File(path, 'w') as stream:
                stream['nu'] = np.ones((12, 8, 8), dtype=np.float32)
                stream['tensor'] = np.arange(
                    12 * 1 * 8 * 8, dtype=np.float32).reshape(12, 1, 8, 8)
            data = load_public_dataset(dict(
                type='pdebench_darcy', path=str(path), n_train=8,
                n_test=2, split_seed=0, spatial_stride=2))
            self.assertEqual(data.train_inputs.shape, (8, 1, 4, 4))
            self.assertEqual(data.train_targets.shape, (8, 1, 4, 4))

    def test_pdebench_shallow_water_group_format(self):
        try:
            import h5py
        except ImportError:
            self.skipTest('h5py is not installed')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'swe.h5'
            with h5py.File(path, 'w') as stream:
                for index in range(12):
                    group = stream.create_group(f'{index:04d}')
                    group['data'] = np.full(
                        (3, 8, 8, 2), index, dtype=np.float32)
            data = load_public_dataset(dict(
                type='pdebench_shallow_water', path=str(path), n_train=8,
                n_test=2, split_seed=0, spatial_stride=2,
                input_time_indices=[0], target_time_indices=[1, 2]))
            self.assertEqual(data.train_inputs.shape, (8, 2, 4, 4))
            self.assertEqual(data.test_targets.shape, (2, 4, 4, 4))
            self.assertEqual(data.metadata['target_time_indices'], [1, 2])

    def test_pdebench_reaction_diffusion_group_format(self):
        try:
            import h5py
        except ImportError:
            self.skipTest('h5py is not installed')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reaction_diffusion.h5'
            with h5py.File(path, 'w') as stream:
                for index in range(12):
                    group = stream.create_group(f'{index:04d}')
                    group['data'] = np.full(
                        (4, 8, 8, 2), index, dtype=np.float32)
            data = load_public_dataset(dict(
                type='pdebench_reaction_diffusion', path=str(path),
                n_train=8, n_test=2, split_seed=0, spatial_stride=2,
                input_time_index=0, target_time_index=-1))
            self.assertEqual(data.train_inputs.shape, (8, 2, 4, 4))
            self.assertEqual(data.test_targets.shape, (2, 2, 4, 4))


if __name__ == '__main__':
    unittest.main()
