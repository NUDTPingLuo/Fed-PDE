"""Adapters for public, application-oriented PDE datasets.

All adapters expose channel-first arrays ``[sample, channel, height, width]``.
The repository never downloads or redistributes third-party data; users point
the YAML configuration at files obtained from the official dataset provider.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PublicPDEData:
    train_inputs: np.ndarray
    train_targets: np.ndarray
    test_inputs: np.ndarray
    test_targets: np.ndarray
    metadata: dict


def _float_nchw(values, name):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 3:
        array = array[:, None]
    if array.ndim != 4:
        raise ValueError(f"{name} must resolve to [N,C,H,W], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _split(inputs, targets, n_train, n_test, seed):
    if len(inputs) != len(targets):
        raise ValueError("input and target sample counts differ")
    required = int(n_train) + int(n_test)
    if len(inputs) < required:
        raise ValueError(
            f"dataset has {len(inputs)} samples but {required} are requested")
    rng = np.random.RandomState(int(seed))
    chosen = rng.permutation(len(inputs))[:required]
    train = chosen[:int(n_train)]
    test = chosen[int(n_train):]
    return PublicPDEData(
        _float_nchw(inputs[train], 'train inputs'),
        _float_nchw(targets[train], 'train targets'),
        _float_nchw(inputs[test], 'test inputs'),
        _float_nchw(targets[test], 'test targets'),
        {},
    )


def _chosen_indices(n_available, n_train, n_test, seed):
    required = int(n_train) + int(n_test)
    if int(n_available) < required:
        raise ValueError(
            f"dataset has {n_available} samples but {required} are requested")
    return np.random.RandomState(int(seed)).permutation(int(n_available))[:required]


def _split_selected(inputs, targets, n_train):
    """Split arrays that are already shuffled and limited to requested cases."""
    split = int(n_train)
    return PublicPDEData(
        _float_nchw(inputs[:split], 'train inputs'),
        _float_nchw(targets[:split], 'train targets'),
        _float_nchw(inputs[split:], 'test inputs'),
        _float_nchw(targets[split:], 'test targets'),
        {},
    )


def _with_metadata(data, metadata):
    return PublicPDEData(
        data.train_inputs, data.train_targets,
        data.test_inputs, data.test_targets, metadata)


def _load_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Public PDEBench adapters require h5py; install requirements.txt") from exc
    return h5py


def load_pdebench_darcy(cfg):
    """Load PDEBench 2-D Darcy: permeability ``nu`` -> pressure ``tensor``."""
    path = Path(cfg['path'])
    if not path.is_file():
        raise FileNotFoundError(f"PDEBench Darcy HDF5 not found: {path}")
    h5py = _load_h5py()
    stride = int(cfg.get('spatial_stride', 1))
    with h5py.File(path, 'r') as stream:
        missing = {'nu', 'tensor'} - set(stream.keys())
        if missing:
            raise ValueError(f"Darcy file misses HDF5 keys: {sorted(missing)}")
        if len(stream['nu']) != len(stream['tensor']):
            raise ValueError("Darcy input and solution sample counts differ")
        chosen = _chosen_indices(
            len(stream['nu']), cfg['n_train'], cfg['n_test'], cfg['split_seed'])
        # h5py requires increasing fancy indices. Restore the seeded order after
        # reading only the requested subset instead of materializing the full file.
        order = np.argsort(chosen)
        restore = np.argsort(order)
        sorted_indices = chosen[order]
        permeability = np.asarray(
            stream['nu'][sorted_indices], dtype=np.float32)[restore]
        solution = np.asarray(
            stream['tensor'][sorted_indices], dtype=np.float32)[restore]
    if permeability.ndim == 4 and permeability.shape[1] == 1:
        permeability = permeability[:, 0]
    if solution.ndim == 4:
        # Official PDEBench convention is [N,T,X,Y], with T=1 for Darcy.
        solution = solution[:, -1]
    inputs = permeability[:, ::stride, ::stride]
    targets = solution[:, ::stride, ::stride]
    data = _split_selected(inputs, targets, cfg['n_train'])
    return _with_metadata(data, {
        'dataset': 'PDEBench 2D Darcy flow',
        'operator': 'permeability field -> pressure solution',
        'source_url': 'https://doi.org/10.18419/darus-2986',
        'recommended_model': 'FNO2D',
    })


def _time_indices(cfg, plural, singular, default):
    values = cfg.get(plural)
    if values is None:
        values = [cfg.get(singular, default)]
    values = [int(value) for value in values]
    if not values:
        raise ValueError(f"{plural} must contain at least one time index")
    if len(values) != len(set(values)):
        raise ValueError(f"{plural} must not contain duplicate time indices")
    return values


def _pdebench_group_pairs(path, n_train, n_test, seed, stride,
                          input_time_indices, target_time_indices):
    """Read selected PDEBench trajectories and concatenate requested frames."""
    h5py = _load_h5py()
    with h5py.File(path, 'r') as stream:
        keys = []
        for key in sorted(stream.keys()):
            node = stream[key]
            if hasattr(node, 'keys') and 'data' in node:
                keys.append(key)
        if not keys:
            raise ValueError("no <group>/data trajectories found in PDEBench H5 file")
        chosen = _chosen_indices(len(keys), n_train, n_test, seed)
        inputs, targets = [], []
        for index in chosen:
            key = keys[int(index)]
            value = stream[key]['data']
            if value.ndim != 4:
                raise ValueError(
                    f"PDEBench trajectory {key}/data must be [T,X,Y,C]")
            input_frames = [np.asarray(
                value[time_index, ::stride, ::stride, :],
                dtype=np.float32) for time_index in input_time_indices]
            target_frames = [np.asarray(
                value[time_index, ::stride, ::stride, :],
                dtype=np.float32) for time_index in target_time_indices]
            inputs.append(np.concatenate(input_frames, axis=-1))
            targets.append(np.concatenate(target_frames, axis=-1))
    if not inputs:
        raise ValueError("no <group>/data trajectories found in PDEBench H5 file")
    return np.stack(inputs), np.stack(targets), len(keys)


def load_pdebench_shallow_water(cfg):
    """Load multihorizon PDEBench radial dam-break trajectories."""
    path = Path(cfg['path'])
    if not path.is_file():
        raise FileNotFoundError(f"PDEBench shallow-water H5 not found: {path}")
    initial = _time_indices(
        cfg, 'input_time_indices', 'input_time_index', 0)
    horizons = _time_indices(
        cfg, 'target_time_indices', 'target_time_index', -1)
    inputs, targets, available = _pdebench_group_pairs(
        path, cfg['n_train'], cfg['n_test'], cfg['split_seed'],
        int(cfg.get('spatial_stride', 1)), initial, horizons)
    inputs = np.moveaxis(inputs, -1, 1)
    targets = np.moveaxis(targets, -1, 1)
    data = _split_selected(inputs, targets, cfg['n_train'])
    return _with_metadata(data, {
        'dataset': 'PDEBench 2D radial dam-break shallow water',
        'operator': (
            f"state t{initial} -> multihorizon states t{horizons}"),
        'input_time_indices': initial,
        'target_time_indices': horizons,
        'available_trajectories': available,
        'source_url': 'https://doi.org/10.18419/darus-2986',
        'recommended_model': 'FNO2D',
    })


def load_pdebench_reaction_diffusion(cfg):
    """Load PDEBench 2-D reaction--diffusion trajectories."""
    path = Path(cfg['path'])
    if not path.is_file():
        raise FileNotFoundError(
            f"PDEBench reaction-diffusion H5 not found: {path}")
    initial = _time_indices(
        cfg, 'input_time_indices', 'input_time_index', 0)
    horizon = _time_indices(
        cfg, 'target_time_indices', 'target_time_index', -1)
    inputs, targets, available = _pdebench_group_pairs(
        path, cfg['n_train'], cfg['n_test'], cfg['split_seed'],
        int(cfg.get('spatial_stride', 1)), initial, horizon)
    inputs = np.moveaxis(inputs, -1, 1)
    targets = np.moveaxis(targets, -1, 1)
    data = _split_selected(inputs, targets, cfg['n_train'])
    return _with_metadata(data, {
        'dataset': 'PDEBench 2D reaction-diffusion',
        'operator': f'state t{initial} -> state t{horizon}',
        'available_trajectories': available,
        'source_url': 'https://doi.org/10.18419/darus-2986',
        'recommended_model': 'FNO2D',
    })


def _trajectory_field(path, stride):
    # Keep the full trajectory memory-mapped. Conversion happens only after
    # selecting the two frames used by the benchmark.
    value = np.load(path, mmap_mode='r')
    value = np.squeeze(value)
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3:
        raise ValueError(f"CFDBench field must be [T,H,W], got {value.shape}: {path}")
    return value[:, ::stride, ::stride]


def _cylinder_case(case, subset):
    u_path, v_path, config_path = case / 'u.npy', case / 'v.npy', case / 'case.json'
    if not (u_path.is_file() and v_path.is_file()):
        return None
    if not config_path.is_file():
        raise FileNotFoundError(f"CFDBench case metadata not found: {config_path}")
    parameters = json.loads(config_path.read_text(encoding='utf-8'))
    required = {
        'vel_in', 'density', 'viscosity', 'radius',
        'x_min', 'x_max', 'y_min', 'y_max'}
    missing = required - set(parameters)
    if missing:
        raise ValueError(
            f"CFDBench case metadata misses {sorted(missing)}: {config_path}")
    diameter = 2.0 * float(parameters['radius'])
    reynolds = (float(parameters['density']) * float(parameters['vel_in'])
                * diameter / float(parameters['viscosity']))
    return dict(
        u_path=u_path, v_path=v_path, case=str(case), subset=str(subset),
        parameters=parameters, reynolds=float(reynolds))


def _cylinder_mask(parameters, height, width):
    x = np.linspace(
        float(parameters['x_min']), float(parameters['x_max']), height,
        dtype=np.float32)
    y = np.linspace(
        float(parameters['y_min']), float(parameters['y_max']), width,
        dtype=np.float32)
    gx, gy = np.meshgrid(x, y, indexing='ij')
    radius = float(parameters['radius'])
    return ((gx ** 2 + gy ** 2) <= radius ** 2).astype(np.float32)


def _cylinder_anchors(length, history_frames, history_stride,
                      target_offsets, window_stride, maximum=None):
    start = (int(history_frames) - 1) * int(history_stride)
    stop = int(length) - max(target_offsets)
    anchors = np.arange(start, stop, int(window_stride), dtype=int)
    if maximum is not None and len(anchors) > int(maximum):
        selected = np.linspace(0, len(anchors) - 1, int(maximum), dtype=int)
        anchors = anchors[selected]
    if len(anchors) == 0:
        raise ValueError(
            'CFDBench trajectory is too short for the requested history and '
            'forecast offsets')
    return anchors


def _cylinder_windows(records, cfg, stride):
    history_frames = int(cfg.get('history_frames', 4))
    history_stride = int(cfg.get('history_stride', 1))
    target_offsets = [int(value) for value in cfg.get(
        'target_offsets', [1, 20, 100])]
    window_stride = int(cfg.get('window_stride', 64))
    maximum = cfg.get('max_windows_per_case')
    if history_frames <= 0 or history_stride <= 0 or window_stride <= 0:
        raise ValueError('CFDBench history and window settings must be positive')
    if not target_offsets or any(value <= 0 for value in target_offsets):
        raise ValueError('CFDBench target_offsets must contain positive values')

    inputs, targets = [], []
    for record in records:
        u = _trajectory_field(record['u_path'], stride)
        v = _trajectory_field(record['v_path'], stride)
        if u.shape != v.shape:
            raise ValueError(f"u/v shape mismatch in {record['case']}")
        parameters = record['parameters']
        mask = _cylinder_mask(parameters, u.shape[1], u.shape[2])
        constants = [
            mask,
            np.full_like(mask, float(parameters['vel_in'])),
            np.full_like(mask, float(parameters['density'])),
            np.full_like(mask, np.log10(float(parameters['viscosity']))),
            np.full_like(mask, float(parameters['radius'])),
        ]
        anchors = _cylinder_anchors(
            len(u), history_frames, history_stride, target_offsets,
            window_stride, maximum)
        for anchor in anchors:
            history = []
            first = anchor - (history_frames - 1) * history_stride
            for time_index in range(first, anchor + 1, history_stride):
                history.extend([u[time_index], v[time_index]])
            future = []
            for offset in target_offsets:
                future.extend([u[anchor + offset], v[anchor + offset]])
            inputs.append(np.stack([*history, *constants]))
            targets.append(np.stack(future))
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def load_cfdbench_cylinder(cfg):
    """Load conditioned, case-disjoint CFDBench multihorizon windows."""
    root = Path(cfg['path'])
    if not root.is_dir():
        raise FileNotFoundError(f"CFDBench cylinder directory not found: {root}")
    subsets = cfg.get('subsets', ['bc', 'geo', 'prop'])
    stride = int(cfg.get('spatial_stride', 1))
    cases = []
    for subset in subsets:
        for case in sorted((root / subset).glob('case*')):
            record = _cylinder_case(case, subset)
            if record is not None:
                cases.append(record)
    if not cases:
        raise ValueError("no CFDBench case*/u.npy + v.npy pairs were found")

    chosen = _chosen_indices(
        len(cases), cfg['n_train'], cfg['n_test'], cfg['split_seed'])
    selected = [cases[int(index)] for index in chosen]
    train_cases = selected[:int(cfg['n_train'])]
    test_cases = selected[int(cfg['n_train']):]
    train_inputs, train_targets = _cylinder_windows(train_cases, cfg, stride)
    test_inputs, test_targets = _cylinder_windows(test_cases, cfg, stride)
    data = PublicPDEData(
        _float_nchw(train_inputs, 'train inputs'),
        _float_nchw(train_targets, 'train targets'),
        _float_nchw(test_inputs, 'test inputs'),
        _float_nchw(test_targets, 'test targets'), {})
    reynolds = [record['reynolds'] for record in selected]
    history_frames = int(cfg.get('history_frames', 4))
    target_offsets = [int(value) for value in cfg.get(
        'target_offsets', [1, 20, 100])]
    return _with_metadata(data, {
        'dataset': 'CFDBench conditioned cylinder flow',
        'operator': (
            f'{history_frames}-frame velocity history and case conditions -> '
            f'velocity offsets {target_offsets}'),
        'available_cases': len(cases),
        'train_cases': len(train_cases),
        'test_cases': len(test_cases),
        'train_windows': len(train_inputs),
        'test_windows': len(test_inputs),
        'history_frames': history_frames,
        'history_stride': int(cfg.get('history_stride', 1)),
        'target_offsets': target_offsets,
        'window_stride': int(cfg.get('window_stride', 64)),
        'condition_channels': [
            'solid_mask', 'inlet_velocity', 'density',
            'log10_viscosity', 'cylinder_radius'],
        'reynolds_number_range': [
            float(np.min(reynolds)), float(np.max(reynolds))],
        'source_url': 'https://github.com/luo-yining/CFDBench',
        'recommended_model': 'FNO2D',
    })


def _local_annotation_path(token, annotation, data_root=None):
    """Resolve an OpenFWI manifest token against local downloaded data."""
    source = Path(token)
    if source.is_absolute() and source.is_file():
        return source
    if not source.is_absolute():
        relative = annotation.parent / source
        if relative.is_file() or data_root is None:
            return relative
    if data_root is not None:
        return Path(data_root) / source.name
    return source


def _annotation_pairs(path, data_root=None):
    """Read one-column local or two-column official OpenFWI manifests."""
    annotation = Path(path)
    if not annotation.is_file():
        raise FileNotFoundError(f"OpenFWI annotation file not found: {annotation}")
    result = []
    for line in annotation.read_text(encoding='utf-8').splitlines():
        value = line.strip()
        if not value:
            continue
        fields = value.split()
        if len(fields) not in (1, 2):
            raise ValueError(
                'OpenFWI annotation lines must contain one seismic path or '
                f'one seismic/velocity path pair: {value}')
        data_path = _local_annotation_path(
            fields[0], annotation, data_root)
        if len(fields) == 2:
            model_path = _local_annotation_path(
                fields[1], annotation, data_root)
        elif data_path.name.startswith('data'):
            model_path = data_path.with_name(
                data_path.name.replace('data', 'model', 1))
        elif data_path.name.startswith('seis'):
            model_path = data_path.with_name(
                data_path.name.replace('seis', 'vel', 1))
        else:
            raise ValueError(
                'OpenFWI seismic files must start with data or seis so the '
                f'paired model/velocity file can be resolved: {data_path}')
        result.append((data_path, model_path))
    return result


def _openfwi_pairs(annotation, maximum, time_stride, receiver_stride,
                   data_root=None):
    seismic_parts, model_parts, remaining = [], [], int(maximum)
    for data_path, model_path in _annotation_pairs(annotation, data_root):
        if remaining <= 0:
            break
        if not data_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(
                f"OpenFWI pair not found: {data_path} / {model_path}")
        seismic = np.load(data_path, mmap_mode='r')
        velocity = np.load(model_path, mmap_mode='r')
        count = min(len(seismic), len(velocity), remaining)
        seismic_parts.append(np.asarray(
            seismic[:count, :, ::time_stride, ::receiver_stride],
            dtype=np.float32))
        model_parts.append(np.asarray(velocity[:count], dtype=np.float32))
        remaining -= count
    if remaining > 0:
        raise ValueError(f"OpenFWI annotations provide {maximum - remaining}/{maximum} samples")
    return np.concatenate(seismic_parts), np.concatenate(model_parts)


def load_openfwi(cfg):
    """Load official OpenFWI waveform/velocity-map pairs."""
    time_stride = int(cfg.get('time_stride', 4))
    receiver_stride = int(cfg.get('receiver_stride', 1))
    data_root = cfg.get('data_root')
    train_x, train_y = _openfwi_pairs(
        cfg['train_annotation'], cfg['n_train'], time_stride, receiver_stride,
        data_root)
    test_x, test_y = _openfwi_pairs(
        cfg['test_annotation'], cfg['n_test'], time_stride, receiver_stride,
        data_root)
    return PublicPDEData(
        _float_nchw(train_x, 'OpenFWI train seismic'),
        _float_nchw(train_y, 'OpenFWI train velocity'),
        _float_nchw(test_x, 'OpenFWI test seismic'),
        _float_nchw(test_y, 'OpenFWI test velocity'),
        {
            'dataset': cfg.get('dataset_name', 'OpenFWI CurveFault-B'),
            'operator': 'seismic shot gathers -> subsurface velocity map',
            'source_url': 'https://openfwi-lanl.github.io/docs/data.html',
            'recommended_model': 'InversionNetLite',
        },
    )


LOADERS = {
    'pdebench_darcy': load_pdebench_darcy,
    'pdebench_shallow_water': load_pdebench_shallow_water,
    'pdebench_reaction_diffusion': load_pdebench_reaction_diffusion,
    'cfdbench_cylinder': load_cfdbench_cylinder,
    'openfwi': load_openfwi,
    'openfwi_flatvel_a': load_openfwi,
}


def load_public_dataset(cfg):
    name = cfg.get('type')
    if name not in LOADERS:
        raise ValueError(f"unknown public dataset {name!r}; choose from {sorted(LOADERS)}")
    return LOADERS[name](cfg)
