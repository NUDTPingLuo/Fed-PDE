"""CLI for application-oriented public PDE experiments.

Examples:
  python real_world_main.py --task all --dry-run
  python real_world_main.py --task darcy --validate-only
  python real_world_main.py --task shallow_water --device cuda:0
"""
import argparse
import copy
import csv
import os

import yaml

from real_world.datasets import load_public_dataset
from util.result_contract import matching_metadata


def load_manifest(path):
    with open(path, encoding='utf-8') as stream:
        manifest = yaml.safe_load(stream)
    return manifest['tasks']


def load_task_config(path):
    with open(path, encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    required = {
        'task', 'dataset', 'model', 'partition',
        'federation', 'training', 'evaluation', 'output'}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"public task config misses {sorted(missing)}")
    if cfg['federation']['n_clients'] * cfg['partition']['min_client_samples'] \
            > cfg['dataset']['n_train']:
        raise ValueError("n_train cannot satisfy the minimum size of every client")
    positive = [
        cfg['dataset']['n_train'], cfg['dataset']['n_test'],
        cfg['partition']['n_bins'], cfg['partition']['min_client_samples'],
        cfg['federation']['n_clients'], cfg['training']['batch_size'],
        cfg['training']['local_steps'], cfg['training']['n_rounds']]
    if any(int(value) <= 0 for value in positive):
        raise ValueError("dataset, partition, federation and training sizes must be positive")
    if cfg['partition']['n_bins'] > cfg['dataset']['n_train']:
        raise ValueError("n_bins cannot exceed n_train")
    if any(float(alpha) <= 0 for alpha in cfg['partition']['alpha_values']):
        raise ValueError("all Dirichlet alpha values must be positive")
    momentum = float(cfg['training'].get('momentum', 0.0))
    if not 0.0 <= momentum < 1.0:
        raise ValueError("training.momentum must satisfy 0 <= momentum < 1")
    if not cfg['evaluation']['seeds']:
        raise ValueError("at least one evaluation seed is required")
    if any(int(value) < 0 for value in
           cfg['evaluation'].get('round_checkpoints', [])):
        raise ValueError("round_checkpoints must be nonnegative")
    from util.schedules import validate_schedule
    validate_schedule(cfg['training'])
    cfg['output']['dir'] = os.path.normpath(cfg['output']['dir'])
    return cfg


def selected_configs(manifest_path, task):
    manifest = load_manifest(manifest_path)
    names = list(manifest) if task == 'all' else [task]
    unknown = set(names) - set(manifest)
    if unknown:
        raise ValueError(f"unknown public tasks: {sorted(unknown)}")
    return [(name, manifest[name]) for name in names]


def _row_count(path):
    if not os.path.isfile(path):
        return 0
    with open(path, newline='', encoding='utf-8') as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _has_seed_bin_counts(path):
    if not os.path.isfile(path):
        return False
    with open(path, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    return bool(rows) and all(
        row.get('client_bin_counts') and row.get('iid_client_bin_counts')
        for row in rows)


def _has_columns(path, required):
    if not os.path.isfile(path):
        return False
    with open(path, newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        return required.issubset(reader.fieldnames or [])


def _complete(path, cfg):
    from util.round_tracking import resolve_round_checkpoints
    expected = len(cfg['partition']['alpha_values'])
    if not path.endswith('_alpha.csv'):
        return False
    stem = path[:-len('_alpha.csv')]
    if not matching_metadata(stem + '_metadata.json', cfg, public=True):
        return False
    raw_path = stem + '_alpha_seed_raw.csv'
    round_raw_path = stem + '_alpha_round_seed_raw.csv'
    iid_path = stem + '_iid.csv'
    checkpoints = resolve_round_checkpoints(
        cfg['training']['n_rounds'],
        cfg['evaluation'].get('round_checkpoints') or None)
    return (_row_count(path) >= expected
            and _row_count(raw_path) >= expected * len(cfg['evaluation']['seeds'])
            and _has_seed_bin_counts(raw_path)
            and _row_count(round_raw_path)
            >= expected * len(cfg['evaluation']['seeds']) * len(checkpoints)
            and _has_columns(round_raw_path, {'iid_error', 'drift'})
            and _row_count(iid_path)
            >= len(cfg['evaluation']['seeds']) * len(checkpoints)
            and _has_columns(iid_path, {'partition', 'test_error'}))


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='config/public_experiments.yaml')
    parser.add_argument('--task', required=True)
    parser.add_argument('--device')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    for name, path in selected_configs(args.manifest, args.task):
        cfg = copy.deepcopy(load_task_config(path))
        if args.device:
            cfg['training']['device'] = args.device
        if args.smoke:
            minimum = (cfg['federation']['n_clients']
                       * cfg['partition']['min_client_samples'])
            cfg['dataset']['n_train'] = max(
                minimum, min(cfg['dataset']['n_train'], 200))
            cfg['dataset']['n_test'] = min(cfg['dataset']['n_test'], 16)
            cfg['partition']['alpha_values'] = [1.0]
            cfg['evaluation']['seeds'] = [0]
            cfg['evaluation']['round_checkpoints'] = [0, 1]
            cfg['training']['n_rounds'] = 1
            cfg['training']['local_steps'] = 1
            cfg['model']['width'] = 8
            if 'modes' in cfg['model']:
                cfg['model']['modes'] = 4
            if 'spatial_stride' in cfg['dataset']:
                cfg['dataset']['spatial_stride'] = max(
                    int(cfg['dataset']['spatial_stride']), 4)
            if cfg['dataset']['type'] == 'cfdbench_cylinder':
                cfg['dataset']['max_windows_per_case'] = 1
            cfg['output']['prefix'] += '_smoke'
        target = os.path.join(
            cfg['output']['dir'], cfg['output']['prefix'] + '_alpha.csv')
        print(f"{name:16s} {cfg['dataset']['type']:28s} -> {target}")
        if args.dry_run:
            continue
        if args.validate_only:
            data = load_public_dataset(cfg['dataset'])
            print(
                f"  train {data.train_inputs.shape} -> {data.train_targets.shape}; "
                f"test {data.test_inputs.shape} -> {data.test_targets.shape}")
            continue
        if not args.force and _complete(target, cfg):
            print(f"  skip completed result: {target}")
            continue
        from real_world.benchmark import run_public_benchmark
        run_public_benchmark(cfg)


if __name__ == '__main__':
    main_cli()
