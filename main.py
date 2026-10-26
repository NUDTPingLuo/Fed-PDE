"""Primary CLI for all paper experiments.

Examples
--------
python main.py --study formal
python main.py --study clients --tasks burgers
python main.py --study viscosity
python main.py --study optimizer
python main.py --study all

Completed result files are skipped by default. Pass ``--force`` to rerun them,
``--dry-run`` to print the plan, or ``--smoke`` for a tiny integration run.
"""
import argparse
import copy
import csv
import os
import yaml

import experiment
from util.io_utils import load_config
from util.result_contract import matching_metadata


STUDIES = ('formal', 'clients', 'viscosity', 'optimizer')


def load_experiment_config(path):
    with open(path, encoding='utf-8') as stream:
        settings = yaml.safe_load(stream)
    missing = {'base_configs', 'common'} - set(settings)
    if missing:
        raise ValueError(f"paper experiment config misses {sorted(missing)}")
    return settings


def selected_tasks(settings, study, requested=None):
    configured = ([settings['viscosity']['task']]
                  if study == 'viscosity' else settings[study]['tasks'])
    if not requested:
        return list(configured)
    unknown = set(requested) - set(configured)
    if unknown:
        raise ValueError(
            f"tasks {sorted(unknown)} are unavailable for study {study}")
    return list(requested)


def study_plan(settings, study, tasks=None, values=None):
    """Return atomic runs, enabling dry-run inspection and unit testing."""
    if study not in STUDIES:
        raise ValueError(f"unknown study {study}")
    tasks = selected_tasks(settings, study, tasks)
    configured_values = settings[study].get('values', [None])
    if values is not None:
        unavailable = [value for value in values
                       if value not in configured_values]
        if unavailable:
            raise ValueError(
                f"values {unavailable} are outside the configured {study} "
                f"protocol {configured_values}")
    chosen = configured_values if values is None else values
    if study == 'formal':
        chosen = [None]
    return [dict(study=study, task=task, value=value)
            for task in tasks for value in chosen]


def prepare_run_config(settings, item, smoke=False, device=None):
    cfg = copy.deepcopy(load_config(settings['base_configs'][item['task']]))
    common = settings['common']
    cfg['n_rounds'] = int(common['n_rounds'])
    cfg['local_steps'] = int(common['local_steps'])
    cfg['seeds'] = [int(seed) for seed in common['seeds']]
    cfg['n_seeds'] = len(cfg['seeds'])
    cfg['round_checkpoints'] = [
        int(value) for value in common.get('round_checkpoints', [])]
    output_root = os.path.normpath(common['output_dir'])
    study, value = item['study'], item['value']
    if study == 'formal':
        cfg['output_dir'] = os.path.join(
            output_root, 'formal', item['task'])
        suffix = f"formal_k{cfg['n_clients']}"
    elif study == 'clients':
        cfg['output_dir'] = os.path.join(
            output_root, 'ablation_clients', item['task'])
        cfg['n_clients'] = int(value)
        suffix = f"clients_k{int(value)}"
    elif study == 'viscosity':
        cfg['output_dir'] = os.path.join(
            output_root, 'ablation_viscosity', item['task'])
        cfg['viscosity'] = float(value)
        suffix = (f"viscosity_{float(value):g}_k{cfg['n_clients']}"
                  .replace('.', 'p'))
    elif study == 'optimizer':
        cfg['output_dir'] = os.path.join(
            output_root, 'ablation_optimizer', item['task'])
        cfg['optimizer'] = str(value).lower()
        if cfg['optimizer'] == 'sgd':
            cfg['lr'] = float(settings['optimizer'].get('sgd_lr', 0.01))
            cfg['min_lr'] = float(settings['optimizer'].get('sgd_min_lr', 0.001))
            cfg['lr_schedule'] = 'cosine'
            cfg['nesterov'] = bool(settings['optimizer'].get('sgd_nesterov', True))
        cfg['momentum'] = (
            float(settings['optimizer'].get('sgd_momentum', 0.9))
            if cfg['optimizer'] == 'sgd' else 0.0)
        momentum_tag = str(cfg['momentum']).replace('.', 'p')
        learning_rate_tag = f"{cfg['lr']:g}".replace('.', 'p')
        nesterov_tag = '_nesterov' if cfg.get('nesterov', False) else ''
        suffix = (
            f"optimizer_{cfg['optimizer']}_lr{learning_rate_tag}"
            f"_m{momentum_tag}{nesterov_tag}_k{cfg['n_clients']}"
            if cfg['optimizer'] == 'sgd'
            else f"optimizer_{cfg['optimizer']}_k{cfg['n_clients']}")
    if smoke:
        apply_smoke_profile(cfg)
    from util.schedules import validate_schedule
    validate_schedule(cfg)
    if cfg.get('nesterov', False) and (
            cfg['optimizer'] != 'sgd' or not 0 < cfg.get('momentum', 0) < 1):
        raise ValueError('Nesterov requires SGD with momentum in (0,1)')
    if device is not None:
        cfg['training_device'] = device
    if cfg['n_clients'] * cfg['min_client_samples'] > cfg['sampling_pool_size']:
        raise ValueError(
            "sampling_pool_size must cover min_client_samples for every client")
    cfg['output_prefix'] = (
        f"{item['task']}_{suffix}_s{cfg['local_steps']}"
        f"_r{cfg['n_rounds']}")
    if smoke:
        cfg['output_prefix'] += '_smoke'
    return cfg


def apply_smoke_profile(cfg):
    """Keep orchestration tests fast without changing publication defaults."""
    cfg.update(
        n_rounds=1, local_steps=1, seeds=[cfg['seeds'][0]], n_seeds=1,
        n_bins=3, sampling_pool_size=50, n_test=6,
        min_client_samples=1, model_width=4, model_depth=2,
        output_grid=5, n_sensors=6, n_modes=3,
        transport_samples=6,
        round_checkpoints=[0, 1],
    )


def result_path(cfg):
    return os.path.join(
        cfg['output_dir'], cfg['output_prefix'] + '_alpha.csv')


def expected_result_rows(cfg):
    return len(cfg['alpha_values'])


def completed_result(path, cfg):
    if not path.endswith('_alpha.csv'):
        raise ValueError("result path must end with _alpha.csv")
    stem = path[:-len('_alpha.csv')]
    if not matching_metadata(stem + '_metadata.json', cfg):
        return False
    seed_path = stem + '_alpha_seed_raw.csv'
    round_seed_path = stem + '_alpha_round_seed_raw.csv'
    iid_path = stem + '_iid.csv'
    seed_rows = _read_rows(seed_path)
    round_seed_rows = _read_rows(round_seed_path)
    checkpoint_count = len(experiment.evaluation_rounds(cfg))
    return (len(_read_rows(path)) >= expected_result_rows(cfg)
            and len(seed_rows) >= expected_result_rows(cfg) * cfg['n_seeds']
            and all(row.get('client_bin_counts') for row in seed_rows)
            and all(row.get('iid_client_bin_counts') for row in seed_rows)
            and len(round_seed_rows)
            >= expected_result_rows(cfg) * cfg['n_seeds'] * checkpoint_count
            and all('iid_error' in row for row in round_seed_rows)
            and len(_read_rows(iid_path)) >= cfg['n_seeds'] * checkpoint_count
            and all(row.get('partition') == 'iid'
                    for row in _read_rows(iid_path)))


def run_federated_item(settings, item, force=False, dry_run=False,
                       smoke=False, device=None):
    cfg = prepare_run_config(settings, item, smoke=smoke, device=device)
    cfg['resume'] = not force
    cfg['checkpoint'] = True
    path = result_path(cfg)
    print(f"{item['study']:16s} task={item['task']:20s} "
          f"value={item['value']} -> {path}")
    if dry_run:
        return path
    if not force and completed_result(path, cfg):
        print(f"Skip completed result: {path}")
        return path
    experiment.run_experiment(cfg)
    return path


def _read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))
def parse_values(study, raw_values):
    if raw_values is None:
        return None
    if study == 'clients':
        return [int(value) for value in raw_values]
    if study == 'viscosity':
        return [float(value) for value in raw_values]
    if study == 'optimizer':
        return [str(value).lower() for value in raw_values]
    if raw_values:
        raise ValueError('--values is not used by the formal study')
    return None


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/paper_experiments.yaml')
    parser.add_argument('--study', choices=(*STUDIES, 'all'), required=True)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--values', nargs='+')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument(
        '--device', default=None,
        help="override the task YAML training device: auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    settings = load_experiment_config(args.config)
    studies = STUDIES if args.study == 'all' else (args.study,)
    if args.study == 'all' and args.values:
        raise ValueError('--values requires one explicit --study')
    for study in studies:
        values = parse_values(study, args.values if study == args.study else None)
        for item in study_plan(settings, study, args.tasks, values):
            run_federated_item(
                settings, item, args.force, args.dry_run, args.smoke,
                args.device)


if __name__ == '__main__':
    main_cli()
