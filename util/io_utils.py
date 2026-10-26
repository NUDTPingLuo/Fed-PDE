"""Configuration loading and reproducible result persistence."""
import csv
import json
import os
import yaml


def load_config(path):
    with open(path, encoding='utf-8') as stream:
        raw = yaml.safe_load(stream)
    data = raw['data']
    physics = raw.get('physics', {})
    benchmark = raw['benchmark']
    partition = benchmark.get('partition', {})
    cfg = dict(
        experiment_profile=raw.get('experiment_profile', 'custom'),
        task=raw['task'],
        n_sensors=int(data['n_sensors']),
        output_grid=int(data['output_grid']),
        n_modes=int(data.get('n_modes', 10)),
        condition_scale=float(data.get('condition_scale', 1.0)),
        diffusion=float(physics.get('diffusion', 0.01)),
        reaction=float(physics.get('reaction', 0.01)),
        viscosity=float(physics.get('viscosity', 0.1)),
        n_bins=int(benchmark['n_bins']),
        sampling_pool_size=int(benchmark['sampling_pool_size']),
        alpha_values=[float(value) for value in benchmark['alpha_values']],
        transport_samples=int(benchmark.get('transport_samples', 100)),
        min_client_samples=int(partition.get('min_client_samples', 16)),
        n_clients=int(raw['federation']['n_clients']),
        model_width=int(raw['model']['width']),
        model_depth=int(raw['model']['depth']),
        model_activation=raw['model'].get('activation', 'relu'),
        optimizer=raw['training']['optimizer'],
        momentum=float(raw['training'].get('momentum', 0.0)),
        training_backend=raw['training'].get('backend', 'numpy'),
        training_device=raw['training'].get('device', 'auto'),
        training_batch_size=raw['training'].get('batch_size'),
        mixed_precision=bool(raw['training'].get('mixed_precision', False)),
        lr=float(raw['training']['lr']),
        lr_schedule=raw['training'].get('lr_schedule', 'cosine'),
        min_lr=float(raw['training'].get('min_lr', 0.0001)),
        local_steps=int(raw['training']['local_steps']),
        n_rounds=int(raw['training']['n_rounds']),
        seeds=[int(value) for value in raw['evaluation']['seeds']],
        round_checkpoints=[int(value) for value in
                           raw['evaluation'].get('round_checkpoints', [])],
        n_test=int(raw['evaluation']['n_test']),
        seed_base=int(raw['seed_base']),
        output_dir=raw['output']['dir'],
        output_prefix=raw['output']['prefix'],
    )
    if 'nesterov' in raw['training']:
        cfg['nesterov'] = bool(raw['training']['nesterov'])
    if cfg.get('nesterov', False) and (
            cfg['optimizer'] != 'sgd' or not 0 < cfg['momentum'] < 1):
        raise ValueError('Nesterov requires SGD with momentum in (0,1)')
    from util.schedules import validate_schedule
    validate_schedule(cfg)
    cfg['n_seeds'] = len(cfg['seeds'])
    valid_tasks = {'antiderivative', 'diffusion_reaction', 'burgers'}
    if cfg['task'] not in valid_tasks:
        raise ValueError(f"task must be one of {sorted(valid_tasks)}")
    if cfg['n_bins'] < 2:
        raise ValueError("benchmark.n_bins must be at least 2")
    if cfg['sampling_pool_size'] < cfg['n_bins']:
        raise ValueError("sampling_pool_size must be at least n_bins")
    if any(alpha <= 0 for alpha in cfg['alpha_values']):
        raise ValueError("all alpha_values must be positive")
    if (not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds'])
            or any(seed < 0 for seed in cfg['seeds'])):
        raise ValueError("evaluation.seeds must contain unique nonnegative integers")
    if cfg['n_modes'] < 2 or cfg['n_sensors'] < cfg['n_modes']:
        raise ValueError("n_sensors must be at least n_modes >= 2")
    positive_keys = (
        'n_sensors', 'output_grid', 'sampling_pool_size',
        'n_clients', 'min_client_samples', 'model_width', 'model_depth',
        'local_steps', 'n_rounds', 'n_seeds', 'n_test',
        'transport_samples',
    )
    if any(cfg[key] <= 0 for key in positive_keys) or cfg['lr'] <= 0:
        raise ValueError("data, model, and training sizes must be positive")
    if cfg['n_clients'] * cfg['min_client_samples'] > cfg['sampling_pool_size']:
        raise ValueError(
            "sampling_pool_size must cover min_client_samples for every client")
    if cfg['model_activation'] not in {'relu', 'tanh', 'sine'}:
        raise ValueError("model.activation must be relu, tanh, or sine")
    if cfg['optimizer'] not in {'sgd', 'adam'}:
        raise ValueError("training.optimizer must be 'sgd' or 'adam'")
    if not 0.0 <= cfg['momentum'] < 1.0:
        raise ValueError("training.momentum must satisfy 0 <= momentum < 1")
    if cfg['training_backend'] not in {'numpy', 'torch', 'auto'}:
        raise ValueError("training.backend must be numpy, torch, or auto")
    if not (cfg['training_device'] == 'auto'
            or cfg['training_device'] == 'cpu'
            or str(cfg['training_device']).startswith('cuda')):
        raise ValueError("training.device must be auto, cpu, cuda, or cuda:N")
    if cfg['training_batch_size'] is not None:
        cfg['training_batch_size'] = int(cfg['training_batch_size'])
        if cfg['training_batch_size'] <= 0:
            raise ValueError("training.batch_size must be null or positive")
    if any(value < 0 for value in cfg['round_checkpoints']):
        raise ValueError("evaluation.round_checkpoints must be nonnegative")
    return cfg


def _write_rows(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_results(rows, cfg, codebook_metadata, seed_rows=None,
                 round_seed_rows=None, iid_rows=None):
    os.makedirs(cfg['output_dir'], exist_ok=True)
    csv_path = os.path.join(
        cfg['output_dir'], cfg['output_prefix'] + '_alpha.csv')
    _write_rows(csv_path, rows)
    metadata_path = os.path.join(cfg['output_dir'], cfg['output_prefix'] + '_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as stream:
        json.dump({'config': cfg, 'benchmark': codebook_metadata}, stream,
                  indent=2, ensure_ascii=False)
    seed_path = None
    if seed_rows is not None:
        seed_path = os.path.join(
            cfg['output_dir'], cfg['output_prefix'] + '_alpha_seed_raw.csv')
        _write_rows(seed_path, seed_rows)
    round_seed_path = None
    if round_seed_rows is not None:
        round_seed_path = os.path.join(
            cfg['output_dir'],
            cfg['output_prefix'] + '_alpha_round_seed_raw.csv')
        _write_rows(round_seed_path, round_seed_rows)
    iid_path = None
    if iid_rows is not None:
        iid_path = os.path.join(
            cfg['output_dir'], cfg['output_prefix'] + '_iid.csv')
        _write_rows(iid_path, iid_rows)
    print(f"Results saved to {csv_path}")
    if seed_path is not None:
        print(f"Seed-level results saved to {seed_path}")
    if round_seed_path is not None:
        print(f"Round-level seed results saved to {round_seed_path}")
    if iid_path is not None:
        print(f"IID reference results saved to {iid_path}")
    print(f"Metadata saved to {metadata_path}")
    return csv_path, metadata_path


def load_results(path):
    """Load checkpoint CSV rows while recovering numeric and boolean types."""
    if not os.path.exists(path):
        return []

    def parse(value):
        if value == 'True':
            return True
        if value == 'False':
            return False
        try:
            return float(value)
        except ValueError:
            return value

    with open(path, newline='', encoding='utf-8') as stream:
        return [{key: parse(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
