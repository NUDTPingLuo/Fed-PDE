"""Data adapters, paired comparisons, atomic output and run-level resume."""
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform

import numpy as np
import torch
import yaml
from scipy.stats import t as student_t

from experiment import make_task, model_spec, prepare_seed_data
from real_world.benchmark import _normalize_data
from real_world.datasets import load_public_dataset
from real_world.models import make_model
from real_world.training import resolve_device
from util.io_utils import load_config
from util.partitioning import (
    cluster_solution_responses, partition_clients, partition_clients_iid)
from util.torch_training import TorchCartesianDeepONet

from .core import ClientShard, count_matrix, train, participation_schedule
from util.schedules import validate_schedule

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('fedavg', 'sb_cfo')
SCHEMA_VERSION = 2


def digest_array(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def effective_base(cfg):
    if cfg['task'] == 'burgers':
        base = load_config(cfg['base_config'])
        base['viscosity'] = cfg['viscosity']
        base.update(n_clients=cfg['n_clients'], n_bins=cfg['n_bins'],
                    min_client_samples=cfg['min_client_samples'])
        if cfg['smoke']:
            base.update(sampling_pool_size=24, n_test=4, n_sensors=9,
                        output_grid=9, n_modes=3, model_width=4, model_depth=2)
    else:
        with open(cfg['base_config'], encoding='utf-8') as stream:
            base = yaml.safe_load(stream)
        source = Path(base['dataset']['path'])
        base['dataset']['path'] = str((source if source.is_absolute() else ROOT / source).resolve())
        if cfg['smoke']:
            base['dataset'].update(n_train=24, n_test=4, spatial_stride=16)
            base['model'].update(width=4, modes=2, layers=2)
    available = base['sampling_pool_size'] if cfg['task'] == 'burgers' else base['dataset']['n_train']
    if cfg['n_clients'] * cfg['min_client_samples'] > available or cfg['n_bins'] > available:
        raise ValueError("training pool cannot satisfy clients/bins/minimum sample count")
    return base


def specification(cfg, base):
    """Hash settings, source versions and external-data identity before reuse."""
    source_names = [
        'sb_cfo_main.py', 'sb_cfo/core.py', 'sb_cfo/runner.py', 'util/schedules.py',
        'experiment.py', 'operator_tasks/tasks.py', 'util/partitioning.py',
        'util/torch_training.py', 'util/io_utils.py', 'real_world/models.py',
        'real_world/datasets.py', 'real_world/benchmark.py',
    ]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in source_names}
    file_identity = None
    if cfg['task'] == 'darcy':
        path = Path(base['dataset']['path'])
        if not path.is_file():
            raise FileNotFoundError("PDEBench Darcy file not found: " + str(path))
        stat = path.stat()
        file_identity = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    payload = dict(schema=SCHEMA_VERSION, config=cfg, base=base,
                   source_sha256=hashes, data_file=file_identity,
                   runtime=dict(python=platform.python_version(), torch=torch.__version__,
                                numpy=np.__version__))
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest(), payload


def prepare_data(cfg, base, seed):
    print("Preparing {} seed={}: training labels, normalization and solution bins".format(
        cfg['task'], seed), flush=True)
    if cfg['task'] == 'burgers':
        task = make_task(base)
        raw = prepare_seed_data(base, task, seed)
        spec = model_spec(base, task)
        data = dict(train_x=np.asarray(raw['conditions'], dtype=np.float32),
                    train_y=np.asarray(raw['responses'], dtype=np.float32),
                    test_x=np.asarray(raw['test_conditions'], dtype=np.float32),
                    test_y=np.asarray(raw['test_responses'], dtype=np.float32),
                    bins=raw['solution_bins'], target_mean=0., target_std=1.,
                    partition_seed=base['seed_base'] + 30000 + seed,
                    batch_size=base['training_batch_size'],
                    model_name='CartesianDeepONet',
                    factory=lambda: TorchCartesianDeepONet(base['n_sensors'], spec, seed),
                    dataset_metadata=dict(source='generated Burgers', viscosity=cfg['viscosity']))
    else:
        public = load_public_dataset(base['dataset'])
        normalized = _normalize_data(public)
        bins = cluster_solution_responses(
            normalized['train_y'].reshape(len(normalized['train_y']), -1),
            cfg['n_bins'], base['partition'].get('clustering_seed', 42),
            base['partition'].get('clustering_max_iter', 100))
        data = dict(normalized, bins=bins, batch_size=base['training']['batch_size'],
                    partition_seed=base['partition'].get('partition_seed_base', 30000) + seed,
                    model_name=base['model']['type'],
                    factory=lambda: make_model(base['model'], normalized['train_x'].shape[1:],
                                               normalized['train_y'].shape[1:]),
                    dataset_metadata=public.metadata)
    data['identity'] = {
        name: digest_array(data[name])
        for name in ('train_x', 'train_y', 'test_x', 'test_y')
    }
    data['identity']['labels'] = digest_array(data['bins'].labels)
    print("  train {} -> {}; test {} -> {}; bins={}".format(
        data['train_x'].shape, data['train_y'].shape,
        data['test_x'].shape, data['test_y'].shape, cfg['n_bins']), flush=True)
    return data


def split_data(cfg, data, alpha):
    labels = data['bins'].labels
    if alpha is None:
        indices, _, _ = partition_clients_iid(
            labels, cfg['n_clients'], cfg['min_client_samples'], data['partition_seed'])
    else:
        indices, _, _ = partition_clients(
            labels, cfg['n_clients'], alpha, cfg['min_client_samples'], data['partition_seed'])
    clients = [ClientShard(data['train_x'][index], data['train_y'][index], labels[index])
               for index in indices]
    return clients, dict(
        client_bin_counts=count_matrix(clients, cfg['n_bins']).tolist(),
        client_index_sha256=[digest_array(index) for index in indices],
        partition_seed=data['partition_seed'])


def condition_name(alpha):
    return 'iid' if alpha is None else 'alpha' + format(alpha, 'g')


def key(row):
    return str(row['method']), str(row['condition']), int(row['seed']), int(row['round'])


def read_rows(path):
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(str(temporary), str(path))


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: '' if isinstance(v, float) and not math.isfinite(v) else v
                                 for k, v in row.items()})
    os.replace(str(temporary), str(path))


def paired_metrics(rows):
    indexed = {key(row): row for row in rows}
    for row in rows:
        method, condition, seed, round_index = key(row)
        def reference(m, c):
            other = indexed.get((m, c, seed, round_index))
            return float(other['test_error']) if other is not None else None
        error = float(row['test_error'])
        iid = reference('fedavg', 'iid')
        base = reference('fedavg', condition)
        own_iid = reference(method, 'iid')
        row['excess_vs_fedavg_iid_pp'] = '' if iid is None else 100 * (error - iid)
        row['improvement_vs_fedavg_pp'] = '' if base is None else 100 * (base - error)
        row['excess_vs_own_iid_pp'] = '' if own_iid is None else 100 * (error - own_iid)
    return rows


def summarize(rows, final_round):
    groups = {}
    for row in rows:
        if int(row['round']) == final_round:
            groups.setdefault((row['method'], row['condition']), []).append(row)
    summary = []
    metrics = ('test_error', 'excess_vs_fedavg_iid_pp', 'improvement_vs_fedavg_pp',
               'excess_vs_own_iid_pp', 'correction_ratio',
               'gradient_variance_before', 'gradient_variance_after',
               'local_update_variance', 'optimizer_examples', 'prototype_examples',
               'diagnostic_examples', 'model_communication_bytes',
               'prototype_communication_bytes', 'elapsed_seconds')
    for (method, condition), group in groups.items():
        row = dict(method=method, condition=condition, n_seeds=len(group))
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in group
                                 if r.get(metric) not in (None, '')], dtype=float)
            values = values[np.isfinite(values)]
            mean = float(np.mean(values)) if len(values) else float('nan')
            half = (float(student_t.ppf(0.975, len(values) - 1)
                          * np.std(values, ddof=1) / np.sqrt(len(values)))
                    if len(values) >= 2 else float('nan'))
            row[metric + '_mean'] = mean
            row[metric + '_ci95_low'] = mean - half
            row[metric + '_ci95_high'] = mean + half
        summary.append(row)
    return summary


def expected_keys(cfg):
    return {(method, condition_name(alpha), seed, t)
            for seed in cfg['seeds'] for alpha in [None] + cfg['alpha_values']
            for method in cfg.get('methods', METHODS) for t in cfg['round_checkpoints']}


def valid_keys(rows):
    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate result keys; preserve files and inspect before rerunning")
    if any(not math.isfinite(float(row['test_error'])) for row in rows):
        raise ValueError("result contains non-finite test error")
    return set(keys)


def run_experiment(cfg, force=False, validate_only=False):
    cfg = copy.deepcopy(cfg)
    methods = tuple(cfg.get('methods', METHODS))
    if (not methods or len(set(methods)) != len(methods) or 'fedavg' not in methods
            or set(methods) - set(METHODS)):
        raise ValueError("methods must be unique and contain fedavg with supported comparators")
    validate_schedule(cfg['training'])
    rate = cfg['training'].get('participation_rate', 1.)
    schedules = {str(seed): participation_schedule(
        cfg['n_clients'], cfg['training']['n_rounds'], rate, seed) for seed in cfg['seeds']}
    if rate < 1 and methods != ('fedavg',):
        raise ValueError('Partial participation currently supports fedavg only')
    cfg['training']['device'] = str(resolve_device(cfg['training']['device']))
    base = effective_base(cfg)
    fingerprint, spec = specification(cfg, base)
    output = Path(cfg['output_dir'])
    metadata_path, rounds_path = output / 'metadata.json', output / 'rounds.csv'
    summary_path = output / 'summary.csv'
    if validate_only:
        data = prepare_data(cfg, base, cfg['seeds'][0])
        for alpha in [None] + cfg['alpha_values']:
            clients, detail = split_data(cfg, data, alpha)
            print("  {} client sizes={}".format(
                condition_name(alpha), [len(c.inputs) for c in clients]), flush=True)
        print("Validation passed; no optimizer steps or result files written.", flush=True)
        return []
    rows, metadata = [], None
    if not force:
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if metadata.get('fingerprint') != fingerprint:
                raise ValueError(
                    "Existing paired results have different settings/source/runtime. "
                    "Use --output-dir for a separate run, or --force to replace only "
                    "this comparison's rounds.csv, summary.csv and metadata.json.")
            rows = read_rows(rounds_path)
        elif rounds_path.exists() or summary_path.exists():
            raise ValueError("Results without metadata; choose a new --output-dir or --force")
    present = valid_keys(rows)
    if present - expected_keys(cfg):
        raise ValueError("unexpected result keys; use a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    if force:
        # Reset our two result tables BEFORE changing the manifest. A crash
        # must never allow stale rows to inherit the new settings fingerprint.
        write_rows(rounds_path, [])
        write_rows(summary_path, [])
    if metadata is None:
        metadata = dict(
            fingerprint=fingerprint, specification=spec, complete=False, datasets={}, partitions={},
            protocol=dict(
                methods=list(methods), reference='sample-balanced bin-stratified IID',
                prototype='client-local conditional mean gradients, weighted by n_kb/N_b',
                correction='strength * sum_b (pi_b - h_kb) c_b; added before optimizer.step',
                optimizer_state='persistent per client, stored on CPU between visits',
                learning_rate_schedule=dict(
                    name=cfg['training'].get('lr_schedule', 'cosine'),
                    initial_lr=cfg['training']['lr'],
                    min_lr=cfg['training'].get('min_lr', 0.0001),
                    cosine_formula='min_lr + (lr-min_lr)*(1+cos(pi*(r-1)/(R-1)))/2; r=1..R; R=1 uses lr',
                    application='global communication round; fixed within round; no warm-up or restarts; identical across methods/conditions',
                    state_restore='set lr after restoring each client optimizer state',
                    recorded_lr='rounds.csv learning_rate is the rate used in the just-completed round; blank at round 0'),
                diagnostics='full training gradients at one common current global model',
                communication='logical uncompressed tensor bytes, not measured network traffic',
                privacy='offline benchmark bins; no DP or secure aggregation guarantee',
                resume='completed method/condition/seed runs only; not mid-round checkpoint recovery',
                units='test_error is relative L2; paired error differences are percentage points',
                scope='SGD identity at a shared model; Adam/momentum variants are empirical'),
        )
    metadata['complete'] = False
    metadata['participation'] = dict(
        requested_rate=rate,
        clients_per_round=len(next(iter(schedules.values()))[0]),
        sampling='uniform without replacement within each round; independent across rounds',
        pairing='same schedule for IID and all alphas at a given seed',
        aggregation='selected-client sample weights n_k / sum_{j in S_t} n_j',
        optimizer_state='inactive clients retain their optimizer state unchanged',
        local_update_variance='weighted variance among selected clients in the preceding round',
        diagnostic_scope='all clients at checkpoints; diagnostic gradients do not update model',
        schedules_zero_based=schedules)
    write_json(metadata_path, metadata)

    def save():
        paired_metrics(rows)
        write_rows(rounds_path, rows)
        write_rows(summary_path, summarize(rows, cfg['training']['n_rounds']))
        metadata['complete'] = valid_keys(rows) == expected_keys(cfg)
        write_json(metadata_path, metadata)

    if present == expected_keys(cfg):
        save()  # Regenerate a missing derived summary without retraining.
        print("Skip complete paired experiment: " + str(output), flush=True)
        return rows

    public_cache = None
    for seed in cfg['seeds']:
        required = {k for k in expected_keys(cfg) if k[2] == seed}
        if required <= present:
            continue
        if cfg['task'] == 'darcy' and public_cache is not None:
            data = dict(public_cache)
            data['partition_seed'] = base['partition'].get('partition_seed_base', 30000) + seed
        else:
            data = prepare_data(cfg, base, seed)
            if cfg['task'] == 'darcy':
                public_cache = data
        metadata['datasets'][str(seed)] = dict(
            identity=data['identity'], n_train=len(data['train_x']), n_test=len(data['test_x']),
            metadata=data['dataset_metadata'], model=data['model_name'],
            response_quantization_error=data['bins'].quantization_error,
            bin_sizes=np.bincount(data['bins'].labels, minlength=cfg['n_bins']).tolist())
        for alpha in [None] + cfg['alpha_values']:
            condition = condition_name(alpha)
            clients, split = split_data(cfg, data, alpha)
            metadata['partitions']['{}:{}'.format(seed, condition)] = split
            for method in methods:
                wanted = {(method, condition, seed, t) for t in cfg['round_checkpoints']}
                if wanted <= present:
                    print("Skip completed seed={} {} {}".format(seed, condition, method), flush=True)
                    continue
                rows[:] = [row for row in rows if key(row)[:3] != (method, condition, seed)]
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                model = data['factory']()
                training = dict(cfg['training'], batch_size=data['batch_size'])
                control = dict(cfg['control'], n_bins=cfg['n_bins'])
                label = "{} nu={} seed={} {}".format(cfg['task'], cfg['viscosity'], seed, condition)
                print("Start " + label + " " + method, flush=True)
                records, _ = train(
                    model, clients, data['test_x'], data['test_y'], training, control,
                    seed, method, cfg['round_checkpoints'], data['target_mean'],
                    data['target_std'], label)
                rows.extend(dict(task=cfg['task'], viscosity=cfg['viscosity'],
                                 method=method, condition=condition,
                                 alpha='' if alpha is None else alpha, seed=seed, **r)
                            for r in records)
                save()
                present = valid_keys(rows)
    save()
    print("Complete: " + str(output), flush=True)
    return rows
