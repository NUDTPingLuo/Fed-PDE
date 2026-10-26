"""Single-run engine for solution-space PDE-Dirichlet experiments."""
import os
import time

import numpy as np
from scipy.stats import t as student_t

from operator_tasks import OperatorTask
from util.io_utils import load_results, save_results
from util.partitioning import (
    client_bin_count_matrix,
    cluster_solution_responses,
    mean_pairwise_sample_transport,
    mean_pairwise_transport,
    pairwise_euclidean,
    partition_clients,
    partition_clients_iid,
    serialize_client_bin_counts,
)
from util.round_tracking import resolve_round_checkpoints
from util.result_contract import matching_metadata
from util.schedules import checkpoint_learning_rate
from util.training import (
    backend_description,
    eval_params,
    fedavg_train,
    gradient_dissimilarity,
    relative_parameter_divergence,
)


METHOD = 'solution_dirichlet'


def make_task(cfg):
    return OperatorTask(
        cfg['task'], cfg['n_sensors'], cfg['output_grid'],
        cfg['condition_scale'], n_modes=cfg['n_modes'],
        diffusion=cfg['diffusion'], reaction=cfg['reaction'],
        viscosity=cfg['viscosity'])


def model_spec(cfg, task):
    return dict(
        coordinates=task.output_coordinates,
        width=cfg['model_width'],
        depth=cfg['model_depth'],
        activation=cfg['model_activation'],
        backend=cfg.get('training_backend', 'numpy'),
        device=cfg.get('training_device', 'auto'),
        batch_size=cfg.get('training_batch_size'),
        mixed_precision=cfg.get('mixed_precision', False),
        momentum=cfg.get('momentum', 0.0),
        nesterov=cfg.get('nesterov', False),
        lr_schedule=cfg.get('lr_schedule', 'cosine'),
        min_lr=cfg.get('min_lr', 0.0001),
    )


def _mean_pairwise_distance(values):
    distances = pairwise_euclidean(values)
    upper = distances[np.triu_indices(len(values), 1)]
    result = float(np.mean(upper))
    if result <= 1e-12:
        raise ValueError("training inputs are effectively identical")
    return result


def prepare_seed_data(cfg, task, seed):
    """Generate one complete labeled training set and an independent test set."""
    conditions = task.sample_conditions(
        cfg['sampling_pool_size'], cfg['seed_base'] + 10000 + seed)
    test_conditions = task.sample_conditions(
        cfg['n_test'], cfg['seed_base'] + 50000 + seed)
    responses = task.solve_batch(conditions)
    bins = cluster_solution_responses(
        responses, cfg['n_bins'], cfg['seed_base'] + 20000 + seed)
    return dict(
        conditions=conditions,
        responses=responses,
        solution_bins=bins,
        input_normalizer=_mean_pairwise_distance(conditions),
        test_conditions=test_conditions,
        test_responses=task.solve_batch(test_conditions),
    )


def make_solution_split(cfg, alpha, seed, data):
    """Partition one labeled training set through its solution-space bins."""
    bins = data['solution_bins']
    partition_seed = cfg['seed_base'] + 30000 + seed
    if alpha is None:
        indices, target_hist, realized_hist = partition_clients_iid(
            bins.labels, cfg['n_clients'], cfg['min_client_samples'],
            seed=partition_seed)
    else:
        indices, target_hist, realized_hist = partition_clients(
            bins.labels, cfg['n_clients'], alpha, cfg['min_client_samples'],
            seed=partition_seed)
    sizes = np.asarray([len(index) for index in indices], dtype=int)
    client_bin_counts = client_bin_count_matrix(
        bins.labels, indices, cfg['n_bins'])
    client_x = [data['conditions'][index] for index in indices]
    client_u = [data['responses'][index] for index in indices]
    input_w1 = mean_pairwise_sample_transport(
        client_x, data['input_normalizer'], cfg['transport_samples'], seed=seed)
    return dict(
        client_x=client_x,
        client_u=client_u,
        test_x=data['test_conditions'],
        test_u=data['test_responses'],
        sizes=sizes,
        client_bin_counts=client_bin_counts,
        bin_sizes=np.bincount(bins.labels, minlength=cfg['n_bins']),
        target_distance=mean_pairwise_transport(
            target_hist, bins.response_cost),
        distance=mean_pairwise_transport(realized_hist, bins.response_cost),
        input_w1=input_w1,
        partition_target_l1=float(np.mean(
            np.sum(np.abs(realized_hist - target_hist), axis=1))),
        response_quantization_error=bins.quantization_error,
        clustering_iterations=bins.iterations,
    )


def evaluation_rounds(cfg):
    requested = cfg.get('round_checkpoints') or None
    return resolve_round_checkpoints(cfg['n_rounds'], requested)

def train_federated_split(cfg, spec, split, seed):
    """Train one FedAvg partition and retain selected communication rounds."""
    gradient = gradient_dissimilarity(
        split['client_x'], split['client_u'], spec, init_seed=seed)
    rounds = evaluation_rounds(cfg)
    federated, history = fedavg_train(
        split['client_x'], split['client_u'], cfg['n_rounds'],
        cfg['local_steps'], cfg['lr'], cfg['optimizer'], spec, init_seed=seed,
        checkpoint_rounds=rounds)
    round_errors = {
        round_index: eval_params(
            history[round_index], split['test_x'], split['test_u'], spec)
        for round_index in rounds
    }
    return dict(
        state=federated,
        error=round_errors[cfg['n_rounds']],
        round_errors=round_errors,
        gradient=gradient,
    )


def compare_with_iid_reference(cfg, candidate, iid_reference):
    """Measure non-IID degradation relative to uniform-IID FedAvg."""
    round_history = []
    for round_index in evaluation_rounds(cfg):
        iid_error = iid_reference['round_errors'][round_index]
        federated_error = candidate['round_errors'][round_index]
        round_history.append(dict(
            round=round_index,
            iid_error=iid_error,
            federated_error=federated_error,
            drift=federated_error - iid_error,
        ))
    return dict(
        iid_error=iid_reference['error'],
        federated_error=candidate['error'],
        drift=candidate['error'] - iid_reference['error'],
        gradient=candidate['gradient'],
        iid_weight_divergence=relative_parameter_divergence(
            iid_reference['state'], candidate['state']),
        round_history=round_history,
    )


def seed_result_row(cfg, alpha, seed, run):
    """Flatten one seed replicate into an analysis-ready CSV record."""
    sizes = np.asarray(run['sizes'], dtype=int)
    bin_sizes = np.asarray(run['bin_sizes'], dtype=int)
    scalar_keys = (
        'target_distance', 'distance', 'input_w1', 'partition_target_l1',
        'response_quantization_error', 'clustering_iterations',
        'gradient', 'iid_weight_divergence',
    )
    row = dict(
        task=cfg['task'], model='CartesianDeepONet', method=METHOD,
        optimizer=cfg['optimizer'], learning_rate=cfg['lr'],
        momentum=cfg.get('momentum', 0.0),
        severity_name='alpha', severity_value=alpha, seed=seed,
        client_samples=';'.join(str(value) for value in sizes.tolist()),
        client_bin_counts=serialize_client_bin_counts(
            run['client_bin_counts']),
        iid_client_bin_counts=serialize_client_bin_counts(
            run['iid_client_bin_counts']),
        client_samples_std=float(np.std(sizes)),
        client_samples_cv=float(np.std(sizes) / max(np.mean(sizes), 1e-12)),
        client_samples_min=int(np.min(sizes)),
        client_samples_max=int(np.max(sizes)),
        bin_samples=';'.join(str(value) for value in bin_sizes.tolist()),
        bin_samples_min=int(np.min(bin_sizes)),
        bin_samples_max=int(np.max(bin_sizes)),
    )
    row.update({key: run[key] for key in scalar_keys})
    return row


def round_seed_result_rows(cfg, alpha, seed, run):
    """Flatten dynamic metrics at the configured communication checkpoints."""
    return [dict(
        task=cfg['task'], model='CartesianDeepONet', method=METHOD,
        optimizer=cfg['optimizer'], learning_rate=cfg['lr'],
        momentum=cfg.get('momentum', 0.0),
        severity_name='alpha', severity_value=alpha, seed=seed,
        round=record['round'], iid_error=record['iid_error'],
        current_learning_rate=checkpoint_learning_rate(cfg, record['round']),
        federated_error=record['federated_error'], drift=record['drift'],
    ) for record in run['round_history']]


def iid_result_rows(cfg, reference_cache):
    """Store each IID seed/checkpoint once in its own condition file."""
    rows = []
    for seed in cfg['seeds']:
        if seed not in reference_cache:
            continue
        reference = reference_cache[seed]
        split, training = reference['split'], reference['training']
        sizes = np.asarray(split['sizes'], dtype=int)
        common = dict(
            task=cfg['task'], model='CartesianDeepONet',
            method='solution_iid', optimizer=cfg['optimizer'],
            learning_rate=cfg['lr'], momentum=cfg.get('momentum', 0.0),
            partition='iid', seed=seed,
            client_samples=';'.join(str(value) for value in sizes.tolist()),
            client_bin_counts=serialize_client_bin_counts(
                split['client_bin_counts']),
            client_samples_cv=float(
                np.std(sizes) / max(np.mean(sizes), 1e-12)),
            client_samples_min=int(np.min(sizes)),
            client_samples_max=int(np.max(sizes)),
            distance=split['distance'], input_w1=split['input_w1'],
            response_quantization_error=split['response_quantization_error'],
            gradient=training['gradient'])
        for round_index in evaluation_rounds(cfg):
            rows.append(dict(
                **common, round=round_index,
                current_learning_rate=checkpoint_learning_rate(cfg, round_index),
                test_error=training['round_errors'][round_index]))
    return rows


def _confidence_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(np.mean(values)) if len(values) else float('nan')
    if len(values) > 1:
        std = float(np.std(values, ddof=1))
        half = float(
            student_t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values))
        )
    else:
        std, half = 0.0, 0.0
    return mean, std, mean - half, mean + half, half


def summarize(cfg, alpha, runs):
    array = lambda key: np.asarray([run[key] for run in runs], dtype=float)
    sizes = [np.asarray(run['sizes']) for run in runs]
    bin_sizes = [np.asarray(run['bin_sizes']) for run in runs]
    row = dict(
        task=cfg['task'], model='CartesianDeepONet', method=METHOD,
        optimizer=cfg['optimizer'], learning_rate=cfg['lr'],
        momentum=cfg.get('momentum', 0.0),
        severity_name='alpha', severity_value=alpha,
        distance_target_mean=float(np.mean(array('target_distance'))),
        distance_mean=float(np.mean(array('distance'))),
        input_w1_mean=float(np.mean(array('input_w1'))),
        partition_target_l1_mean=float(np.mean(array('partition_target_l1'))),
        response_quantization_error_mean=float(np.mean(
            array('response_quantization_error'))),
        clustering_iterations_mean=float(np.mean(array('clustering_iterations'))),
        err_iid_mean=float(np.mean(array('iid_error'))),
        err_fedavg_mean=float(np.mean(array('federated_error'))),
        drift_mean=float(np.mean(array('drift'))),
        gradient_dissimilarity_mean=float(np.mean(array('gradient'))),
        iid_weight_divergence_mean=float(np.mean(
            array('iid_weight_divergence'))),
        client_samples_std=float(np.mean([np.std(value) for value in sizes])),
        client_samples_cv_mean=float(np.mean([
            np.std(value) / max(np.mean(value), 1e-12) for value in sizes
        ])),
        client_samples_min=int(min(np.min(value) for value in sizes)),
        client_samples_max=int(max(np.max(value) for value in sizes)),
        bin_samples_min=int(min(np.min(value) for value in bin_sizes)),
        bin_samples_max=int(max(np.max(value) for value in bin_sizes)),
        n_seeds=cfg['n_seeds'],
    )
    ci_fields = {
        'distance': 'distance',
        'input_w1': 'input_w1',
        'err_iid': 'iid_error',
        'err_fedavg': 'federated_error',
        'drift': 'drift',
        'gradient_dissimilarity': 'gradient',
        'iid_weight_divergence': 'iid_weight_divergence',
    }
    for prefix, key in ci_fields.items():
        _, std, low, high, half = _confidence_interval(array(key))
        row[f'{prefix}_std'] = std
        row[f'{prefix}_ci95_low'] = low
        row[f'{prefix}_ci95_high'] = high
        row[f'{prefix}_ci95_half_width'] = half
    return row


def distance_correlations(seed_rows, round_seed_rows):
    """Compare solution transport and input W1 on all seed-level replicates."""
    final_round_rows = {}
    for row in round_seed_rows:
        key = (float(row['severity_value']), int(float(row['seed'])))
        if (key not in final_round_rows
                or float(row['round']) > float(final_round_rows[key]['round'])):
            final_round_rows[key] = row
    targets = (
        'gradient_dissimilarity', 'iid_weight_divergence',
        'non_iid_degradation')
    result = {}
    for distance_name, distance_key in (
            ('solution_transport', 'distance'), ('input_w1', 'input_w1')):
        x = np.asarray([row[distance_key] for row in seed_rows], dtype=float)
        result[distance_name] = {}
        for target_name in targets:
            if target_name == 'non_iid_degradation':
                y = np.asarray([
                    final_round_rows[(float(row['severity_value']),
                                      int(float(row['seed'])))]['drift']
                    for row in seed_rows], dtype=float)
            else:
                target_key = (
                    'gradient' if target_name == 'gradient_dissimilarity'
                    else 'iid_weight_divergence')
                y = np.asarray([row[target_key] for row in seed_rows], dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            enough = (
                np.sum(finite) > 1 and np.std(x[finite]) > 1e-12
                and np.std(y[finite]) > 1e-12
            )
            result[distance_name][target_name] = (
                float(np.corrcoef(x[finite], y[finite])[0, 1])
                if enough else float('nan')
            )
    return result


def _metadata(cfg, seed_data):
    return {
        'protocol': {
            'training_labels': 'high-fidelity PDE responses for every training sample',
            'continuous_to_discrete': (
                'K-means++ on the complete normalized training solution set, '
                'fitted independently for each seed'),
            'bin_reuse': 'one fixed set of bin labels per seed for every alpha',
            'client_partition': (
                'non-reference rows use independent per-bin symmetric Dirichlet '
                'allocation; all rows are exhaustive and strictly without replacement'),
            'quantity_heterogeneity': (
                'induced by the same allocation; FedAvg uses realized sample-count weights'),
            'iid_reference': (
                'an independent uniform-IID reference, not a Dirichlet alpha: '
                'every solution bin '
                'and the total sample count are balanced across clients up to unavoidable '
                'integer remainders'),
            'drift_definition': (
                'FedAvg test relative-L2 error at alpha minus the same-seed '
                'FedAvg error under the uniform-IID partition'),
            'minimum_client_samples': cfg['min_client_samples'],
            'local_training': (
                f"{cfg['local_steps']} shuffled mini-batch optimizer steps per client "
                'and communication round'),
            'reference_training': (
                'one uniform-IID FedAvg model per seed, reused as the comparison '
                'reference for every non-IID alpha'),
            'seed_level_results': (
                'Dirichlet rows are stored in '
                '<output_prefix>_alpha_seed_raw.csv'),
            'client_bin_counts': (
                'exact client-by-solution-bin counts in seed-level results; '
                "rows are separated by '|' and columns by ';'"),
            'round_checkpoints': evaluation_rounds(cfg),
            'round_level_results': (
                'Dirichlet checkpoints are stored in '
                '<output_prefix>_alpha_round_seed_raw.csv; the matched IID '
                'checkpoints are stored once in <output_prefix>_iid.csv'),
        },
        'solution_binning': {
            str(seed): {
                'n_training_samples': len(data['responses']),
                'n_bins': cfg['n_bins'],
                'bin_sizes': np.bincount(
                    data['solution_bins'].labels,
                    minlength=cfg['n_bins']).tolist(),
                'response_normalizer': data['solution_bins'].response_normalizer,
                'quantization_error': data['solution_bins'].quantization_error,
                'iterations': data['solution_bins'].iterations,
            }
            for seed, data in seed_data.items()
        },
    }


def run_experiment(cfg):
    stem = os.path.join(cfg['output_dir'], cfg['output_prefix'])
    if cfg.get('resume', False) and any(os.path.exists(stem + suffix) for suffix in
            ('_alpha.csv', '_alpha_seed_raw.csv', '_alpha_round_seed_raw.csv', '_iid.csv')):
        if not matching_metadata(stem + '_metadata.json', cfg):
            raise ValueError('Existing results use different settings or lack metadata; '
                             'use --force or a separate output directory.')
    task = make_task(cfg)
    spec = model_spec(cfg, task)
    print('=' * 78)
    print(f"Task: {cfg['task']} | model: CartesianDeepONet "
          f"(width={cfg['model_width']}, depth={cfg['model_depth']})")
    print(f"Training backend: {backend_description(spec)} | "
          f"batch_size={spec['batch_size'] or 'full'} | "
          f"mixed_precision={spec['mixed_precision']}")
    print(f"Optimizer: {cfg['optimizer']} | learning_rate={cfg['lr']:g} | "
          f"momentum={cfg.get('momentum', 0.0):g} | "
          f"nesterov={cfg.get('nesterov', False)} | "
          f"cosine_min_lr={cfg.get('min_lr', 0.0001):g}")
    print(f"Schedule: cosine | first_round_lr={cfg['lr']:g} | "
          f"last_round_lr={checkpoint_learning_rate(cfg, cfg['n_rounds']):g}")
    print(
        "Protocol: uniform-IID reference + solution-bin Dirichlet "
        f"non-IID | non-IID alphas: {cfg['alpha_values']}")
    print("Phase 1/2: generating labeled PDE data and clustering training solutions")
    print('=' * 78)
    seed_data = {}
    for seed in cfg['seeds']:
        print(f"Preparing seed {seed}: solve train/test PDEs and fit {cfg['n_bins']} bins")
        seed_data[seed] = prepare_seed_data(cfg, task, seed)
    metadata = _metadata(cfg, seed_data)
    print("Phase 2/2: training uniform-IID reference and non-IID FedAvg models")

    checkpoint_path = os.path.join(
        cfg['output_dir'], cfg['output_prefix'] + '_alpha.csv')
    seed_checkpoint_path = os.path.join(
        cfg['output_dir'], cfg['output_prefix'] + '_alpha_seed_raw.csv')
    round_seed_checkpoint_path = os.path.join(
        cfg['output_dir'], cfg['output_prefix'] + '_alpha_round_seed_raw.csv')
    rows = load_results(checkpoint_path) if cfg.get('resume', False) else []
    seed_rows = load_results(seed_checkpoint_path) if cfg.get('resume', False) else []
    round_seed_rows = (load_results(round_seed_checkpoint_path)
                       if cfg.get('resume', False) else [])
    raw_counts = {}
    for raw in seed_rows:
        if not raw.get('client_bin_counts') or 'iid_weight_divergence' not in raw:
            continue
        key = float(raw['severity_value'])
        raw_counts[key] = raw_counts.get(key, 0) + 1
    round_raw_counts = {}
    for raw in round_seed_rows:
        if 'iid_error' not in raw:
            continue
        key = float(raw['severity_value'])
        round_raw_counts[key] = round_raw_counts.get(key, 0) + 1
    checkpoint_count = len(evaluation_rounds(cfg))
    completed = {
        float(row['severity_value']) for row in rows
        if row.get('method') == METHOD
        and raw_counts.get(float(row['severity_value']), 0) == cfg['n_seeds']
        and round_raw_counts.get(float(row['severity_value']), 0)
        == cfg['n_seeds'] * checkpoint_count
    }
    rows = [row for row in rows if row.get('method') == METHOD
            and float(row['severity_value']) in completed]
    seed_rows = [row for row in seed_rows if row.get('method') == METHOD
                 and float(row['severity_value']) in completed]
    round_seed_rows = [
        row for row in round_seed_rows if row.get('method') == METHOD
        and float(row['severity_value']) in completed]
    if rows:
        print(f"Resume from {len(rows)} completed alpha settings")

    start = time.time()
    reference_cache = {}

    def iid_reference(seed, data):
        if seed not in reference_cache:
            print(
                f"Training uniform-IID FedAvg reference for seed {seed}")
            split = make_solution_split(cfg, None, seed, data)
            reference_cache[seed] = dict(
                split=split,
                training=train_federated_split(cfg, spec, split, seed),
            )
        return reference_cache[seed]

    for alpha in cfg['alpha_values']:
        if float(alpha) in completed:
            continue
        runs = []
        for seed, data in seed_data.items():
            print(f"alpha={alpha:g}, seed={seed}: partition and train FedAvg")
            reference = iid_reference(seed, data)
            split = make_solution_split(cfg, alpha, seed, data)
            candidate = train_federated_split(cfg, spec, split, seed)
            run = {**split, **compare_with_iid_reference(
                cfg, candidate, reference['training'])}
            run['iid_client_bin_counts'] = reference['split']['client_bin_counts']
            runs.append(run)
            seed_rows.append(seed_result_row(cfg, alpha, seed, run))
            round_seed_rows.extend(
                round_seed_result_rows(cfg, alpha, seed, run))
        row = summarize(cfg, alpha, runs)
        rows.append(row)
        if cfg.get('checkpoint', False):
            metadata['complete'] = False
            metadata['correlations'] = distance_correlations(
                seed_rows, round_seed_rows)
            save_results(
                rows, cfg, metadata, seed_rows,
                round_seed_rows, iid_result_rows(cfg, reference_cache))
        print(f"[{time.time() - start:6.1f}s] alpha={alpha:g} "
              f"D={row['distance_mean']:.3f} drift={row['drift_mean']:.3%}")

    metadata['complete'] = True
    metadata['correlations'] = distance_correlations(
        seed_rows, round_seed_rows)
    save_results(
        rows, cfg, metadata, seed_rows,
        round_seed_rows, iid_result_rows(cfg, reference_cache))
    return rows
