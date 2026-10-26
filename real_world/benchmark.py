"""Solution-bin Dirichlet benchmark on public application PDE datasets."""
import os

import numpy as np
from util.schedules import checkpoint_learning_rate
from scipy.stats import t as student_t

from util.io_utils import save_results
from util.partitioning import (
    client_bin_count_matrix, cluster_solution_responses,
    mean_pairwise_transport, partition_clients, partition_clients_iid,
    serialize_client_bin_counts)
from util.round_tracking import resolve_round_checkpoints

from .datasets import load_public_dataset
from .training import (
    fedavg_train, gradient_dissimilarity, relative_error, state_divergence)


METHOD = 'solution_dirichlet_public_pde'


def _evaluation_rounds(cfg):
    requested = cfg['evaluation'].get('round_checkpoints') or None
    return resolve_round_checkpoints(cfg['training']['n_rounds'], requested)


def _normalization(values):
    mean = np.mean(values, axis=(0, 2, 3), keepdims=True)
    std = np.std(values, axis=(0, 2, 3), keepdims=True)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def _normalize_data(data):
    input_mean, input_std = _normalization(data.train_inputs)
    target_mean, target_std = _normalization(data.train_targets)
    return dict(
        train_x=(data.train_inputs - input_mean) / input_std,
        train_y=(data.train_targets - target_mean) / target_std,
        test_x=(data.test_inputs - input_mean) / input_std,
        test_y=(data.test_targets - target_mean) / target_std,
        target_mean=target_mean,
        target_std=target_std,
    )


def _ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0, mean, mean, 0.0
    std = float(np.std(values, ddof=1))
    half = float(student_t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
    return mean, std, mean - half, mean + half, half


def _summarize(cfg, alpha, runs):
    array = lambda key: np.asarray([run[key] for run in runs], dtype=float)
    sizes = [run['sizes'] for run in runs]
    row = dict(
        task=cfg['task'], dataset=cfg['dataset']['type'],
        model=cfg['model']['type'], method=METHOD,
        optimizer=cfg['training']['optimizer'],
        learning_rate=cfg['training']['lr'],
        momentum=cfg['training'].get('momentum', 0.0),
        severity_name='alpha', severity_value=alpha,
        distance_target_mean=float(np.mean(array('target_distance'))),
        distance_mean=float(np.mean(array('distance'))),
        partition_target_l1_mean=float(np.mean(array('partition_target_l1'))),
        response_quantization_error_mean=runs[0]['quantization_error'],
        err_iid_mean=float(np.mean(array('iid_error'))),
        err_fedavg_mean=float(np.mean(array('federated_error'))),
        drift_mean=float(np.mean(array('drift'))),
        gradient_dissimilarity_mean=float(np.mean(array('gradient'))),
        iid_weight_divergence_mean=float(np.mean(
            array('iid_weight_divergence'))),
        client_samples_cv_mean=float(np.mean([
            np.std(value) / np.mean(value) for value in sizes])),
        client_samples_min=int(min(np.min(value) for value in sizes)),
        client_samples_max=int(max(np.max(value) for value in sizes)),
        n_seeds=len(cfg['evaluation']['seeds']),
    )
    for prefix, key in (
            ('distance', 'distance'),
            ('err_iid', 'iid_error'),
            ('err_fedavg', 'federated_error'),
            ('drift', 'drift'),
            ('gradient_dissimilarity', 'gradient'),
            ('iid_weight_divergence', 'iid_weight_divergence')):
        _, std, low, high, half = _ci(array(key))
        row[f'{prefix}_std'] = std
        row[f'{prefix}_ci95_low'] = low
        row[f'{prefix}_ci95_high'] = high
        row[f'{prefix}_ci95_half_width'] = half
    return row


def _seed_row(cfg, alpha, seed, run):
    return dict(
        task=cfg['task'], dataset=cfg['dataset']['type'],
        model=cfg['model']['type'], method=METHOD,
        optimizer=cfg['training']['optimizer'],
        learning_rate=cfg['training']['lr'],
        momentum=cfg['training'].get('momentum', 0.0),
        severity_name='alpha', severity_value=alpha, seed=seed,
        client_samples=';'.join(str(value) for value in run['sizes']),
        client_bin_counts=serialize_client_bin_counts(
            run['client_bin_counts']),
        iid_client_bin_counts=serialize_client_bin_counts(
            run['iid_client_bin_counts']),
        client_samples_cv=float(np.std(run['sizes']) / np.mean(run['sizes'])),
        target_distance=run['target_distance'], distance=run['distance'],
        partition_target_l1=run['partition_target_l1'],
        response_quantization_error=run['quantization_error'],
        gradient=run['gradient'],
        iid_weight_divergence=run['iid_weight_divergence'])


def _round_seed_rows(cfg, alpha, seed, run):
    return [dict(
        task=cfg['task'], dataset=cfg['dataset']['type'],
        model=cfg['model']['type'], method=METHOD,
        optimizer=cfg['training']['optimizer'],
        learning_rate=cfg['training']['lr'],
        momentum=cfg['training'].get('momentum', 0.0),
        severity_name='alpha', severity_value=alpha, seed=seed,
        round=record['round'], iid_error=record['iid_error'],
        current_learning_rate=checkpoint_learning_rate(cfg['training'], record['round']),
        federated_error=record['federated_error'], drift=record['drift'],
    ) for record in run['round_history']]


def _iid_rows(cfg, bins, reference_cache, rounds):
    """Store one explicit IID trajectory per seed without alpha duplication."""
    rows = []
    for seed in cfg['evaluation']['seeds']:
        if seed not in reference_cache:
            continue
        reference = reference_cache[seed]
        split, training = reference['split'], reference['training']
        sizes = np.asarray([len(index) for index in split['indices']])
        counts = client_bin_count_matrix(
            bins.labels, split['indices'], cfg['partition']['n_bins'])
        common = dict(
            task=cfg['task'], dataset=cfg['dataset']['type'],
            model=cfg['model']['type'], method='solution_iid',
            optimizer=cfg['training']['optimizer'],
            learning_rate=cfg['training']['lr'],
            momentum=cfg['training'].get('momentum', 0.0),
            partition='iid', seed=seed,
            client_samples=';'.join(str(value) for value in sizes.tolist()),
            client_bin_counts=serialize_client_bin_counts(counts),
            client_samples_cv=float(np.std(sizes) / np.mean(sizes)),
            distance=split['distance'])
        for round_index in rounds:
            rows.append(dict(
                **common, round=round_index,
                current_learning_rate=checkpoint_learning_rate(cfg['training'], round_index),
                test_error=training['round_errors'][round_index]))
    return rows


def run_public_benchmark(cfg):
    data = load_public_dataset(cfg['dataset'])
    normalized = _normalize_data(data)
    flattened = normalized['train_y'].reshape(len(normalized['train_y']), -1)
    bins = cluster_solution_responses(
        flattened, cfg['partition']['n_bins'],
        cfg['partition'].get('clustering_seed', 42),
        max_iter=cfg['partition'].get('clustering_max_iter', 100))
    train_cfg = dict(cfg['training'])
    train_cfg['model'] = cfg['model']
    os.makedirs(cfg['output']['dir'], exist_ok=True)
    result_cfg = dict(
        task=cfg['task'], output_dir=cfg['output']['dir'],
        output_prefix=cfg['output']['prefix'],
        model_name=cfg['model']['type'])
    rows, seed_rows, round_seed_rows = [], [], []
    reference_cache = {}
    rounds = _evaluation_rounds(cfg)

    def make_split(alpha, seed):
        partition_seed = (
            cfg['partition'].get('partition_seed_base', 30000) + int(seed))
        if alpha is None:
            indices, target_hist, realized_hist = partition_clients_iid(
                bins.labels, cfg['federation']['n_clients'],
                cfg['partition']['min_client_samples'], partition_seed)
        else:
            indices, target_hist, realized_hist = partition_clients(
                bins.labels, cfg['federation']['n_clients'], float(alpha),
                cfg['partition']['min_client_samples'], partition_seed)
        return dict(
            indices=indices,
            target_hist=target_hist,
            realized_hist=realized_hist,
            # Validate transport before any long training run, and reuse it in
            # result writing instead of solving the same LPs repeatedly.
            target_distance=mean_pairwise_transport(
                target_hist, bins.response_cost),
            distance=mean_pairwise_transport(
                realized_hist, bins.response_cost),
            client_x=[normalized['train_x'][index] for index in indices],
            client_y=[normalized['train_y'][index] for index in indices],
        )

    def train_split(split, seed):
        state, history = fedavg_train(
            split['client_x'], split['client_y'], train_cfg, seed,
            checkpoint_rounds=rounds)
        round_errors = {
            round_index: relative_error(
                history[round_index], normalized['test_x'],
                normalized['test_y'], train_cfg, normalized['target_mean'],
                normalized['target_std'])
            for round_index in rounds
        }
        return dict(
            state=state,
            error=round_errors[cfg['training']['n_rounds']],
            round_errors=round_errors,
        )

    def iid_reference(seed):
        if seed not in reference_cache:
            split = make_split(None, seed)
            reference_cache[seed] = dict(
                split=split, training=train_split(split, seed))
        return reference_cache[seed]

    for alpha in cfg['partition']['alpha_values']:
        runs = []
        for seed in cfg['evaluation']['seeds']:
            split = make_split(alpha, seed)
            reference = iid_reference(seed)
            candidate = train_split(split, seed)
            indices = split['indices']
            target_hist = split['target_hist']
            realized_hist = split['realized_hist']
            iid_training = reference['training']
            round_history = [dict(
                round=round_index,
                iid_error=iid_training['round_errors'][round_index],
                federated_error=candidate['round_errors'][round_index],
                drift=(candidate['round_errors'][round_index]
                       - iid_training['round_errors'][round_index]),
            ) for round_index in rounds]
            run = dict(
                sizes=np.asarray([len(index) for index in indices]),
                client_bin_counts=client_bin_count_matrix(
                    bins.labels, indices, cfg['partition']['n_bins']),
                iid_client_bin_counts=client_bin_count_matrix(
                    bins.labels, reference['split']['indices'],
                    cfg['partition']['n_bins']),
                target_distance=split['target_distance'],
                distance=split['distance'],
                partition_target_l1=float(np.mean(
                    np.sum(np.abs(realized_hist - target_hist), axis=1))),
                quantization_error=bins.quantization_error,
                iid_error=iid_training['error'],
                federated_error=candidate['error'],
                drift=candidate['error'] - iid_training['error'],
                gradient=gradient_dissimilarity(
                    split['client_x'], split['client_y'], train_cfg, seed),
                iid_weight_divergence=state_divergence(
                    iid_training['state'], candidate['state']),
                round_history=round_history)
            runs.append(run)
            seed_rows.append(_seed_row(cfg, alpha, seed, run))
            round_seed_rows.extend(_round_seed_rows(cfg, alpha, seed, run))
            print(
                f"{cfg['task']} alpha={alpha:g} seed={seed}: "
                f"D={run['distance']:.3f}, drift={100 * run['drift']:.3f} pp")
        rows.append(_summarize(cfg, alpha, runs))
        metadata = {
            'complete': False, 'config': cfg, 'dataset': data.metadata,
            'round_tracking': {
                'round_checkpoints': rounds,
                'seed_file': '<output_prefix>_alpha_round_seed_raw.csv',
                'summary_policy': (
                    'round summaries are derived downstream from seed-level rows'),
            },
            'iid_reference': {
                'interpretation': (
                    'independent uniform-IID allocation, not a Dirichlet alpha; '
                    'every solution bin and total client size '
                    'are balanced up to unavoidable integer remainders'),
                'drift_definition': (
                    'FedAvg error at alpha minus same-seed FedAvg error at the '
                    'uniform-IID reference'),
            },
            'client_bin_counts': (
                'exact client-by-solution-bin counts in seed-level results; '
                "rows are separated by '|' and columns by ';'"),
            'solution_binning': {
                'bin_sizes': np.bincount(
                    bins.labels, minlength=cfg['partition']['n_bins']).tolist(),
                'response_normalizer': bins.response_normalizer,
                'quantization_error': bins.quantization_error,
                'iterations': bins.iterations,
            },
        }
        save_results(
            rows, result_cfg, metadata, seed_rows,
            round_seed_rows, _iid_rows(
                cfg, bins, reference_cache, rounds))
    metadata['complete'] = True
    save_results(
        rows, result_cfg, metadata, seed_rows,
        round_seed_rows, _iid_rows(
            cfg, bins, reference_cache, rounds))
    return rows, seed_rows
