"""Sample-weighted SB-CFO on a shared PyTorch model.

Clients compute bin gradients locally at the broadcast model. Only conditional
gradient estimates and counts enter prototype aggregation. This is a single-
process FL simulator, not a secure aggregation or differential privacy system.
"""
import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from util.schedules import round_learning_rate, validate_schedule


@dataclass
class ClientShard:
    inputs: np.ndarray
    targets: np.ndarray
    labels: np.ndarray


def real_vector(tensors):
    """Real coordinates preserve the Euclidean norm of complex FNO weights."""
    parts = []
    for tensor in tensors:
        value = tensor.detach()
        if value.is_complex():
            value = torch.view_as_real(value.resolve_conj())
        parts.append(value.reshape(-1).cpu())
    return torch.cat(parts)


def add_correction(model, correction):
    """Add q (not subtract q) BEFORE optimizer.step, in parameter order."""
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel() * (2 if parameter.is_complex() else 1)
            value = correction[offset:offset + size].to(parameter.device)
            if parameter.is_complex():
                value = torch.view_as_complex(
                    # A contiguous slice can still have an odd storage offset.
                    # Clone so view_as_complex always starts at offset zero.
                    value.reshape(*parameter.shape, 2).clone(memory_format=torch.contiguous_format))
            else:
                value = value.reshape(parameter.shape)
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad.add_(value.to(parameter.dtype))
            offset += size
    if offset != correction.numel():
        raise ValueError("correction dimension does not match model parameters")


def count_matrix(clients, n_bins):
    counts = []
    for client in clients:
        labels = np.asarray(client.labels)
        if (len(client.inputs) == 0 or len(client.inputs) != len(client.targets)
                or labels.shape != (len(client.inputs),)
                or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < 0) or np.any(labels >= n_bins)):
            raise ValueError("invalid or empty client shard")
        counts.append(np.bincount(labels, minlength=n_bins))
    counts = np.asarray(counts, dtype=np.int64)
    if np.any(counts.sum(axis=0) == 0):
        raise ValueError("every global bin must contain training examples")
    return counts


def mean_gradient(model, client, indices, batch_size, device):
    """Exact gradient of the mean MSE on the selected local examples."""
    model.zero_grad(set_to_none=True)
    model.train()
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        x = torch.as_tensor(client.inputs[batch], dtype=torch.float32, device=device)
        y = torch.as_tensor(client.targets[batch], dtype=torch.float32, device=device)
        loss = (model(x) - y).square().mean()
        (loss * (len(batch) / len(indices))).backward()
    values = [p.grad if p.grad is not None else torch.zeros_like(p)
              for p in model.parameters()]
    vector = real_vector(values)
    if not torch.isfinite(vector).all():
        raise FloatingPointError("non-finite bin gradient; check learning rate/data")
    return vector


def bin_gradient_snapshot(model, clients, n_bins, batch_size, device,
                          samples_per_client_bin=0, seed=0):
    """Aggregate c_b = sum_k (n_kb/N_b) * local_mean_gradient_kb.

    Zero sampling cap uses ALL training examples. With a positive cap, each
    occupied client-bin is sampled uniformly without replacement; aggregation
    still uses its actual count n_kb, not its sampled count. Test labels never
    enter this routine. The sampling RNG is separate from the local-step RNG.
    """
    counts = count_matrix(clients, n_bins)
    rng = np.random.RandomState(seed)
    dimension = real_vector(model.parameters()).numel()
    prototypes = torch.zeros(n_bins, dimension)
    client_gradients = torch.zeros(len(clients), dimension)
    examples = 0
    for k, client in enumerate(clients):
        for b in np.flatnonzero(counts[k]):
            indices = np.flatnonzero(client.labels == b)
            if samples_per_client_bin > 0 and len(indices) > samples_per_client_bin:
                indices = rng.choice(indices, samples_per_client_bin, replace=False)
            gradient = mean_gradient(model, client, indices, batch_size, device)
            prototypes[b].add_(gradient, alpha=float(counts[k, b] / counts[:, b].sum()))
            client_gradients[k].add_(gradient, alpha=float(counts[k, b] / counts[k].sum()))
            examples += len(indices)
    return dict(prototypes=prototypes, gradients=client_gradients,
                counts=counts, examples=examples)


def correction_vectors(counts, prototypes, strength=1.0):
    counts = np.asarray(counts, dtype=np.float64)
    histograms = counts / counts.sum(axis=1, keepdims=True)
    pi = counts.sum(axis=0) / counts.sum()
    coefficients = torch.as_tensor(pi[None] - histograms, dtype=prototypes.dtype)
    return float(strength) * (coefficients @ prototypes)


def gradient_diagnostics(snapshot, applied):
    """Common-model diagnostics, using full-data gradients, not noisy batches."""
    gradients = snapshot['gradients'].double()
    counts = snapshot['counts']
    weights = torch.as_tensor(counts.sum(1) / counts.sum(), dtype=torch.float64)
    mean = weights @ gradients
    before = torch.sum(weights * (gradients - mean).square().sum(1))
    corrected = gradients + applied.double()
    after = torch.sum(weights * (corrected - mean).square().sum(1))
    ideal = correction_vectors(counts, snapshot['prototypes']).double()
    residual = torch.sum(weights * (gradients + ideal - mean).square().sum(1))
    energy = torch.sum(weights * gradients.square().sum(1))
    ratio = lambda value: float(1 - value / before) if before > 1e-24 else float('nan')
    return dict(
        gradient_variance_before=float(before),
        gradient_variance_after=float(after),
        gradient_dissimilarity_before=float(before / energy.clamp_min(1e-24)),
        gradient_dissimilarity_after=float(after / energy.clamp_min(1e-24)),
        correction_ratio=ratio(after),
        exact_bin_correction_ratio=ratio(residual),
        weighted_correction_norm=float(torch.linalg.vector_norm(weights @ applied.double())),
    )


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def make_optimizer(model, training):
    if training['optimizer'] == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=training['lr'],
                               momentum=training.get('momentum', 0.0),
                               nesterov=training.get('nesterov', False))
    if training['optimizer'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=training['lr'])
    raise ValueError("optimizer must be sgd or adam")


def evaluate(model, inputs, targets, batch_size, device, target_mean=0., target_std=1.):
    model.eval()
    mean = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(target_std, dtype=torch.float32, device=device)
    numerator = torch.zeros((), device=device)
    denominator = torch.zeros((), device=device)
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            x = torch.as_tensor(inputs[start:start + batch_size], dtype=torch.float32, device=device)
            y = torch.as_tensor(targets[start:start + batch_size], dtype=torch.float32, device=device)
            prediction = model(x) * std + mean
            truth = y * std + mean
            numerator += (prediction - truth).square().sum()
            denominator += truth.square().sum()
    result = float(torch.sqrt(numerator / denominator.clamp_min(1e-24)).cpu())
    if not np.isfinite(result):
        raise FloatingPointError("non-finite test error; no result will be marked complete")
    return result


def participation_schedule(n_clients, n_rounds, rate, seed):
    """Independent paired schedule; IDs zero-based, max(1, floor(rate*K)) visits.

    Uniform sampling without replacement each round. Sorting preserves the
    original aggregation order at full participation.
    """
    if not np.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError('participation_rate must be finite and in (0,1]')
    size = max(1, int(np.floor(rate * n_clients)))
    if size == n_clients:
        return [list(range(n_clients)) for _ in range(n_rounds)]
    rng = np.random.RandomState((int(seed) + 700001) % 2**32)
    return [sorted(rng.choice(n_clients, size, replace=False).tolist())
            for _ in range(n_rounds)]


def train(model, clients, test_inputs, test_targets, training, control, seed,
          method, checkpoints, target_mean=0., target_std=1., label=''):
    """Paired trainer for FedAvg/SB-CFO; optimizer state persists per client.

    Only one worker model resides on the accelerator. Client optimizer states
    are retained on CPU between visits. Both algorithms use the same initial
    model, client minibatch seeds, local-step budget and sample-count weights.
    """
    if method not in {'fedavg', 'sb_cfo'}:
        raise ValueError("unknown federated method")
    validate_schedule(training)
    rate = training.get('participation_rate', 1.)
    schedule = participation_schedule(len(clients), training['n_rounds'], rate, seed)
    if rate < 1 and method != 'fedavg':
        raise ValueError('Partial participation currently supports fedavg only')
    device = torch.device(training['device'])
    model = model.to(device)
    worker = copy.deepcopy(model)
    counts = count_matrix(clients, control['n_bins'])
    weights = counts.sum(1) / counts.sum()
    dimension = real_vector(model.parameters()).numel()
    model_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    vector_bytes = dimension * 4
    optim_states = [None for _ in clients]
    generators = [np.random.RandomState(int(seed) + 1009 * (k + 1))
                  for k in range(len(clients))]
    corrections = torch.zeros(len(clients), dimension)
    last_refresh = -1
    costs = dict(optimizer_examples=0, prototype_examples=0, diagnostic_examples=0,
                 prototype_refreshes=0, model_communication_bytes=0,
                 prototype_communication_bytes=0)
    started = time.perf_counter()
    prototype_seconds = 0.
    local_variance = float('nan')
    records = []
    rounds, batch_size = training['n_rounds'], training['batch_size']
    for t in range(rounds + 1):
        snapshot = None
        refresh = (method == 'sb_cfo' and t < rounds
                   and t % control['refresh_every'] == 0)
        if refresh:
            then = time.perf_counter()
            snapshot = bin_gradient_snapshot(
                model, clients, control['n_bins'], batch_size, device,
                control['samples_per_client_bin'], seed + 400003 + t)
            corrections = correction_vectors(counts, snapshot['prototypes'], control['strength'])
            prototype_seconds += time.perf_counter() - then
            costs['prototype_examples'] += snapshot['examples']
            costs['prototype_refreshes'] += 1
            # Logical FL payload: one mean per occupied client-bin upstream;
            # broadcast all B prototypes to each of K clients downstream.
            costs['prototype_communication_bytes'] += vector_bytes * (
                int(np.count_nonzero(counts)) + len(clients) * control['n_bins'])
            last_refresh = t
        if t in checkpoints:
            error = evaluate(model, test_inputs, test_targets, batch_size, device,
                             target_mean, target_std)
            if snapshot is None or control['samples_per_client_bin'] != 0:
                snapshot = bin_gradient_snapshot(
                    model, clients, control['n_bins'], batch_size, device)
                costs['diagnostic_examples'] += snapshot['examples']
            metrics = gradient_diagnostics(snapshot, corrections)
            records.append(dict(
                round=t, test_error=error, local_update_variance=local_variance,
                learning_rate=(round_learning_rate(training, t - 1) if t > 0 else float('nan')),
                prototype_age_rounds=t - last_refresh if last_refresh >= 0 else -1,
                elapsed_seconds=time.perf_counter() - started,
                prototype_seconds=prototype_seconds,
                **costs, **metrics))
            print("{} {} round={}/{} error={:.6f} R={:.3f} lr={}".format(
                label, method, t, rounds, error, metrics['correction_ratio'],
                'initial' if t == 0 else '{:.6g}'.format(records[-1]['learning_rate'])), flush=True)
        if t == rounds:
            break
        current_lr = round_learning_rate(training, t)
        broadcast = cpu_tree(model.state_dict())
        initial_vector = real_vector(model.parameters()).double()
        averaged = {}
        update_mean = torch.zeros(dimension, dtype=torch.float64)
        update_second = 0.
        selected = schedule[t]
        # Selected-client FedAvg: sum_{k in S} n_k*w_k / sum_{j in S} n_j.
        # For unequal sizes this ratio estimator is not exactly an unbiased
        # full-population update (it is not Horvitz--Thompson weighting).
        round_weights = weights.copy() if len(selected) == len(clients) else (
            weights / weights[selected].sum())
        for k in selected:
            client, rng = clients[k], generators[k]
            worker.load_state_dict(broadcast)
            worker.train()
            optimizer = make_optimizer(worker, training)
            if optim_states[k] is not None:
                optimizer.load_state_dict(optim_states[k])
            # load_state_dict restores the previous visit's learning rate too.
            # Apply the global-round schedule AFTER restoring moments/velocity.
            for group in optimizer.param_groups:
                group['lr'] = current_lr
            correction = corrections[k].to(device)
            for _ in range(training['local_steps']):
                indices = rng.choice(len(client.inputs),
                                     min(batch_size, len(client.inputs)), replace=False)
                x = torch.as_tensor(client.inputs[indices], dtype=torch.float32, device=device)
                y = torch.as_tensor(client.targets[indices], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = (worker(x) - y).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite local loss; check learning rate")
                loss.backward()
                if method == 'sb_cfo':
                    add_correction(worker, correction)
                optimizer.step()
                costs['optimizer_examples'] += len(indices)
            optim_states[k] = cpu_tree(optimizer.state_dict())
            state = cpu_tree(worker.state_dict())
            for name, value in state.items():
                if value.is_floating_point() or value.is_complex():
                    if name not in averaged:
                        averaged[name] = value * float(round_weights[k])
                    else:
                        averaged[name].add_(value, alpha=float(round_weights[k]))
                elif name not in averaged:
                    averaged[name] = value.clone()
            delta = real_vector(worker.parameters()).double() - initial_vector
            if not torch.isfinite(delta).all():
                raise FloatingPointError("non-finite local model; check learning rate")
            update_mean.add_(delta, alpha=float(round_weights[k]))
            update_second += float(round_weights[k]) * float(delta.square().sum())
        local_variance = max(0., update_second - float(update_mean.square().sum()))
        model.load_state_dict(averaged)
        costs['model_communication_bytes'] += 2 * len(selected) * model_bytes
        if (t + 1) % training['log_every'] == 0 and t + 1 not in checkpoints:
            print("{} {} round={}/{} lr={:.6g} elapsed={:.1f}s".format(
                label, method, t + 1, rounds, current_lr, time.perf_counter() - started), flush=True)
    return records, cpu_tree(model.state_dict())
