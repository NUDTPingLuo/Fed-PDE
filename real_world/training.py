"""PyTorch FedAvg training for public field-to-field PDE tasks."""
import copy

from util.schedules import round_learning_rate

import numpy as np
import torch

from .models import make_model


def resolve_device(requested):
    value = str(requested).lower()
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f"device={requested!r}, but CUDA is unavailable")
    return device


def _seed(seed, device):
    torch.manual_seed(int(seed))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _optimizer(model, cfg):
    name, lr = cfg['optimizer'].lower(), float(cfg['lr'])
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == 'sgd':
        return torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=float(cfg.get('momentum', 0.0)),
            nesterov=bool(cfg.get('nesterov', False)))
    raise ValueError("optimizer must be adam or sgd")


def _state(model):
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def _load(model, state):
    model.load_state_dict(state)


def _model(cfg, inputs, targets, seed, device):
    _seed(seed, device)
    return make_model(
        cfg['model'], inputs.shape[1:], targets.shape[1:]).to(device)


def _batch_indices(n_samples, batch_size, steps, rng):
    size = min(int(batch_size), int(n_samples))
    for _ in range(int(steps)):
        yield rng.choice(n_samples, size=size, replace=False)


def _step(model, optimizer, inputs, targets, indices, device):
    x = torch.as_tensor(inputs[indices], dtype=torch.float32, device=device)
    y = torch.as_tensor(targets[indices], dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.mean((model(x) - y) ** 2)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def fedavg_train(client_inputs, client_targets, cfg, seed,
                 checkpoint_rounds=None):
    device = resolve_device(cfg['device'])
    global_model = _model(
        cfg, client_inputs[0], client_targets[0], seed, device)
    clients = [copy.deepcopy(global_model) for _ in client_inputs]
    trainers = [_optimizer(model, cfg) for model in clients]
    generators = [
        np.random.RandomState(int(seed) + 1009 * (index + 1))
        for index in range(len(clients))]
    counts = np.asarray([len(value) for value in client_inputs], dtype=float)
    weights = counts / counts.sum()
    requested = (None if checkpoint_rounds is None else
                 {int(value) for value in checkpoint_rounds})
    history = {}
    if requested is not None and 0 in requested:
        history[0] = _state(global_model)
    for round_index in range(int(cfg['n_rounds'])):
        round_lr = round_learning_rate(cfg, round_index)
        for trainer in trainers:
            for group in trainer.param_groups:
                group['lr'] = round_lr
        broadcast = _state(global_model)
        for model, trainer, x, y, rng in zip(
                clients, trainers, client_inputs, client_targets, generators):
            _load(model, broadcast)
            model.train()
            for indices in _batch_indices(
                    len(x), cfg['batch_size'], cfg['local_steps'], rng):
                _step(model, trainer, x, y, indices, device)
        averaged = {}
        states = [_state(model) for model in clients]
        for key in states[0]:
            if states[0][key].is_floating_point() or states[0][key].is_complex():
                averaged[key] = sum(
                    float(weight) * state[key]
                    for weight, state in zip(weights, states))
            else:
                averaged[key] = states[0][key]
        _load(global_model, averaged)
        completed_round = round_index + 1
        if requested is not None and completed_round in requested:
            history[completed_round] = _state(global_model)
    final = _state(global_model)
    return ((final, history) if requested is not None else final)


def relative_error(state, inputs, targets, cfg, target_mean, target_std, seed=0):
    device = resolve_device(cfg['device'])
    model = _model(cfg, inputs, targets, seed, device)
    _load(model, state)
    mean = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(target_std, dtype=torch.float32, device=device)
    numerator = torch.zeros((), device=device)
    denominator = torch.zeros((), device=device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), int(cfg['batch_size'])):
            end = min(start + int(cfg['batch_size']), len(inputs))
            x = torch.as_tensor(inputs[start:end], dtype=torch.float32, device=device)
            y = torch.as_tensor(targets[start:end], dtype=torch.float32, device=device)
            prediction = model(x) * std + mean
            truth = y * std + mean
            numerator += torch.sum((prediction - truth) ** 2)
            denominator += torch.sum(truth ** 2)
    return float(torch.sqrt(numerator / denominator.clamp_min(1e-24)).cpu())


def gradient_dissimilarity(client_inputs, client_targets, cfg, seed):
    device = resolve_device(cfg['device'])
    model = _model(cfg, client_inputs[0], client_targets[0], seed, device)
    gradients = []
    for x, y in zip(client_inputs, client_targets):
        model.zero_grad(set_to_none=True)
        count = len(x)
        for start in range(0, count, int(cfg['batch_size'])):
            end = min(start + int(cfg['batch_size']), count)
            tx = torch.as_tensor(x[start:end], dtype=torch.float32, device=device)
            ty = torch.as_tensor(y[start:end], dtype=torch.float32, device=device)
            loss = torch.mean((model(tx) - ty) ** 2) * ((end - start) / count)
            loss.backward()
        gradients.append(torch.cat([
            parameter.grad.reshape(-1) for parameter in model.parameters()]))
    gradients = torch.stack(gradients)
    weights = torch.as_tensor(
        [len(value) for value in client_inputs],
        dtype=torch.float32, device=device)
    weights /= weights.sum()
    mean = torch.sum(weights[:, None] * gradients, dim=0)
    numerator = torch.sum(
        weights * torch.sum(torch.abs(gradients - mean) ** 2, dim=1))
    denominator = torch.sum(
        weights * torch.sum(torch.abs(gradients) ** 2, dim=1))
    return float((numerator / denominator.clamp_min(1e-12)).detach().cpu())


def state_divergence(first, second):
    keys = [key for key in first
            if first[key].is_floating_point() or first[key].is_complex()]
    numerator = sum(
        torch.sum(torch.abs(first[key] - second[key]) ** 2) for key in keys)
    denominator = sum(torch.sum(torch.abs(first[key]) ** 2) for key in keys)
    return float(torch.sqrt(numerator / denominator.clamp_min(1e-24)))
