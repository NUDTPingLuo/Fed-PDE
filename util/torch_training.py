"""PyTorch/CUDA backend for Cartesian DeepONet and sample-weighted FedAvg."""
import warnings

from util.schedules import round_learning_rate

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise RuntimeError(
        "training.backend is 'torch', but PyTorch is not installed. "
        "Install a CUDA-enabled PyTorch build in the active environment."
    ) from exc


class Sine(nn.Module):
    def forward(self, inputs):
        return torch.sin(inputs)


def _activation(name):
    if name == 'relu':
        return nn.ReLU
    if name == 'tanh':
        return nn.Tanh
    if name == 'sine':
        return Sine
    raise ValueError(f"unknown activation: {name}")


class TorchCartesianDeepONet(nn.Module):
    """Mathematically equivalent branch/trunk model implemented on torch."""

    def __init__(self, branch_dim, model_spec, seed):
        super().__init__()
        torch.manual_seed(int(seed or 0))
        width = int(model_spec['width'])
        depth = int(model_spec['depth'])
        if depth < 2:
            raise ValueError("DeepONet depth must include at least two linear layers")
        coordinates = torch.as_tensor(
            model_spec['coordinates'], dtype=torch.float32)
        self.register_buffer('coordinates', coordinates, persistent=False)
        self.branch = self._subnetwork(
            int(branch_dim), width, depth, model_spec['activation'])
        self.trunk = self._subnetwork(
            coordinates.shape[1], width, depth, model_spec['activation'])
        self.output_bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _subnetwork(input_dim, width, depth, activation):
        layers = []
        for layer in range(depth):
            linear = nn.Linear(input_dim if layer == 0 else width, width)
            nn.init.normal_(
                linear.weight,
                std=np.sqrt(2.0 / max(linear.in_features + linear.out_features, 1)))
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if layer < depth - 1:
                layers.append(_activation(activation)())
        return nn.Sequential(*layers)

    def forward(self, branch_inputs):
        return self.branch(branch_inputs) @ self.trunk(self.coordinates).T + self.output_bias


def _device(model_spec):
    requested = str(model_spec.get('device', 'auto')).lower()
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        warnings.warn(
            "CUDA is unavailable; PyTorch training is falling back to CPU. "
            "Check the CUDA-enabled PyTorch installation and NVIDIA driver.",
            RuntimeWarning, stacklevel=2)
        return torch.device('cpu')
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            f"training.device={requested!r}, but torch.cuda.is_available() is False")
    return device


def backend_description(model_spec):
    device = _device(model_spec)
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"PyTorch {torch.__version__} / CUDA / {torch.cuda.get_device_name(index)}"
    return f"PyTorch {torch.__version__} / CPU fallback"


def _configure_reproducibility(seed, device):
    torch.manual_seed(int(seed))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _tensor(values, device):
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _make_model(inputs, model_spec, seed, device):
    return TorchCartesianDeepONet(inputs.shape[1], model_spec, seed).to(device)


def _make_optimizer(model, name, lr, momentum=0.0, nesterov=False):
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == 'sgd':
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=float(momentum), nesterov=nesterov)
    raise ValueError(f"unknown optimizer: {name}")


def _batch_size(model_spec, n_samples):
    configured = model_spec.get('batch_size')
    return n_samples if configured is None else min(int(configured), n_samples)


def _loss_backward(model, inputs, targets, model_spec):
    """Accumulate an exact full-dataset mean gradient in sample batches."""
    n_samples = len(inputs)
    batch_size = _batch_size(model_spec, n_samples)
    use_amp = bool(model_spec.get('mixed_precision', False)) and inputs.device.type == 'cuda'
    total = 0.0
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        with torch.autocast(device_type=inputs.device.type, enabled=use_amp):
            difference = model(inputs[start:end]) - targets[start:end]
            batch_loss = torch.mean(difference.square())
            weighted_loss = batch_loss * ((end - start) / n_samples)
        weighted_loss.backward()
        total += float(weighted_loss.detach())
    return total


def _cpu_state(model):
    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in model.state_dict().items()
    }


def _load_cpu_state(model, state):
    device = next(model.parameters()).device
    model.load_state_dict({key: _tensor(value, device) for key, value in state.items()})


def _step_batches(n_samples, batch_size, steps, rng):
    """Draw one shuffled mini-batch for each fixed local optimizer step."""
    for _ in range(steps):
        yield rng.choice(n_samples, size=batch_size, replace=False)


def fedavg_train(client_inputs, client_targets, n_rounds, local_steps, lr,
                 optimizer, model_spec, init_seed, verbose=False,
                 checkpoint_rounds=None):
    device = _device(model_spec)
    _configure_reproducibility(init_seed, device)
    datasets = [
        (_tensor(inputs, device), _tensor(targets, device))
        for inputs, targets in zip(client_inputs, client_targets)
    ]
    global_model = _make_model(client_inputs[0], model_spec, init_seed, device)
    client_models = [
        _make_model(inputs, model_spec, init_seed, device)
        for inputs in client_inputs
    ]
    trainers = [
        _make_optimizer(
            model, optimizer, lr, model_spec.get('momentum', 0.0),
            model_spec.get('nesterov', False))
        for model in client_models]
    client_rngs = [
        np.random.RandomState(int(init_seed) + 1009 * (index + 1))
        for index in range(len(client_inputs))
    ]
    sample_counts = torch.as_tensor(
        [len(inputs) for inputs in client_inputs], dtype=torch.float32, device=device)
    weights = sample_counts / sample_counts.sum()
    requested = (None if checkpoint_rounds is None else
                 {int(value) for value in checkpoint_rounds})
    history = {}
    if requested is not None and 0 in requested:
        history[0] = _cpu_state(global_model)

    schedule = dict(model_spec, lr=lr, n_rounds=n_rounds)
    for round_index in range(n_rounds):
        round_lr = round_learning_rate(schedule, round_index)
        for trainer in trainers:
            for group in trainer.param_groups:
                group['lr'] = round_lr
        broadcast = global_model.state_dict()
        for model, trainer, (inputs, targets), rng in zip(
                client_models, trainers, datasets, client_rngs):
            model.load_state_dict(broadcast)
            model.train()
            batch_size = _batch_size(model_spec, len(inputs))
            for indices in _step_batches(
                    len(inputs), batch_size, local_steps, rng):
                batch = torch.as_tensor(
                    indices, dtype=torch.long, device=device)
                trainer.zero_grad(set_to_none=True)
                _loss_backward(
                    model, inputs[batch], targets[batch], model_spec)
                trainer.step()
        with torch.no_grad():
            averaged = {}
            states = [model.state_dict() for model in client_models]
            for key in states[0]:
                averaged[key] = sum(
                    weights[index] * state[key]
                    for index, state in enumerate(states))
            global_model.load_state_dict(averaged)
        completed_round = round_index + 1
        if requested is not None and completed_round in requested:
            history[completed_round] = _cpu_state(global_model)
        if verbose:
            print(f"    [FedAvg/{device}] round {round_index + 1}/{n_rounds}")
    final = _cpu_state(global_model)
    return ((final, history) if requested is not None else final)


def eval_params(params, inputs, targets, model_spec):
    device = _device(model_spec)
    model = _make_model(inputs, model_spec, 0, device)
    _load_cpu_state(model, params)
    x = _tensor(inputs, device)
    y = _tensor(targets, device)
    batch_size = _batch_size(model_spec, len(x))
    numerator = torch.zeros((), device=device)
    denominator = torch.zeros((), device=device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            difference = model(x[start:end]) - y[start:end]
            numerator += difference.square().sum()
            denominator += y[start:end].square().sum()
    return float(torch.sqrt(numerator / denominator.clamp_min(1e-24)).cpu())


def gradient_dissimilarity(client_inputs, client_targets, model_spec, init_seed):
    device = _device(model_spec)
    _configure_reproducibility(init_seed, device)
    model = _make_model(client_inputs[0], model_spec, init_seed, device)
    gradients = []
    for inputs, targets in zip(client_inputs, client_targets):
        model.zero_grad(set_to_none=True)
        _loss_backward(
            model, _tensor(inputs, device), _tensor(targets, device), model_spec)
        gradients.append(torch.cat([
            parameter.grad.detach().reshape(-1)
            for parameter in model.parameters()
        ]))
    gradients = torch.stack(gradients)
    weights = torch.as_tensor(
        [len(inputs) for inputs in client_inputs], dtype=torch.float32, device=device)
    weights /= weights.sum()
    mean_gradient = torch.sum(weights[:, None] * gradients, dim=0)
    numerator = torch.sum(weights * torch.sum((gradients - mean_gradient) ** 2, dim=1))
    denominator = torch.sum(weights * torch.sum(gradients ** 2, dim=1))
    return float((numerator / denominator.clamp_min(1e-12)).cpu())
