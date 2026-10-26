"""Federated training for Cartesian-product DeepONets."""
from util.schedules import round_learning_rate

import numpy as np
from util.model import CartesianDeepONet, average_params


def _uses_torch(model_spec):
    backend = model_spec.get('backend', 'numpy')
    if backend == 'auto':
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    return backend == 'torch'


def backend_description(model_spec):
    """Return the resolved training backend/device for startup logging."""
    if not _uses_torch(model_spec):
        return 'NumPy / CPU'
    from util.torch_training import backend_description as describe
    return describe(model_spec)


def _make_model(inputs, model_spec, seed=None):
    return CartesianDeepONet(
        inputs.shape[1], model_spec['coordinates'], model_spec['width'],
        depth=model_spec['depth'], activation=model_spec['activation'], seed=seed)


def _step_batches(n_samples, batch_size, steps, rng):
    """Draw one shuffled mini-batch for each fixed local optimizer step."""
    size = n_samples if batch_size is None else min(int(batch_size), n_samples)
    for _ in range(steps):
        yield rng.choice(n_samples, size=size, replace=False)


def fedavg_train(client_inputs, client_targets, n_rounds, local_steps, lr,
                 optimizer, model_spec, init_seed, verbose=False,
                 checkpoint_rounds=None):
    if _uses_torch(model_spec):
        from util.torch_training import fedavg_train as torch_fedavg_train
        return torch_fedavg_train(
            client_inputs, client_targets, n_rounds, local_steps, lr,
            optimizer, model_spec, init_seed, verbose, checkpoint_rounds)
    model = _make_model(client_inputs[0], model_spec, init_seed)
    global_params = model.clone_params()
    sample_counts = [len(x) for x in client_inputs]
    # The supplied FedSciML implementation keeps each compiled client model
    # alive across communication rounds.  Loading broadcast weights therefore
    # does not discard local Adam moments.
    client_models = [
        _make_model(inputs, model_spec, init_seed)
        for inputs in client_inputs
    ]
    client_rngs = [
        np.random.RandomState(int(init_seed) + 1009 * (index + 1))
        for index in range(len(client_inputs))
    ]
    requested = (None if checkpoint_rounds is None else
                 {int(value) for value in checkpoint_rounds})
    history = {}
    if requested is not None and 0 in requested:
        history[0] = model.clone_params()
    schedule = dict(model_spec, lr=lr, n_rounds=n_rounds)
    for round_index in range(n_rounds):
        round_lr = round_learning_rate(schedule, round_index)
        local_params = []
        for client_model, inputs, targets, rng in zip(
                client_models, client_inputs, client_targets, client_rngs):
            client_model.load_params(global_params)
            for batch in _step_batches(
                    len(inputs), model_spec.get('batch_size'), local_steps, rng):
                client_model.step(
                    inputs[batch], targets[batch], round_lr, optimizer,
                    model_spec.get('momentum', 0.0),
                    model_spec.get('nesterov', False))
            local_params.append(client_model.clone_params())
        global_params = average_params(local_params, sample_counts)
        completed_round = round_index + 1
        if requested is not None and completed_round in requested:
            history[completed_round] = {
                key: value.copy() for key, value in global_params.items()}
        if verbose:
            print(f"    [FedAvg] round {round_index + 1}/{n_rounds}")
    return ((global_params, history) if requested is not None else global_params)


def eval_params(params, inputs, targets, model_spec):
    if _uses_torch(model_spec):
        from util.torch_training import eval_params as torch_eval_params
        return torch_eval_params(params, inputs, targets, model_spec)
    model = _make_model(inputs, model_spec)
    model.load_params(params)
    prediction, _ = model.forward(inputs)
    denominator = max(float(np.linalg.norm(targets)), 1e-12)
    return float(np.linalg.norm(prediction - targets) / denominator)


def gradient_dissimilarity(client_inputs, client_targets, model_spec, init_seed):
    """Sample-weighted normalized gradient variance at initialization."""
    if _uses_torch(model_spec):
        from util.torch_training import gradient_dissimilarity as torch_gradient
        return torch_gradient(
            client_inputs, client_targets, model_spec, init_seed)
    model = _make_model(client_inputs[0], model_spec, init_seed)
    gradients = np.stack([
        model.gradient_vector(x, y)
        for x, y in zip(client_inputs, client_targets)
    ])
    weights = np.asarray([len(x) for x in client_inputs], dtype=float)
    weights /= weights.sum()
    mean_gradient = np.sum(weights[:, None] * gradients, axis=0)
    numerator = np.sum(weights * np.sum((gradients - mean_gradient) ** 2, axis=1))
    denominator = np.sum(weights * np.sum(gradients ** 2, axis=1))
    return float(numerator / max(denominator, 1e-12))


def relative_parameter_divergence(reference_params, candidate_params):
    """Relative parameter distance from the same-seed IID FedAvg model."""
    numerator = np.sqrt(sum(
        np.sum((candidate_params[key] - reference_params[key]) ** 2)
        for key in reference_params
    ))
    denominator = np.sqrt(sum(
        np.sum(value ** 2) for value in reference_params.values()))
    return float(numerator / max(denominator, 1e-12))
