"""Cartesian-product DeepONet and standard sample-weighted FedAvg."""
import numpy as np


class CartesianDeepONet:
    """NumPy implementation of the branch/trunk model used by FedDeepONet.

    A branch network encodes an input function sampled at sensors.  A trunk
    network encodes output coordinates.  Their latent inner product predicts
    the complete solution field on the shared coordinate grid.
    """

    def __init__(self, branch_dim, coordinates, width, depth=2,
                 activation='relu', seed=None):
        if depth < 2:
            raise ValueError("DeepONet depth must include at least two linear layers")
        self.coordinates = np.asarray(coordinates, dtype=float)
        if self.coordinates.ndim != 2:
            raise ValueError("coordinates must have shape (n_points, coordinate_dim)")
        self.width = int(width)
        self.depth = int(depth)
        self.activation = activation
        rng = np.random.RandomState(seed)
        self.params = {'output_bias': np.zeros(1)}
        self._init_subnetwork('branch', int(branch_dim), rng)
        self._init_subnetwork('trunk', self.coordinates.shape[1], rng)
        self._adam_state = None
        self._sgd_state = None

    def _init_subnetwork(self, prefix, input_dim, rng):
        dimensions = [input_dim] + [self.width] * self.depth
        for layer, (fan_in, fan_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            scale = np.sqrt(2.0 / max(fan_in + fan_out, 1))
            self.params[f'{prefix}_W{layer}'] = rng.randn(fan_in, fan_out) * scale
            self.params[f'{prefix}_b{layer}'] = np.zeros(fan_out)

    def _activate(self, value):
        if self.activation == 'relu':
            return np.maximum(value, 0.0)
        if self.activation == 'tanh':
            return np.tanh(value)
        if self.activation == 'sine':
            return np.sin(value)
        raise ValueError(f"unknown activation: {self.activation}")

    def _activation_gradient(self, preactivation):
        if self.activation == 'relu':
            return (preactivation > 0.0).astype(float)
        if self.activation == 'tanh':
            value = np.tanh(preactivation)
            return 1.0 - value ** 2
        if self.activation == 'sine':
            return np.cos(preactivation)
        raise ValueError(f"unknown activation: {self.activation}")

    def _subnetwork_forward(self, inputs, prefix):
        activations = [np.asarray(inputs, dtype=float)]
        preactivations = []
        value = activations[0]
        for layer in range(self.depth):
            preactivation = (
                value @ self.params[f'{prefix}_W{layer}']
                + self.params[f'{prefix}_b{layer}']
            )
            preactivations.append(preactivation)
            value = (self._activate(preactivation)
                     if layer < self.depth - 1 else preactivation)
            activations.append(value)
        return value, (activations, preactivations)

    def _subnetwork_backward(self, output_gradient, cache, prefix):
        activations, preactivations = cache
        gradient = output_gradient
        result = {}
        for layer in reversed(range(self.depth)):
            if layer < self.depth - 1:
                gradient = gradient * self._activation_gradient(preactivations[layer])
            result[f'{prefix}_W{layer}'] = activations[layer].T @ gradient
            result[f'{prefix}_b{layer}'] = gradient.sum(axis=0)
            gradient = gradient @ self.params[f'{prefix}_W{layer}'].T
        return result

    def forward(self, branch_inputs):
        branch, branch_cache = self._subnetwork_forward(branch_inputs, 'branch')
        trunk, trunk_cache = self._subnetwork_forward(self.coordinates, 'trunk')
        prediction = branch @ trunk.T + self.params['output_bias'][0]
        return prediction, (branch, trunk, branch_cache, trunk_cache)

    def _grads(self, branch_inputs, targets):
        prediction, cache = self.forward(branch_inputs)
        branch, trunk, branch_cache, trunk_cache = cache
        difference = prediction - np.asarray(targets, dtype=float)
        loss = float(np.mean(difference ** 2))
        output_gradient = 2.0 * difference / difference.size
        gradients = {'output_bias': np.array([output_gradient.sum()])}
        gradients.update(self._subnetwork_backward(
            output_gradient @ trunk, branch_cache, 'branch'))
        gradients.update(self._subnetwork_backward(
            output_gradient.T @ branch, trunk_cache, 'trunk'))
        return loss, gradients

    def gradient_vector(self, inputs, targets):
        _, gradients = self._grads(inputs, targets)
        return np.concatenate([gradients[key].ravel() for key in sorted(gradients)])

    def step(self, inputs, targets, lr, optimizer='adam', momentum=0.0,
             nesterov=False):
        if nesterov and (optimizer != 'sgd' or not 0 < momentum < 1):
            raise ValueError('Nesterov requires SGD with momentum in (0,1)')
        loss, gradients = self._grads(inputs, targets)
        keys = tuple(self.params)
        if optimizer == 'sgd':
            if self._sgd_state is None:
                self._sgd_state = {
                    key: np.zeros_like(self.params[key]) for key in keys}
            for key in keys:
                self._sgd_state[key] = (
                    float(momentum) * self._sgd_state[key] + gradients[key])
                direction = (gradients[key] + float(momentum) * self._sgd_state[key]
                             if nesterov else self._sgd_state[key])
                self.params[key] -= lr * direction
        elif optimizer == 'adam':
            if self._adam_state is None:
                self._adam_state = dict(
                    m={key: np.zeros_like(self.params[key]) for key in keys},
                    v={key: np.zeros_like(self.params[key]) for key in keys},
                    t=0,
                )
            state = self._adam_state
            state['t'] += 1
            for key in keys:
                state['m'][key] = 0.9 * state['m'][key] + 0.1 * gradients[key]
                state['v'][key] = 0.999 * state['v'][key] + 0.001 * gradients[key] ** 2
                mhat = state['m'][key] / (1.0 - 0.9 ** state['t'])
                vhat = state['v'][key] / (1.0 - 0.999 ** state['t'])
                self.params[key] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        else:
            raise ValueError(f"unknown optimizer: {optimizer}")
        return loss

    def clone_params(self):
        return {key: value.copy() for key, value in self.params.items()}

    def load_params(self, params):
        if set(params) != set(self.params):
            raise ValueError("parameter structure does not match DeepONet architecture")
        self.params = {key: np.asarray(value).copy() for key, value in params.items()}


def average_params(param_list, weights):
    """Standard FedAvg: client weight equals its local sample count."""
    if not param_list:
        raise ValueError("param_list must not be empty")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(param_list),) or np.any(weights <= 0):
        raise ValueError("weights must contain one positive sample count per client")
    weights /= weights.sum()
    return {
        key: np.tensordot(weights, np.stack([p[key] for p in param_list]), axes=(0, 0))
        for key in param_list[0]
    }
