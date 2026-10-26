"""Chebyshev-input operator datasets for federated PDE experiments.

The implementation is independent and lightweight: inputs are functions
sampled at fixed sensors and targets are numerical solution fields.  It
supports the antiderivative, diffusion-reaction, and viscous Burgers
operators described in Sections 6.2--6.4 of Zhang et al. (2025).
"""
import numpy as np


class OperatorTask:
    VALID_TASKS = {'antiderivative', 'diffusion_reaction', 'burgers'}

    def __init__(self, name, n_sensors, output_grid, condition_scale,
                 n_modes=10, diffusion=0.01, reaction=0.01, viscosity=0.1):
        if name not in self.VALID_TASKS:
            raise ValueError(f"unknown operator task {name}; choose from {sorted(self.VALID_TASKS)}")
        if n_sensors < n_modes or output_grid < 4:
            raise ValueError("n_sensors must cover n_modes and output_grid must be >= 4")
        self.name = name
        self.n_sensors = int(n_sensors)
        self.output_grid = int(output_grid)
        self.condition_scale = float(condition_scale)
        self.n_modes = int(n_modes)
        self.diffusion = float(diffusion)
        self.reaction = float(reaction)
        self.viscosity = float(viscosity)
        self.sensor_grid = np.linspace(0.0, 1.0, self.n_sensors)
        self.chebyshev_basis = np.polynomial.chebyshev.chebvander(
            2.0 * self.sensor_grid - 1.0, self.n_modes - 1)

        if name == 'antiderivative':
            self.space_grid = np.linspace(0.0, 1.0, self.output_grid)
            self.output_coordinates = self.space_grid[:, None]
        else:
            self.space_grid = np.linspace(0.0, 1.0, self.output_grid)
            self.time_grid = np.linspace(0.0, 1.0, self.output_grid)
            self.output_coordinates = np.array(
                [(x, t) for x in self.space_grid for t in self.time_grid],
                dtype=float)

    @property
    def output_size(self):
        return len(self.output_coordinates)

    def _coefficients_to_sensors(self, coefficients):
        return np.asarray(coefficients, dtype=float) @ self.chebyshev_basis.T

    def sample_conditions(self, n_samples, seed):
        """Sample from the complete 10-term Chebyshev function space."""
        rng = np.random.RandomState(seed)
        coefficients = rng.uniform(
            -self.condition_scale, self.condition_scale,
            size=(int(n_samples), self.n_modes))
        return self._coefficients_to_sensors(coefficients)

    def solve(self, condition):
        return self.solve_batch(np.asarray(condition, dtype=float)[None, :])[0]

    def solve_batch(self, conditions):
        conditions = np.asarray(conditions, dtype=float)
        if conditions.ndim != 2 or conditions.shape[1] != self.n_sensors:
            raise ValueError("conditions must have shape (samples, n_sensors)")
        source = np.stack([
            np.interp(self.space_grid, self.sensor_grid, row)
            for row in conditions
        ])
        if self.name == 'antiderivative':
            dx = np.diff(self.space_grid)
            increments = 0.5 * (source[:, 1:] + source[:, :-1]) * dx
            return np.column_stack([np.zeros(len(source)), np.cumsum(increments, axis=1)])
        if self.name == 'diffusion_reaction':
            return self._solve_diffusion_reaction(source)
        return self._solve_burgers(source)

    def _solve_diffusion_reaction(self, source):
        n_samples, nx = source.shape
        dx = 1.0 / (nx - 1)
        output_dt = 1.0 / (self.output_grid - 1)
        stable_dt = 0.4 * dx ** 2 / max(self.diffusion, 1e-12)
        n_substeps = max(1, int(np.ceil(output_dt / stable_dt)))
        dt = output_dt / n_substeps
        state = np.zeros((n_samples, nx))
        field = np.zeros((n_samples, nx, self.output_grid))
        for time_index in range(1, self.output_grid):
            for _ in range(n_substeps):
                laplacian = (
                    state[:, 2:] - 2.0 * state[:, 1:-1] + state[:, :-2]
                ) / dx ** 2
                state[:, 1:-1] += dt * (
                    self.diffusion * laplacian
                    + self.reaction * state[:, 1:-1] ** 2
                    + source[:, 1:-1]
                )
                state[:, (0, -1)] = 0.0
            field[:, :, time_index] = state
        return field.reshape(n_samples, -1)

    def _solve_burgers(self, initial):
        n_samples, nx = initial.shape
        dx = 1.0 / (nx - 1)
        output_dt = 1.0 / (self.output_grid - 1)
        state = initial[:, :-1].copy()
        field = np.zeros((n_samples, nx, self.output_grid))
        field[:, :, 0] = np.column_stack([state, state[:, 0]])
        for time_index in range(1, self.output_grid):
            max_speed = max(float(np.max(np.abs(state))), 1e-8)
            stable_dt = 0.35 / (
                max_speed / dx + 2.0 * self.viscosity / dx ** 2)
            n_substeps = max(1, int(np.ceil(output_dt / stable_dt)))
            dt = output_dt / n_substeps
            for _ in range(n_substeps):
                left = np.roll(state, 1, axis=1)
                right = np.roll(state, -1, axis=1)
                flux = 0.5 * state ** 2
                interface_speed = np.maximum(np.abs(state), np.abs(right))
                flux_right = (
                    0.5 * (flux + 0.5 * right ** 2)
                    - 0.5 * interface_speed * (right - state))
                flux_left = np.roll(flux_right, 1, axis=1)
                convection = (flux_right - flux_left) / dx
                diffusion = (right - 2.0 * state + left) / dx ** 2
                state = state + dt * (-convection + self.viscosity * diffusion)
                if not np.all(np.isfinite(state)):
                    raise FloatingPointError(
                        "Burgers solver became non-finite; reduce condition_scale")
            field[:, :, time_index] = np.column_stack([state, state[:, 0]])
        return field.reshape(n_samples, -1)
