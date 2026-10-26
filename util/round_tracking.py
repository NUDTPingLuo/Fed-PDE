"""Shared helpers for communication-round evaluation checkpoints."""


DEFAULT_ROUND_CHECKPOINTS = (0, 1, 5, 10, 20, 50, 100, 200, 500, 1000)


def resolve_round_checkpoints(n_rounds, requested=None):
    """Return sorted, unique checkpoints clipped to the training horizon."""
    n_rounds = int(n_rounds)
    values = DEFAULT_ROUND_CHECKPOINTS if requested is None else requested
    checkpoints = {
        int(value) for value in values
        if 0 <= int(value) <= n_rounds
    }
    checkpoints.update((0, n_rounds))
    return sorted(checkpoints)
