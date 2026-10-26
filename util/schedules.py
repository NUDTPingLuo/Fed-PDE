"""Round-indexed schedules shared by all paired training trajectories."""
import math


def checkpoint_learning_rate(training, completed_round):
    """Rate of the just-completed round; no update has occurred at round zero."""
    return (None if completed_round == 0 else
            round_learning_rate(training, completed_round - 1))


def checkpoint_learning_rate(training, completed_round):
    """Rate of the just-completed round; no update has occurred at round zero."""
    return (None if completed_round == 0 else
            round_learning_rate(training, completed_round - 1))


def validate_schedule(training):
    name = training.get('lr_schedule', 'cosine')
    if name != 'cosine':
        raise ValueError('lr_schedule must be cosine; legacy fixed-rate runs are retired')
    lr = training['lr']
    floor = training.get('min_lr', 0.0001)
    rounds = training['n_rounds']
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('learning rate must be finite and positive')
    if not math.isfinite(floor) or floor < 0:
        raise ValueError('min_lr must be finite and nonnegative')
    if floor > lr:
        raise ValueError('min_lr must not exceed lr for cosine decay')
    if int(rounds) != rounds or rounds < 1:
        raise ValueError('n_rounds must be a positive integer')


def round_learning_rate(training, round_index):
    """Zero-based training round; first uses lr, last uses min_lr.

    All local steps and participating clients share this value. With only one
    training round, use the initial lr. Direct trainer calls use the same
    cosine schedule and default floor as the command-line entry point.
    """
    validate_schedule(training)
    rounds = training['n_rounds']
    if int(round_index) != round_index or not 0 <= round_index < rounds:
        raise ValueError('training round index is out of range')
    lr = float(training['lr'])
    if rounds == 1:
        return lr
    floor = float(training.get('min_lr', 0.0001))
    return floor + 0.5 * (lr - floor) * (1 + math.cos(math.pi * round_index / (rounds - 1)))
