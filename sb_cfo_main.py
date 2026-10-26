"""Paired FedAvg/SB-CFO experiments on low-viscosity Burgers and Darcy.

The default configuration uses Adam (0.001 to 0.0001), 5 local steps and
1000 rounds, comparing IID against alpha=0.01 with round-wise cosine decay.
Select --optimizer sgd for
the Nesterov SGD ablation (cosine 0.01 to 0.001, momentum 0.9).
"""
import argparse
import copy
from pathlib import Path

import yaml

from util.round_tracking import resolve_round_checkpoints
from util.schedules import validate_schedule

ROOT = Path(__file__).resolve().parent


def build_plan(args):
    path = Path(args.config).resolve()
    with path.open(encoding='utf-8') as stream:
        settings = yaml.safe_load(stream)
    tasks = ('burgers', 'darcy') if args.task == 'all' else (args.task,)
    plans = []
    for task in tasks:
        values = (args.viscosities or settings['viscosities']) if task == 'burgers' else [None]
        for viscosity in values:
            cfg = copy.deepcopy(settings)
            cfg['task'], cfg['viscosity'] = task, viscosity
            cfg['base_config'] = str((ROOT / cfg['base_configs'][task]).resolve())
            training = cfg['training']
            training.setdefault('lr_schedule', 'cosine')
            training.setdefault('min_lr', 0.0001)
            if (args.optimizer or training['optimizer']) == 'sgd':
                training.update(lr=0.01, min_lr=0.001, lr_schedule='cosine',
                                momentum=0.9, nesterov=True)
            for name in ('optimizer', 'lr', 'momentum', 'device', 'local_steps', 'log_every'):
                value = getattr(args, name)
                if value is not None:
                    training[name] = value
            for name in ('lr_schedule', 'min_lr'):
                value = getattr(args, name, None)
                if value is not None:
                    training[name] = value
            if getattr(args, 'nesterov', None) is not None:
                training['nesterov'] = args.nesterov
            if getattr(args, 'methods', None) is not None:
                cfg['methods'] = args.methods
            cfg.setdefault('methods', ['fedavg', 'sb_cfo'])
            if args.optimizer is not None and args.lr is None:
                training['lr'] = 0.001 if args.optimizer == 'adam' else 0.01
            if args.rounds is not None:
                training['n_rounds'] = args.rounds
            if getattr(args, 'participation_rate', None) is not None:
                training['participation_rate'] = args.participation_rate
            if args.seeds is not None:
                cfg['seeds'] = args.seeds
            if args.alphas is not None:
                cfg['alpha_values'] = args.alphas
            for name in ('strength', 'refresh_every', 'samples_per_client_bin'):
                value = getattr(args, name)
                if value is not None:
                    cfg['control'][name] = value
            cfg['smoke'] = bool(args.smoke)
            if args.smoke:
                cfg.update(seeds=[0], alpha_values=[0.1], n_clients=2,
                           n_bins=2, min_client_samples=2)
                training.update(n_rounds=2, local_steps=2, log_every=1)
            cfg['round_checkpoints'] = resolve_round_checkpoints(
                training['n_rounds'], cfg['round_checkpoints'])
            validate(cfg)
            output = Path(args.output_dir or cfg['output_dir'])
            if not output.is_absolute():
                output = ROOT / output
            if args.smoke:
                output /= 'smoke'
            output /= task
            if viscosity is not None:
                output /= 'nu' + format(viscosity, 'g').replace('.', 'p')
            optimizer_dir = training['optimizer']
            if optimizer_dir == 'sgd':
                tag = 'nesterov' if training.get('nesterov', False) else 'momentum'
                optimizer_dir = (f"sgd_{tag}_lr{training['lr']:g}"
                                 f"_min{training['min_lr']:g}_m{training['momentum']:g}").replace('.', 'p')
            cfg['output_dir'] = str((output / optimizer_dir).resolve())
            plans.append(cfg)
    return plans


def validate(cfg):
    import math
    training, control = cfg['training'], cfg['control']
    validate_schedule(training)
    methods = cfg.get('methods', ['fedavg', 'sb_cfo'])
    if (not methods or len(set(methods)) != len(methods) or 'fedavg' not in methods
            or set(methods) - {'fedavg', 'sb_cfo'}):
        raise ValueError('methods must be unique and include fedavg; supported: fedavg, sb_cfo')
    for value in (cfg['n_clients'], cfg['n_bins'], cfg['min_client_samples'],
                  training['n_rounds'], training['local_steps'], training['log_every'],
                  control['refresh_every']):
        if int(value) != value or value <= 0:
            raise ValueError("counts, rounds and refresh intervals must be positive integers")
    if cfg['n_bins'] < 2 or cfg['n_clients'] < 2:
        raise ValueError("at least two clients and bins are required")
    if not math.isfinite(training['lr']) or training['lr'] <= 0:
        raise ValueError("learning rate must be finite and positive")
    rate = training.get('participation_rate', 1.0)
    if not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("participation_rate must be finite and in (0,1]")
    if rate < 1 and list(methods) != ['fedavg']:
        raise ValueError('Partial participation currently supports fedavg only')
    if training['optimizer'] not in ('sgd', 'adam'):
        raise ValueError("optimizer must be sgd or adam")
    if not 0 <= training['momentum'] < 1:
        raise ValueError("momentum must be in [0,1)")
    if training.get('nesterov', False) and (
            training['optimizer'] != 'sgd' or training['momentum'] <= 0):
        raise ValueError('Nesterov requires SGD with positive momentum')
    if training['optimizer'] == 'adam' and training['momentum'] != 0:
        raise ValueError("--momentum applies only to SGD; use 0 for Adam")
    if (control['samples_per_client_bin'] < 0
            or int(control['samples_per_client_bin']) != control['samples_per_client_bin']
            or not math.isfinite(control['strength']) or control['strength'] < 0):
        raise ValueError("sampling cap must be a nonnegative integer; strength must be finite >= 0")
    for name in ('seeds', 'alpha_values'):
        if not cfg[name] or len(set(cfg[name])) != len(cfg[name]):
            raise ValueError(name + " must be nonempty and unique")
    if any(not math.isfinite(a) or a <= 0 for a in cfg['alpha_values']):
        raise ValueError("alphas must be finite and positive")
    if any(int(s) != s or not 0 <= s < 2**31 for s in cfg['seeds']):
        raise ValueError("seeds must be nonnegative integers below 2**31")
    if cfg['task'] == 'burgers' and (
            not math.isfinite(cfg['viscosity']) or not 0 < cfg['viscosity'] <= 0.05):
        raise ValueError("this study supports low-viscosity Burgers: 0 < nu <= 0.05")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config/sb_cfo.yaml'))
    parser.add_argument('--task', choices=['burgers', 'darcy', 'all'], required=True)
    parser.add_argument('--viscosities', nargs='+', type=float)
    parser.add_argument('--alphas', nargs='+', type=float)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--methods', nargs='+', choices=['fedavg', 'sb_cfo'],
                        help='default: paired fedavg and sb_cfo; IID is always included')
    parser.add_argument(
        '--optimizer', choices=['sgd', 'adam'],
        help='override the YAML optimizer; project default: adam; sgd is an ablation')
    parser.add_argument(
        '--lr', type=float,
        help='explicit learning rate; optimizer overrides default to Adam 0.001 / SGD 0.01')
    parser.add_argument(
        '--momentum', type=float,
        help='SGD momentum only; default 0.9 for SGD, 0 for Adam')
    nesterov_group = parser.add_mutually_exclusive_group()
    nesterov_group.add_argument('--nesterov', action='store_true', default=None,
                               help='enable Nesterov momentum (default for SGD)')
    nesterov_group.add_argument('--no-nesterov', dest='nesterov', action='store_false',
                               help='disable Nesterov momentum')
    parser.add_argument('--device')
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--local-steps', type=int)
    parser.add_argument('--participation-rate', type=float,
                        help='fraction of clients per round; partial participation: FedAvg only')
    parser.add_argument('--lr-schedule', choices=['cosine'],
                        help='round-wise cosine decay (the only supported schedule)')
    parser.add_argument('--min-lr', type=float,
                        help='cosine final-round learning rate; Adam: 0.0001, SGD: 0.001')
    parser.add_argument('--log-every', type=int)
    parser.add_argument('--strength', type=float)
    parser.add_argument('--refresh-every', type=int)
    parser.add_argument('--samples-per-client-bin', type=int)
    parser.add_argument('--output-dir')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    if args.viscosities and args.task == 'darcy':
        parser.error('--viscosities applies to Burgers, not Darcy')
    return args


def main_cli(argv=None):
    args = parse_args(argv)
    for cfg in build_plan(args):
        t = cfg['training']
        print("{} nu={} | {} lr={} momentum={} | {} rounds x {} steps".format(
            cfg['task'], cfg['viscosity'], t['optimizer'], t['lr'], t['momentum'],
            t['n_rounds'], t['local_steps']), flush=True)
        print("  methods={} | schedule={} min_lr={} participation={}".format(
            cfg['methods'], t['lr_schedule'], t['min_lr'],
            t.get('participation_rate', 1.)), flush=True)
        print(f"  nesterov={t.get('nesterov', False)}", flush=True)
        print("  IID + alphas={} | seeds={} | {}".format(
            cfg['alpha_values'], cfg['seeds'], cfg['output_dir']), flush=True)
        if args.dry_run:
            continue
        from sb_cfo.runner import run_experiment
        run_experiment(cfg, force=args.force, validate_only=args.validate_only)


if __name__ == '__main__':
    main_cli()
