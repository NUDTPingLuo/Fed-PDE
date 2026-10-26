"""Schedule endpoints and independent persistent-optimizer references."""
import copy
import math
import unittest
import numpy as np
import torch
from sb_cfo.core import ClientShard, participation_schedule, train
from util.schedules import round_learning_rate, validate_schedule
from sb_cfo_main import build_plan, parse_args


class ScheduleTests(unittest.TestCase):
    def test_requested_plan(self):
        cfg = build_plan(parse_args([
            '--task', 'burgers', '--viscosities', '0.005', '--alphas', '0.01',
            '--methods', 'fedavg', 'sb_cfo', '--optimizer', 'adam', '--lr', '0.001',
            '--lr-schedule', 'cosine', '--min-lr', '0.0001',
            '--local-steps', '5', '--rounds', '1000']))[0]
        self.assertEqual(cfg['methods'], ['fedavg', 'sb_cfo'])
        self.assertEqual(cfg['seeds'], [0, 1, 42, 999, 2026])
        self.assertEqual(cfg['alpha_values'], [.01])
        rates = [round_learning_rate(cfg['training'], r) for r in range(1000)]
        self.assertAlmostEqual(rates[0], .001)
        self.assertAlmostEqual(rates[-1], .0001)
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
        self.assertAlmostEqual(rates[499] + rates[500], .0011)

    def test_validation_and_single_round(self):
        cfg = dict(lr=.01, min_lr=.0001, lr_schedule='cosine', n_rounds=1)
        self.assertEqual(round_learning_rate(cfg, 0), .01)
        for change in (dict(min_lr=-1), dict(min_lr=.02), dict(min_lr=float('nan')),
                       dict(lr_schedule='bad'), dict(n_rounds=0), dict(lr=0)):
            with self.assertRaises(ValueError):
                validate_schedule(dict(cfg, **change))
        for r in (-1, 1, .5):
            with self.assertRaises(ValueError):
                round_learning_rate(cfg, r)
        for args in (['--min-lr', '-1'], ['--min-lr', '.1'],
                     ['--methods', 'sb_cfo'], ['--methods', 'fedavg', 'fedavg']):
            with self.assertRaises(ValueError):
                build_plan(parse_args(['--task', 'burgers'] + args))

    def test_default_cosine_and_rejected_legacy_schedule(self):
        config = dict(lr=.001, n_rounds=4)
        rates = [round_learning_rate(config, r) for r in range(4)]
        self.assertAlmostEqual(rates[0], .001)
        self.assertAlmostEqual(rates[-1], .0001)
        self.assertTrue(all(a > b for a, b in zip(rates, rates[1:])))
        with self.assertRaisesRegex(ValueError, 'lr_schedule must be cosine'):
            validate_schedule(dict(config, lr_schedule='constant'))
        with self.assertRaises(SystemExit):
            parse_args(['--task', 'darcy', '--lr-schedule', 'constant'])

    def test_actual_updates_match_independent_persistent_optimizers(self):
        # Reference retains a separate worker and optimizer per client.
        # Production shares a worker: this catches stale lr after state restore.
        torch.set_num_threads(1)
        clients = []
        for k, n in enumerate((4, 6, 8)):
            x = np.linspace(.1, 1, n).astype(np.float32)[:, None]
            clients.append(ClientShard(x, x * (k + 1), np.arange(n) % 2))
        control = dict(n_bins=2, strength=0., refresh_every=1, samples_per_client_bin=0)
        for name, momentum in (('adam', 0.), ('sgd', .9)):
            for rate in (1., .5):
                with self.subTest(optimizer=name, participation=rate):
                    initial = torch.nn.Linear(1, 1, bias=False)
                    with torch.no_grad():
                        initial.weight.fill_(.3)
                    cfg = dict(optimizer=name, momentum=momentum,
                               lr=.01, min_lr=.0001, lr_schedule='cosine', n_rounds=5,
                               local_steps=2, batch_size=99, device='cpu', log_every=99,
                               participation_rate=rate)
                    workers = [copy.deepcopy(initial) for _ in clients]
                    optimizers = [(torch.optim.Adam(w.parameters(), lr=.01) if name == 'adam'
                                   else torch.optim.SGD(w.parameters(), lr=.01, momentum=.9))
                                  for w in workers]
                    reference = copy.deepcopy(initial)
                    for t, selected in enumerate(participation_schedule(3, 5, rate, 7)):
                        lr = .0001 + (.01 - .0001) * (1 + math.cos(math.pi * t / 4)) / 2
                        for k in selected:
                            workers[k].load_state_dict(reference.state_dict())
                            optimizers[k].param_groups[0]['lr'] = lr
                            x, y = torch.from_numpy(clients[k].inputs), torch.from_numpy(clients[k].targets)
                            for _ in range(2):
                                optimizers[k].zero_grad()
                                (workers[k](x) - y).square().mean().backward()
                                optimizers[k].step()
                        total = sum(len(clients[k].inputs) for k in selected)
                        average = sum(workers[k].weight.detach() * (len(clients[k].inputs) / total)
                                      for k in selected)
                        with torch.no_grad():
                            reference.weight.copy_(average)
                    rows, actual = train(copy.deepcopy(initial), clients, clients[0].inputs,
                                         clients[0].targets, cfg, control, 7, 'fedavg', list(range(6)))
                    torch.testing.assert_close(actual['weight'], reference.weight, atol=1e-7, rtol=1e-6)
                    self.assertTrue(math.isnan(rows[0]['learning_rate']))
                    self.assertAlmostEqual(rows[-1]['learning_rate'], .0001)
                    if rate == 1.:
                        controlled, other = train(copy.deepcopy(initial), clients, clients[0].inputs,
                                                  clients[0].targets, cfg, control, 7, 'sb_cfo', list(range(6)))
                        torch.testing.assert_close(other['weight'], actual['weight'], atol=0, rtol=0)
                        self.assertEqual([r['learning_rate'] for r in rows[1:]],
                                         [r['learning_rate'] for r in controlled[1:]])


if __name__ == '__main__':
    unittest.main()
