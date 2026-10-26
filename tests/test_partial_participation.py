"""Independent reference for selected-client averaging and persistent momentum."""
import copy
import unittest
import numpy as np
import torch
from sb_cfo_main import parse_args, build_plan
from sb_cfo.core import ClientShard, participation_schedule, train
from util.schedules import round_learning_rate


class PartialParticipationTests(unittest.TestCase):
    def test_paired_schedule_and_validation(self):
        a = participation_schedule(10, 500, .5, 42)
        self.assertEqual(a, participation_schedule(10, 500, .5, 42))
        self.assertNotEqual(a, participation_schedule(10, 500, .5, 999))
        self.assertTrue(all(len(s) == len(set(s)) == 5 for s in a))
        self.assertEqual(participation_schedule(3, 2, 1., 0), [[0, 1, 2]] * 2)
        for rate in (0, -1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                participation_schedule(10, 5, rate, 0)

    def test_plan(self):
        cfg = build_plan(parse_args(['--task', 'burgers', '--methods', 'fedavg',
                                '--local-steps', '10', '--rounds', '500',
                                '--participation-rate', '.5', '--optimizer', 'sgd',
                                '--momentum', '.9', '--lr', '.001']))[0]
        self.assertEqual(cfg['methods'], ['fedavg'])
        self.assertEqual(cfg['training']['local_steps'], 10)
        self.assertEqual(cfg['training']['n_rounds'], 500)
        self.assertEqual(cfg['training']['participation_rate'], .5)
        self.assertEqual(cfg['training']['momentum'], .9)
        self.assertEqual(cfg['training']['lr'], .001)
        self.assertEqual(cfg['training']['lr_schedule'], 'cosine')
        with self.assertRaises(ValueError):
            build_plan(parse_args(['--task', 'burgers', '--participation-rate', '.5',
                                   '--methods', 'fedavg', 'sb_cfo']))

    def test_scalar_reference_and_costs(self):
        torch.set_num_threads(1)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(.3)
        clients = []
        for k in range(4):
            x = np.linspace(.1, 1., k + 2).astype(np.float32)[:, None]
            clients.append(ClientShard(x, x * (k + 1.), np.arange(len(x)) % 2))
        cfg = dict(device='cpu', optimizer='sgd', momentum=.9, lr=.001,
                   local_steps=10, n_rounds=12, batch_size=99, log_every=100,
                   participation_rate=.5)
        control = dict(n_bins=2, strength=1., refresh_every=1, samples_per_client_bin=0)
        schedule = participation_schedule(4, 12, .5, 0)
        w, velocities, examples = .3, np.zeros(4), 0
        for t, selected in enumerate(schedule):
            lr = round_learning_rate(cfg, t)
            local_models, sizes = [], []
            for k in selected:
                client, local = clients[k], w
                for _ in range(10):
                    gradient = float(np.mean(2 * client.inputs * (local * client.inputs - client.targets)))
                    velocities[k] = .9 * velocities[k] + gradient
                    local -= lr * velocities[k]
                    examples += len(client.inputs)
                local_models.append(local)
                sizes.append(len(client.inputs))
            w = np.average(local_models, weights=sizes)
        rows, state = train(copy.deepcopy(model), clients, clients[0].inputs, clients[0].targets,
                            cfg, control, 0, 'fedavg', [0, 12])
        self.assertAlmostEqual(float(state['weight']), w, places=6)
        self.assertEqual(rows[-1]['optimizer_examples'], examples)
        self.assertEqual(rows[-1]['model_communication_bytes'], 12 * 2 * 2 * 4)
        full = dict(cfg, participation_rate=1.)
        _, a = train(copy.deepcopy(model), clients, clients[0].inputs, clients[0].targets,
                     full, control, 0, 'fedavg', [12])
        del full['participation_rate']
        _, b = train(copy.deepcopy(model), clients, clients[0].inputs, clients[0].targets,
                     full, control, 0, 'fedavg', [12])
        torch.testing.assert_close(a['weight'], b['weight'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
