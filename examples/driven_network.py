"""Fit a Neural Chart to a small driven network, then read its physics.  python examples/driven_network.py"""
import torch

from neuralchart import Conditions, Hypergraph, NeuralChart, Trajectories

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
net = Hypergraph.from_strings(['A + X <-> 2 X', 'X <-> Y', 'Y <-> B'], chemostats=['A', 'B'])
print('cycles', net.cycle_basis().shape[1], 'conservation laws', net.conservation_laws().shape[1])


def true_rate(c, log_a):
    x, y = c.T
    ax = x * torch.exp(0.5 * x)                # the activity of X: an excess free energy the chart has to learn
    j1, j2, j3 = log_a.exp() * ax - ax ** 2, 0.5 * (ax - y), y - 1.0
    return torch.stack([j1 - j2, j2 - j3], 1)


log_a, dt, c = torch.linspace(-0.5, 0.5, 16), 0.02, [0.2 + torch.rand(16, 2)]
for _ in range(100):
    c.append(c[-1] + dt * true_rate(c[-1], log_a))
conditions = Conditions(chem_logc=torch.stack([log_a, torch.zeros(16)], 1))
data = Trajectories(torch.stack(c, 1), conditions, dt=dt)

chart = NeuralChart(net, chemostats=[0.0, 0.0])
chart.fit(data, iters=300, log=lambda *a: None)
rollout = chart.rollout(data.c[:, 0], conditions, data.c.shape[1] - 1)
print('relative rollout error %.3f' % float((rollout - data.c).norm() / data.c.norm()))
print('entropy production', chart.entropy_production(data.c[:, -1], conditions)[:3].detach().numpy().round(3))
print('cycle affinity', chart.cycle_affinity(conditions)[:3].detach().numpy().round(3).ravel())
