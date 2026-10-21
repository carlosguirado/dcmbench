"""
ResLogit (Wong and Farooq, 2021) on the three benchmark datasets.

A linear-in-parameters utility V (generic time and cost) passes through M
residual layers that mix utilities across alternatives:

    h⁰ = V,   hᵐ = hᵐ⁻¹ − softplus(hᵐ⁻¹ Θᵐ),   U = hᴹ + ASC

with J × J matrices Θᵐ initialised at the identity and ASCs added after the
residual stack, as in the original code (github.com/mwong009/reslogit).
Utilities of unavailable alternatives are zeroed before the residual layers
so they cannot leak into the others.

Time and cost reach the model only through V_j, so every VOT definition, the
coefficient ratio or any derivative ratio, reduces to β_time/β_cost.

    python examples/reslogit.py [--protocol cv] [--datasets lpmc]
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

import common
from dcmbench.engines.torch_module import TorchModuleFitter, Training
from dcmbench.evaluation import evaluate


class ResLogit(nn.Module):
    def __init__(self, dataset_name: str, problem, n_layers: int = 16):
        super().__init__()
        self.dataset_name, self.problem = dataset_name, problem
        j = len(problem.alternatives)
        self.reference = problem.alternatives.index(common.REFERENCE[dataset_name])
        self.beta = nn.Parameter(torch.zeros(2))  # time, cost (generic)
        self.asc = nn.Parameter(torch.zeros(j - 1))
        self.theta = nn.ParameterList([nn.Parameter(torch.eye(j)) for _ in range(n_layers)])

    def forward(self, x, available):
        time, cost = common.time_and_cost(self.dataset_name, self.problem, x)
        h = (self.beta[0] * time + self.beta[1] * cost) * available
        for theta in self.theta:
            h = h - functional.softplus(h @ theta)
        asc = torch.cat([self.asc[:self.reference], torch.zeros(1, dtype=h.dtype), self.asc[self.reference:]])
        return h + asc


def fitter(dataset, seed: int = 0, n_layers: int = 16):
    return TorchModuleFitter(
        lambda: ResLogit(dataset.name, dataset.problem, n_layers),
        dataset.problem,
        columns=common.module_columns(dataset),
        # As in the paper: RMSprop, learning rate 1e-3, mini-batches of 64, early stopping, no regularization.
        training=Training(optimizer="rmsprop", learning_rate=1e-3, batch_size=64, max_epochs=200, patience=10),
        seed=seed,
        name=f"ResLogit-{n_layers}",
    )


def main():
    protocol, datasets = common.protocol_from_args(__doc__.splitlines()[1])
    for name in datasets:
        dataset = common.benchmark_dataset(name)
        result = evaluate(fitter(dataset), dataset, name="ResLogit-16", protocol=protocol,
                          scenarios=[common.car_cost_scenario(name)])
        common.report(result)


if __name__ == "__main__":
    main()
