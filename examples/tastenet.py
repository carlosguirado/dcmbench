"""
TasteNet-MNL (Han, Pereira, Ben-Akiva and Zegras, 2022) on the three benchmark datasets.

A taste network maps individual characteristics z to each person's
coefficients; the utility stays MNL-shaped, in willingness-to-pay space:

    V_nj = ASC_j(z_n) + λ · (β_time,j(z_n) · time_nj − cost_nj),   β_time = −vot0 · exp(−raw)

The network (one ReLU hidden layer) outputs J alternative-specific time
coefficients, kept negative by −exp, and J − 1 ASCs. Following the official
code (github.com/deborahmit/TasteNet-MNL), the cost coefficient is fixed;
here a free scale λ is added so the same model works across datasets and
currencies (with λ = 1 and the Swissmetro inputs it reduces to the original).

Each person gets their own VOT; DCMBench computes it from the model's
marginal utilities by autograd (VOT Route A).

    python examples/tastenet.py [--protocol cv] [--datasets swissmetro]
"""

from __future__ import annotations

import math

import torch
from torch import nn

import common
from dcmbench.engines.torch_module import TorchModuleFitter, Training
from dcmbench.evaluation import evaluate

# Typical VOT in each dataset's module units (cost per time unit), for a sensible starting point.
VOT0 = {"swissmetro": 1.0, "modecanada": 0.5, "lpmc": 10.0}
SCALE0 = {"swissmetro": 1.0, "modecanada": 3.0, "lpmc": 1.0}


class TasteNet(nn.Module):
    def __init__(self, dataset_name: str, problem, characteristics, hidden: int = 110, free_scale: bool = True):
        super().__init__()
        self.dataset_name, self.problem, self.characteristics = dataset_name, problem, list(characteristics)
        j = len(problem.alternatives)
        self.reference = problem.alternatives.index(common.REFERENCE[dataset_name])
        self.net = nn.Sequential(nn.Linear(len(characteristics), hidden), nn.ReLU(), nn.Linear(hidden, 2 * j - 1))
        self.log_vot0 = math.log(VOT0[dataset_name])
        self.log_scale = nn.Parameter(torch.tensor(math.log(SCALE0[dataset_name])), requires_grad=free_scale)

    def tastes(self, x):
        z = torch.stack([x[c] for c in self.characteristics], dim=1)
        raw = self.net(z)
        j = len(self.problem.alternatives)
        beta_time = -torch.exp(self.log_vot0 - raw[:, :j])
        asc = torch.cat([raw[:, j:j + self.reference], torch.zeros_like(raw[:, :1]), raw[:, j + self.reference:]], dim=1)
        return beta_time, asc

    def forward(self, x, available):
        beta_time, asc = self.tastes(x)
        time, cost = common.time_and_cost(self.dataset_name, self.problem, x)
        return asc + torch.exp(self.log_scale) * (beta_time * time - cost)


def fitter(dataset, seed: int = 0):
    characteristics = common.characteristics_present(dataset)
    return TorchModuleFitter(
        lambda: TasteNet(dataset.name, dataset.problem, characteristics),
        dataset.problem,
        columns=common.module_columns(dataset) + characteristics,
        standardize=characteristics,
        # As in the official code: Adam, lr 1e-3, batch 32, best-validation checkpoint.
        training=Training(optimizer="adam", learning_rate=1e-3, batch_size=32, max_epochs=100, patience=5),
        seed=seed,
        name="TasteNet-MNL",
    )


def main():
    protocol, datasets = common.protocol_from_args(__doc__.splitlines()[1])
    for name in datasets:
        dataset = common.benchmark_dataset(name)
        result = evaluate(fitter(dataset), dataset, name="TasteNet-MNL", protocol=protocol,
                          scenarios=[common.car_cost_scenario(name)])
        common.report(result)


if __name__ == "__main__":
    main()
