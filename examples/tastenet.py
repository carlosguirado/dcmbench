"""
TasteNet-MNL (Han, Pereira, Ben-Akiva and Zegras, 2022), for any dataset.

A taste network maps individual characteristics z to each person's
coefficients; the utility stays MNL-shaped, in willingness-to-pay space:

    V_nj = ASC_j(z_n) + λ · (β_time,j(z_n) · time_nj − cost_nj),   β_time = −exp(−raw)

The network (one ReLU hidden layer) outputs J alternative-specific time
coefficients, kept negative by −exp, and J − 1 ASCs. Following the official
code (github.com/deborahmit/TasteNet-MNL), the cost coefficient is fixed;
here a free scale λ is added so the same model works across datasets and
currencies (``examples/tastenet_fidelity.py`` runs the original exactly).

Each person gets their own VOT; DCMBench computes it from the model's
marginal utilities by autograd (VOT Route A).

    python examples/tastenet.py [--protocol cv] [--datasets swissmetro]

On your own data: ``fitter(dataset, ModelInputs.from_dataset(dataset, reference=..., characteristics=[...]))``.
"""

from __future__ import annotations

import torch
from torch import nn

import common
from common import ModelInputs
from dcmbench.engines.torch_module import TorchModuleFitter, Training
from dcmbench.evaluation import evaluate


class TasteNet(nn.Module):
    def __init__(self, inputs: ModelInputs, hidden: int = 110):
        super().__init__()
        if not inputs.characteristics:
            raise ValueError("TasteNet needs individual characteristics")
        self.inputs = inputs
        j = len(inputs.alternatives)
        self.net = nn.Sequential(nn.Linear(len(inputs.characteristics), hidden), nn.ReLU(), nn.Linear(hidden, 2 * j - 1))
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def tastes(self, x):
        z = torch.stack([x[c] for c in self.inputs.characteristics], dim=1)
        raw = self.net(z)
        j = len(self.inputs.alternatives)
        return -torch.exp(-raw[:, :j]), self.inputs.asc(raw[:, j:], raw.dtype)

    def forward(self, x, available):
        beta_time, asc = self.tastes(x)
        time, cost = self.inputs.time_and_cost(x)
        return asc + torch.exp(self.log_scale) * (beta_time * time - cost)


def fitter(dataset, inputs: ModelInputs, seed: int = 0):
    return TorchModuleFitter(
        lambda: TasteNet(inputs),
        dataset.problem,
        columns=inputs.columns,
        standardize=[c for c in inputs.characteristics if c not in inputs.cost_multiplier.values()],
        # As in the official code: Adam, lr 1e-3, batch 32, best-validation checkpoint.
        training=Training(optimizer="adam", learning_rate=1e-3, batch_size=32, max_epochs=100, patience=5),
        seed=seed,
        name="TasteNet-MNL",
    )


def main():
    protocol, datasets = common.protocol_from_args(__doc__.splitlines()[1])
    for name in datasets:
        dataset = common.benchmark_dataset(name)
        result = evaluate(fitter(dataset, common.benchmark_inputs(dataset)), dataset, name="TasteNet-MNL",
                          protocol=protocol, scenarios=[common.car_cost_scenario(name)])
        common.report(result)


if __name__ == "__main__":
    main()
