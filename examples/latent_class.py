"""
Latent class logit on the three benchmark datasets.

K classes, each with its own time and cost coefficients, share the ASCs; class
membership shares are constant. Coefficients are kept negative (β = −exp(raw)):
without that constraint a class can pick up wrongly signed time or cost
sensitivity, which the behavioral-regularity diagnostic flags (try
``--unconstrained``). On ModeCanada, where time and cost are strongly
confounded (air is both fast and expensive), two classes separate into a
time-only and a cost-only class, so the implied VOT is degenerate: a real
property of that specification, which the VOT distribution makes visible.
DCMBench sees the model as a ``Mixture``, so
its VOT distribution is K weighted values (population view), or each person's
posterior class probabilities (``vot_posterior=True``).

    python examples/latent_class.py [--protocol cv] [--classes 3]
"""

from __future__ import annotations

import torch
from torch import nn

import common
from dcmbench.engines.torch_module import Mixture, TorchModuleFitter, Training
from dcmbench.evaluation import evaluate


class LatentClassLogit(nn.Module):
    def __init__(self, dataset_name: str, problem, n_classes: int = 2, constrained: bool = True):
        super().__init__()
        self.dataset_name, self.problem, self.n_classes, self.constrained = dataset_name, problem, n_classes, constrained
        j = len(problem.alternatives)
        self.reference = problem.alternatives.index(common.REFERENCE[dataset_name])
        # Classes start apart (more and less time-sensitive) to break the symmetry between them.
        start = torch.tensor([[-1.0 * (k + 1) / n_classes, -1.0] for k in range(n_classes)], dtype=torch.float64)
        self.raw = nn.Parameter(torch.log(-start) if constrained else start)  # (K, 2): time, cost
        self.asc = nn.Parameter(torch.zeros(j - 1))
        self.log_share = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x, available):
        time, cost = common.time_and_cost(self.dataset_name, self.problem, x)  # (n, J)
        asc = torch.cat([self.asc[:self.reference], torch.zeros(1, dtype=time.dtype), self.asc[self.reference:]])
        beta = -torch.exp(self.raw) if self.constrained else self.raw
        utilities = asc + beta[None, :, 0:1] * time[:, None, :] + beta[None, :, 1:2] * cost[:, None, :]
        return Mixture(utilities, self.log_share.expand(time.shape[0], self.n_classes))


def fitter(dataset, n_classes: int = 2, constrained: bool = True, seed: int = 0):
    return TorchModuleFitter(
        lambda: LatentClassLogit(dataset.name, dataset.problem, n_classes, constrained),
        dataset.problem,
        columns=common.module_columns(dataset),
        training=Training(optimizer="lbfgs", learning_rate=1.0, batch_size=None, max_epochs=50, patience=3,
                          validation_fraction=0.0),
        seed=seed,
        name=f"latent class ({n_classes} classes)",
    )


def main():
    protocol, datasets, args = common.protocol_from_args(
        __doc__.splitlines()[1],
        extra=lambda parser: (parser.add_argument("--classes", type=int, default=2),
                              parser.add_argument("--unconstrained", action="store_true")))
    for name in datasets:
        dataset = common.benchmark_dataset(name)
        label = f"latent class ({args.classes}{', unconstrained' if args.unconstrained else ''})"
        result = evaluate(fitter(dataset, args.classes, not args.unconstrained), dataset, name=label, protocol=protocol,
                          scenarios=[common.car_cost_scenario(name)])
        common.report(result)


if __name__ == "__main__":
    main()
