"""
RUMBoost (Salvadé and Hillel, 2025) on the three benchmark datasets.

Each alternative's utility is boosted from its own time and cost (monotone
non-increasing) and the individual characteristics, one variable per tree.
Utilities are step functions, so DCMBench flags derivative-based outputs;
RUMBoost cannot use availability while training (see dcmbench.engines.rumboost).

    PYTHONPATH=third_party/rumboost python examples/rumboost_benchmark.py [--protocol cv]
"""

from __future__ import annotations

import common
from dcmbench.engines.rumboost import RumboostFitter
from dcmbench.evaluation import evaluate


def fitter(dataset, seed: int = 0):
    problem = dataset.problem
    characteristics = common.characteristics_present(dataset)
    own = {alt: [problem.attributes[a].columns[alt] for a in ("time", "cost") if alt in problem.attributes[a].columns]
           for alt in problem.alternatives}
    variables = {alt: own[alt] + characteristics for alt in problem.alternatives}
    monotone = {column: -1 for columns in own.values() for column in columns}
    return RumboostFitter(problem, variables, monotone, learning_rate=0.1, max_depth=1, seed=seed)


def main():
    protocol, datasets = common.protocol_from_args(__doc__.splitlines()[1])
    for name in datasets:
        dataset = common.benchmark_dataset(name)
        result = evaluate(fitter(dataset), dataset, name="RUMBoost", protocol=protocol, vot_route="B",
                          scenarios=[common.car_cost_scenario(name)])
        common.report(result)


if __name__ == "__main__":
    main()
