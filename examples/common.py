"""
Shared pieces for the example scripts: the benchmark samples, dataset-specific
inputs for hand-written models, and a compact report.

Run the examples from the repository root, e.g. ``python examples/tastenet.py``.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np

import dcmbench.data
from dcmbench.evaluation import CrossValidation, Holdout, Perturbation, Scenario

DATASETS = ("swissmetro", "modecanada", "lpmc")

# Reference alternative (no ASC) and the alternative whose cost the standard scenario raises.
REFERENCE = {"swissmetro": "sm", "modecanada": "car", "lpmc": "drive"}
CAR = {"swissmetro": "car", "modecanada": "car", "lpmc": "drive"}

# Divisors bringing time and cost to similar magnitudes inside hand-written modules.
# Marginal utilities are computed by autograd in the raw data units regardless.
UNITS = {"swissmetro": (100.0, 100.0), "modecanada": (100.0, 100.0), "lpmc": (1.0, 1.0)}

# Individual characteristics for models with taste heterogeneity (numeric columns only).
CHARACTERISTICS = {
    "swissmetro": ["MALE", "FIRST", "GA", "business", "age_2", "age_3", "age_4", "age_5", "age_6",
                   "income_2", "income_3", "income_4", "who_2", "who_3", "luggage_1", "luggage_3"],
    "modecanada": ["income", "urban_1", "urban_2", "dist"],
    "lpmc": ["age", "female", "driving_license", "car_ownership", "purpose_HBW", "purpose_HBE",
             "purpose_NHBO", "purpose_B", "distance"],
}


def benchmark_dataset(name: str):
    """The benchmark sample of a dataset, with the derived columns the examples use."""
    dataset = dcmbench.data.load(name)
    if "benchmark" in dataset.preparations:
        dataset = dataset.prepare("benchmark")
    if name == "swissmetro":
        codes = {"age": ("AGE", (2, 3, 4, 5, 6)), "income": ("INCOME", (2, 3, 4)), "who": ("WHO", (2, 3)),
                 "luggage": ("LUGGAGE", (1, 3))}
        dummies = {f"{prefix}_{v}": (lambda f, c=column, v=v: (f[c] == v).astype(int))
                   for prefix, (column, values) in codes.items() for v in values}
        dataset = dataset.derive("characteristic dummies (AGE, INCOME, WHO, LUGGAGE; business = PURPOSE 3)",
                                 business=lambda f: (f.PURPOSE == 3).astype(int), **dummies)
    if name == "modecanada":
        dataset = dataset.derive("urban dummies", urban_1=lambda f: (f.urban == 1).astype(int),
                                 urban_2=lambda f: (f.urban == 2).astype(int))
    if name == "lpmc":
        dataset = dataset.derive("trip purpose dummies (reference: HBO)",
                                 **{f"purpose_{p}": (lambda f, p=p: (f.purpose == p).astype(int))
                                    for p in ("HBW", "HBE", "NHBO", "B")})
    return dataset


def attribute_columns(dataset) -> List[str]:
    """Every time and cost column of the choice problem."""
    problem = dataset.problem
    return sorted({c for a in ("time", "cost") for c in problem.attributes[a].columns.values()})


def module_columns(dataset) -> List[str]:
    """Columns hand-written modules receive: time and cost, plus GA for Swissmetro's free travel."""
    return attribute_columns(dataset) + (["GA"] if dataset.name == "swissmetro" else [])


def time_and_cost(dataset_name: str, problem, x) -> Tuple[object, object]:
    """
    ``(n, J)`` torch tensors of time and cost per alternative in the dataset's
    module units. Alternatives without an attribute get 0; Swissmetro train and
    Swissmetro costs are 0 for travel-pass (GA) holders, as in the benchmark specs.
    """
    import torch

    time_div, cost_div = UNITS[dataset_name]
    n = next(iter(x.values())).shape[0]
    zero = torch.zeros(n, dtype=torch.float64)
    time = torch.stack([x[problem.attributes["time"].columns[a]] / time_div
                        if a in problem.attributes["time"].columns else zero for a in problem.alternatives], dim=1)
    costs = []
    for a in problem.alternatives:
        column = problem.attributes["cost"].columns.get(a)
        cost = zero if column is None else x[column] / cost_div
        if dataset_name == "swissmetro" and a in ("train", "sm"):
            cost = cost * (1 - x["GA"])
        costs.append(cost)
    return time, torch.stack(costs, dim=1)


def car_cost_scenario(dataset_name: str) -> Scenario:
    car = CAR[dataset_name]
    return Scenario(f"{car} cost x1.5", [Perturbation("cost", (car,), multiply=1.5)])


def protocol_from_args(description: str, extra=None):
    """Parse ``--protocol`` and ``--datasets`` (plus script-specific options via ``extra(parser)``)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--protocol", choices=("holdout", "cv"), default="holdout",
                        help="80/20 stratified holdout (default) or 5-fold respondent-grouped CV")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    if extra is not None:
        extra(parser)
    args = parser.parse_args()
    protocol = Holdout() if args.protocol == "holdout" else CrossValidation(n_splits=5)
    return (protocol, args.datasets, args) if extra is not None else (protocol, args.datasets)


def report(result) -> Dict[str, object]:
    """A one-line-per-dataset summary of an EvaluationResult."""
    prediction = result["prediction"].summary.iloc[0] if "prediction" in result.analyses else None

    def pick(*names):
        for name in names:
            if prediction is not None and name in prediction.index:
                return float(prediction[name])
        return float("nan")

    line = {
        "model": result.model_name,
        "dataset": result.problem_name,
        "fit_ll": round(result.fit.loglikelihood, 1),
        "rho_bar_sq": None if result.fit.rho_bar_sq is None else round(result.fit.rho_bar_sq, 3),
        "oos_ll": round(pick("loglikelihood", "avg_ll_mean"), 3),
        "oos_macro_f1": round(pick("macro_f1", "macro_f1_mean"), 3),
    }
    if "vot" in result.analyses:
        vot = result["vot"].summary.set_index("alternative")
        line["vot_route"] = result["vot"].settings["route"]
        line["vot_median"] = {a: round(v, 1) for a, v in vot["q50"].items()}
    if "regularity" in result.analyses:
        reg = result["regularity"].summary
        own_cost = reg[reg.attribute == "cost"]
        line["cost_regularity_strong"] = {r.alternative: round(r.strong, 3) for r in own_cost.itertuples()}
    scenarios = [k for k in result.analyses if k.startswith("scenario")]
    if scenarios:
        table = result[scenarios[0]].summary.set_index("alternative")
        car = CAR[result.problem_name]
        line["car_share_pct_change"] = round(float(table.loc[car, "pct_change"]), 1)
    print(line, flush=True)
    return line


def characteristics_present(dataset) -> List[str]:
    missing = [c for c in CHARACTERISTICS[dataset.name] if c not in dataset.frame.columns]
    if missing:
        raise KeyError(f"{dataset.name} lacks characteristics {missing}")
    return CHARACTERISTICS[dataset.name]


def seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
