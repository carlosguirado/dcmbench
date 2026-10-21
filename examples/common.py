"""
Shared pieces for the example scripts.

The hand-written models (TasteNet, ResLogit, latent class) work on any
dataset: everything they need to know about it is a ``ModelInputs``, built
from the ``ChoiceProblem`` (time and cost columns, availability) plus a little
configuration (reference alternative, individual characteristics, cost
multipliers). ``benchmark_inputs`` gives that configuration for the three
benchmark datasets; for your own data, build a ``ModelInputs`` yourself:

    inputs = ModelInputs.from_dataset(my_dataset, reference="car", characteristics=["age", "income"])

Run the examples from the repository root, e.g. ``python examples/tastenet.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

import dcmbench.data
from dcmbench.evaluation import CrossValidation, Holdout, Perturbation, Scenario

DATASETS = ("swissmetro", "modecanada", "lpmc")


@dataclass(frozen=True)
class ModelInputs:
    """
    What a hand-written utility module needs to know about a choice problem.

    - ``reference``: the alternative without an ASC.
    - ``characteristics``: numeric columns describing the decision maker (for taste heterogeneity).
    - ``cost_multiplier``: optional column multiplying an alternative's cost (e.g. Swissmetro's
      ``PAYS = 1 - GA``: travel-pass holders pay nothing for train and Swissmetro).
    - ``time_scale``, ``cost_scale``: one divisor per attribute, shared by all alternatives so
      generic coefficients stay generic. Marginal utilities are computed by autograd in raw
      units regardless.
    """

    problem: object
    reference: str
    characteristics: Tuple[str, ...] = ()
    cost_multiplier: Mapping[str, str] = field(default_factory=dict)
    time_attribute: str = "time"
    cost_attribute: str = "cost"
    time_scale: float = 1.0
    cost_scale: float = 1.0

    @classmethod
    def from_dataset(cls, dataset, reference: str, characteristics=(), cost_multiplier=None,
                     time_attribute: str = "time", cost_attribute: str = "cost") -> "ModelInputs":
        """Scales are the mean absolute value of each attribute over all its columns."""
        problem, frame = dataset.problem, dataset.frame

        def scale(attribute):
            values = np.concatenate([frame[c].to_numpy(dtype=float) for c in problem.attributes[attribute].columns.values()])
            return float(np.mean(np.abs(values))) or 1.0

        if reference not in problem.alternatives:
            raise ValueError(f"reference {reference!r} is not an alternative of {problem.name}")
        return cls(problem, reference, tuple(characteristics), dict(cost_multiplier or {}), time_attribute,
                   cost_attribute, scale(time_attribute), scale(cost_attribute))

    @property
    def alternatives(self) -> Tuple[str, ...]:
        return self.problem.alternatives

    def _columns(self, attribute) -> Dict[str, str]:
        return dict(self.problem.attributes[attribute].columns)

    @property
    def columns(self) -> List[str]:
        """Every column the modules read."""
        columns = list(self._columns(self.time_attribute).values()) + list(self._columns(self.cost_attribute).values())
        columns += list(self.cost_multiplier.values()) + list(self.characteristics)
        return list(dict.fromkeys(columns))

    def time_and_cost(self, x):
        """``(n, J)`` torch tensors of scaled time and cost; alternatives without the attribute get 0."""
        import torch

        n = next(iter(x.values())).shape[0]
        zero = torch.zeros(n, dtype=torch.float64)
        time_cols, cost_cols = self._columns(self.time_attribute), self._columns(self.cost_attribute)
        time = torch.stack([x[time_cols[a]] / self.time_scale if a in time_cols else zero for a in self.alternatives], dim=1)
        costs = []
        for a in self.alternatives:
            cost = x[cost_cols[a]] / self.cost_scale if a in cost_cols else zero
            if a in self.cost_multiplier:
                cost = cost * x[self.cost_multiplier[a]]
            costs.append(cost)
        return time, torch.stack(costs, dim=1)

    def asc(self, free, dtype):
        """Insert a zero for the reference alternative into ``free`` (J − 1 values) → J values."""
        import torch

        k = self.alternatives.index(self.reference)
        return torch.cat([free[..., :k], torch.zeros(free.shape[:-1] + (1,), dtype=dtype), free[..., k:]], dim=-1)


# --- the three benchmark datasets ----------------------------------------------------------------

REFERENCE = {"swissmetro": "sm", "modecanada": "car", "lpmc": "drive"}
CAR = {"swissmetro": "car", "modecanada": "car", "lpmc": "drive"}
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
        dataset = dataset.derive("characteristic dummies (AGE, INCOME, WHO, LUGGAGE; business = PURPOSE 3); "
                                 "PAYS = 1 - GA", business=lambda f: (f.PURPOSE == 3).astype(int),
                                 PAYS=lambda f: 1 - f.GA, **dummies)
    if name == "modecanada":
        dataset = dataset.derive("urban dummies", urban_1=lambda f: (f.urban == 1).astype(int),
                                 urban_2=lambda f: (f.urban == 2).astype(int))
    if name == "lpmc":
        dataset = dataset.derive("trip purpose dummies (reference: HBO)",
                                 **{f"purpose_{p}": (lambda f, p=p: (f.purpose == p).astype(int))
                                    for p in ("HBW", "HBE", "NHBO", "B")})
    return dataset


def benchmark_inputs(dataset) -> ModelInputs:
    """The ``ModelInputs`` configuration of a benchmark dataset."""
    multiplier = {"train": "PAYS", "sm": "PAYS"} if dataset.name == "swissmetro" else {}
    return ModelInputs.from_dataset(dataset, reference=REFERENCE[dataset.name],
                                    characteristics=CHARACTERISTICS[dataset.name], cost_multiplier=multiplier)


def car_cost_scenario(dataset_name: str) -> Scenario:
    car = CAR[dataset_name]
    return Scenario(f"{car} cost x1.5", [Perturbation("cost", (car,), multiply=1.5)])


# --- command line and reporting ------------------------------------------------------------------

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
    """A one-line summary of an EvaluationResult."""
    prediction = result["prediction"].summary.iloc[0] if "prediction" in result.analyses else None

    def pick(*names) -> Optional[float]:
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
    if scenarios and result.problem_name in CAR:
        table = result[scenarios[0]].summary.set_index("alternative")
        line["car_share_pct_change"] = round(float(table.loc[CAR[result.problem_name], "pct_change"]), 1)
    print(line, flush=True)
    return line
