"""Policy scenarios: predicted shares, diversion ratios and arc elasticities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dcmbench.evaluation.perturb import Perturbation
from dcmbench.evaluation.predict import predict
from dcmbench.problem import ChoiceProblem
from dcmbench.results import AnalysisResult


@dataclass(frozen=True)
class Scenario:
    """A named set of perturbations applied together, e.g. ``car cost × 1.5``."""

    name: str
    perturbations: Tuple[Perturbation, ...]

    def __post_init__(self):
        object.__setattr__(self, "perturbations", tuple(self.perturbations))
        if not self.perturbations:
            raise ValueError("a scenario needs at least one perturbation")

    def apply(self, data: pd.DataFrame, problem: ChoiceProblem) -> pd.DataFrame:
        for perturbation in self.perturbations:
            data = perturbation.apply(data, problem)
        return data


def market_shares(probabilities: pd.DataFrame, data: pd.DataFrame, problem: ChoiceProblem) -> pd.Series:
    """Sample (or weighted) mean of predicted probabilities."""
    if problem.weight_column is None:
        return probabilities.mean()
    weights = data[problem.weight_column].to_numpy(dtype=float)
    return pd.Series(np.average(probabilities.to_numpy(), axis=0, weights=weights), index=probabilities.columns)


def _single_target(scenario: Scenario, problem: ChoiceProblem) -> Optional[str]:
    targets = {alt for p in scenario.perturbations for alt in p.targets(problem)}
    return targets.pop() if len(targets) == 1 else None


def run_scenario(
    model,
    data: pd.DataFrame,
    problem: ChoiceProblem,
    scenario: Scenario,
    diversion_from: Optional[str] = None,
) -> AnalysisResult:
    """
    Shares before and after ``scenario``, with:

    - ``diversion``: each alternative's gain as a fraction of the loss of
      ``diversion_from`` (default: the single perturbed alternative);
    - ``arc_elasticity``: midpoint arc elasticity of each share, when the
      scenario is a single multiplicative change.
    """
    base = market_shares(predict(model, data, problem), data, problem)
    after = market_shares(predict(model, scenario.apply(data, problem), problem), data, problem)
    table = pd.DataFrame({"alternative": problem.alternatives, "share_base": base.to_numpy(),
                          "share_scenario": after.to_numpy()})
    table["change"] = table["share_scenario"] - table["share_base"]
    table["pct_change"] = 100 * table["change"] / table["share_base"]

    flags = []
    source = diversion_from or _single_target(scenario, problem)
    if source is not None:
        loss = -float(table.loc[table["alternative"] == source, "change"].iloc[0])
        if loss > 0:
            table["diversion"] = np.where(table["alternative"] == source, np.nan, table["change"] / loss)
        else:
            table["diversion"] = np.nan
            flags.append(f"{source} does not lose share; diversion ratios undefined")

    if len(scenario.perturbations) == 1 and scenario.perturbations[0].multiply is not None:
        m = scenario.perturbations[0].multiply
        relative_x = (m - 1) / ((m + 1) / 2)
        midpoint_share = (table["share_scenario"] + table["share_base"]) / 2
        table["arc_elasticity"] = (table["change"] / midpoint_share) / relative_x

    if getattr(model, "piecewise_constant", False):
        flags.append("model is piecewise constant: responses to small changes may be zero or step-like")
    return AnalysisResult(
        kind="scenario",
        tables={"summary": table},
        settings={"scenario": scenario.name, "perturbations": [p.describe() for p in scenario.perturbations],
                  "diversion_from": source},
        flags=flags,
    )


def sensitivity(
    model,
    data: pd.DataFrame,
    problem: ChoiceProblem,
    attribute: str,
    alternatives: Optional[Sequence[str]],
    multipliers: Sequence[float],
) -> AnalysisResult:
    """Predicted shares as one attribute is scaled by each multiplier (a sensitivity curve)."""
    rows = []
    for m in multipliers:
        scenario = Scenario(f"{attribute} × {m}", (Perturbation(attribute, alternatives, multiply=m),))
        shares = market_shares(predict(model, scenario.apply(data, problem), problem), data, problem)
        rows += [{"multiplier": m, "alternative": alt, "share": float(shares[alt])} for alt in problem.alternatives]
    return AnalysisResult(
        kind="sensitivity",
        tables={"summary": pd.DataFrame(rows)},
        settings={"attribute": attribute, "alternatives": None if alternatives is None else list(alternatives),
                  "multipliers": list(multipliers)},
    )
