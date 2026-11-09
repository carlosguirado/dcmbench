"""
Behavioral regularity (Feng et al., 2024, Transportation Research Part C, Eqs. 5–6;
arXiv:2404.14701; code at github.com/siqi-feng/DNN-behavioral-regularity).

For alternative i and its own attribute d, the share of observations whose
choice probability falls when x_d rises: strong requires ΔP < −τ, weak
ΔP < +τ. As in the authors' code: a one-sided forward difference with step
h = range(x_d)/200 over the full dataset, and τ = range(x_d)/1e6 applied to
the raw ΔP.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dcmbench.evaluation.perturb import attribute_column, shift
from dcmbench.evaluation.predict import predict
from dcmbench.problem import ChoiceProblem
from dcmbench.results import AnalysisResult

STEP_DIVISOR = 200
TOLERANCE_DIVISOR = 1e6


def behavioral_regularity(
    model,
    data: pd.DataFrame,
    problem: ChoiceProblem,
    attributes: Optional[Sequence[str]] = None,
    range_data: Optional[pd.DataFrame] = None,
    step_divisor: float = STEP_DIVISOR,
    tolerance_divisor: float = TOLERANCE_DIVISOR,
) -> AnalysisResult:
    """
    Strong and weak regularity for every (alternative, own attribute) pair.

    ``attributes`` defaults to all time and cost attributes; the expected
    sign is negative for both. ``range_data`` (default ``data``) is the full
    dataset the step and tolerance are scaled to. Observations where the
    alternative is unavailable are excluded (Feng et al. treat every
    alternative as available).
    """
    if attributes is None:
        attributes = [n for n, a in problem.attributes.items() if a.kind in ("time", "cost")]
    range_data = data if range_data is None else range_data
    base = predict(model, data, problem)
    available = problem.availability_matrix(data)

    rows = []
    for attribute in attributes:
        kind = problem.attributes[attribute].kind
        for alternative in problem.attributes[attribute].columns:
            column = range_data[attribute_column(problem, attribute, alternative)]
            spread = float(column.max() - column.min())
            definition = "feng2024" if kind == "cost" and step_divisor == STEP_DIVISOR \
                and tolerance_divisor == TOLERANCE_DIVISOR else "extension"
            if spread <= 0:
                rows.append({"alternative": alternative, "attribute": attribute, "definition": definition,
                             "strong": np.nan, "weak": np.nan, "n_obs": 0,
                             "n_excluded_unavailable": int((~available[alternative].to_numpy()).sum()),
                             "n_zero_change": 0, "step": np.nan, "tolerance": np.nan})
                continue
            step, tolerance = spread / step_divisor, spread / tolerance_divisor
            after = predict(model, shift(data, problem, attribute, alternative, step), problem)[alternative]
            keep = available[alternative].to_numpy()
            delta = (after - base[alternative]).to_numpy()[keep]
            rows.append({
                "alternative": alternative,
                "attribute": attribute,
                "definition": definition,
                "strong": float(np.mean(delta < -tolerance)),
                "weak": float(np.mean(delta < tolerance)),
                "n_obs": int(keep.sum()),
                "n_excluded_unavailable": int((~keep).sum()),
                "n_zero_change": int(np.sum(delta == 0)),
                "step": step,
                "tolerance": tolerance,
            })

    flags = ["observations where the alternative is unavailable are excluded (Feng et al. assume full availability)"]
    flags += [
        f"no observation responds to {row['attribute']} of {row['alternative']}: the model probably does not use it"
        for row in rows if row["n_obs"] and row["n_zero_change"] == row["n_obs"]
    ]
    flags += [
        f"{row['attribute']} of {row['alternative']} is constant in the data: regularity not applicable"
        for row in rows if np.isnan(row["step"])
    ]
    if any(problem.attributes[a].kind == "time" for a in attributes):
        flags.append("time attributes are an extension; Feng et al. check own cost only")
    if getattr(model, "piecewise_constant", False):
        flags.append("model is piecewise constant: most changes are exactly zero, so weak is close to 1 by construction")
    return AnalysisResult(
        kind="regularity",
        tables={"summary": pd.DataFrame(rows)},
        settings={"step_divisor": step_divisor, "tolerance_divisor": tolerance_divisor,
                  "difference": "forward", "threshold_on": "raw probability change"},
        flags=flags,
    )
