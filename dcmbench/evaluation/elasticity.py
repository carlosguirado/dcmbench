"""Point elasticities of choice probabilities, from probabilities alone."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dcmbench.evaluation.perturb import attribute_column, shift
from dcmbench.evaluation.predict import predict
from dcmbench.problem import ChoiceProblem
from dcmbench.results import AnalysisResult

PIECEWISE_FLAG = (
    "model is piecewise constant: point elasticities from small steps are mostly zero; "
    "use arc elasticities from a policy scenario instead"
)


def point_elasticities(
    model,
    data: pd.DataFrame,
    problem: ChoiceProblem,
    attribute: str,
    relative_step: float = 1e-4,
    keep_individual: bool = False,
) -> AnalysisResult:
    """
    E_nij = ∂P_ni/∂x_nj · x_nj / P_ni by central differences with step
    ``relative_step · |x_nj|``, for every alternative j with the attribute
    (own: i = j, cross: i ≠ j).

    Observations where i or j is unavailable are excluded. Where x_nj = 0 the
    elasticity is 0 by definition. The aggregate elasticity is the
    probability-weighted mean Σ_n P_ni E_nij / Σ_n P_ni (Train 2009, 3.6).
    """
    base = predict(model, data, problem)
    available = problem.availability_matrix(data)
    rows, summary = [], []
    n_zero_x = {}

    for changed in problem.attributes[attribute].columns:
        x = data[attribute_column(problem, attribute, changed)].to_numpy(dtype=float)
        step = relative_step * np.abs(x)
        nonzero = step > 0
        n_zero_x[changed] = int((~nonzero).sum())
        up = predict(model, shift(data, problem, attribute, changed, step), problem).to_numpy()
        down = predict(model, shift(data, problem, attribute, changed, -step), problem).to_numpy()
        derivative = np.divide(up - down, 2 * step[:, None], out=np.zeros_like(up), where=nonzero[:, None])

        for i, responding in enumerate(problem.alternatives):
            p = base[responding].to_numpy()
            keep = available[changed].to_numpy() & available[responding].to_numpy() & (p > 0)
            elasticity = np.divide(derivative[:, i] * x, p, out=np.zeros_like(p), where=p > 0)
            weights = p[keep]
            summary.append({
                "attribute": attribute,
                "changed": changed,
                "responding": responding,
                "type": "own" if responding == changed else "cross",
                "aggregate": float(np.sum(weights * elasticity[keep]) / np.sum(weights)) if keep.any() else np.nan,
                "mean_individual": float(elasticity[keep].mean()) if keep.any() else np.nan,
                "n_obs": int(keep.sum()),
            })
            if keep_individual:
                rows.append(pd.DataFrame({
                    "observation": data.index[keep],
                    "changed": changed,
                    "responding": responding,
                    "elasticity": elasticity[keep],
                }))

    tables = {"summary": pd.DataFrame(summary)}
    if keep_individual:
        tables["individual"] = pd.concat(rows, ignore_index=True)
    flags = [PIECEWISE_FLAG] if getattr(model, "piecewise_constant", False) else []
    flags += [f"{n} observations with {attribute} = 0 for {alt} (elasticity 0 by definition)"
              for alt, n in n_zero_x.items() if n]
    return AnalysisResult(
        kind="elasticities",
        tables=tables,
        settings={"attribute": attribute, "method": "point, central difference", "relative_step": relative_step},
        flags=flags,
    )
