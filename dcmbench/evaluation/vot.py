"""
Value-of-time distributions.

Route A uses the model's marginal utilities (``ProvidesMarginalUtilities``):
VOT = (∂V/∂time) / (∂V/∂cost) per observation and draw, weighted by the
model's prior weights (population view) or by posterior weights given the
observed choices (Train 2009, ch. 11).

Route B uses only probabilities: VOT = (∂P_i/∂time_i) / (∂P_i/∂cost_i) by
central differences. For logit models with utility linear in time and cost the
two routes agree exactly. Route B is not meaningful for mixed logit (it
averages over the taste distribution) and is flagged for piecewise-constant
models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from dcmbench.evaluation.perturb import attribute_column, shift
from dcmbench.evaluation.predict import predict
from dcmbench.problem import ChoiceProblem, to_currency_per_hour
from dcmbench.protocols import MarginalUtilities, ProvidesMarginalUtilities
from dcmbench.results import AnalysisResult

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles=QUANTILES) -> np.ndarray:
    """
    Quantiles of a weighted sample: the inverted weighted CDF, averaging the
    two neighbours where the CDF hits a quantile exactly. With integer weights
    this equals the median convention of the sample expanded by the weights.
    """
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights) / weights.sum()
    result = []
    for q in np.atleast_1d(quantiles):
        i = min(int(np.searchsorted(cdf, q - 1e-12, side="left")), len(values) - 1)
        exact = np.isclose(cdf[i], q) and i + 1 < len(values)
        result.append((values[i] + values[i + 1]) / 2 if exact else values[i])
    return np.asarray(result)


def _units(problem: ChoiceProblem, time_attribute: str, cost_attribute: str):
    time, cost = problem.attributes[time_attribute], problem.attributes[cost_attribute]
    if time.kind != "time" or cost.kind != "cost":
        raise ValueError(f"{time_attribute!r} must be a time attribute and {cost_attribute!r} a cost attribute")
    return time.unit, cost.unit


def _summarize(samples: pd.DataFrame, upper_bound: Optional[float]) -> pd.DataFrame:
    """Per-alternative summary; each observation carries total weight 1."""
    rows = []
    for alternative, group in samples.groupby("alternative", sort=False):
        values, weights = group["vot"].to_numpy(), group["weight"].to_numpy()
        mean = np.average(values, weights=weights)
        quantiles = weighted_quantiles(values, weights)
        row = {
            "alternative": alternative,
            "n_obs": int(group["observation"].nunique()),
            "mean": float(mean),
            "sd": float(np.sqrt(np.average((values - mean) ** 2, weights=weights))),
            **{f"q{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, quantiles)},
            "share_negative": float(weights[values < 0].sum() / weights.sum()),
        }
        if upper_bound is not None:
            row["share_above_bound"] = float(weights[values > upper_bound].sum() / weights.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _posterior_weights(mu: MarginalUtilities, data: pd.DataFrame, problem: ChoiceProblem) -> np.ndarray:
    if mu.chosen_prob is None:
        raise ValueError("posterior VOT needs MarginalUtilities.chosen_prob")
    log_likelihood = np.log(np.clip(mu.chosen_prob, 1e-300, None))
    if mu.panel:
        if problem.group_column is None:
            raise ValueError("panel draws need ChoiceProblem.group_column")
        groups = data[problem.group_column].to_numpy()
        per_group = pd.DataFrame(log_likelihood).groupby(groups).transform("sum").to_numpy()
        log_likelihood = per_group
    log_posterior = np.log(mu.weights) + log_likelihood
    log_posterior -= log_posterior.max(axis=1, keepdims=True)
    posterior = np.exp(log_posterior)
    return posterior / posterior.sum(axis=1, keepdims=True)


def vot_from_marginal_utilities(
    model,
    data: pd.DataFrame,
    problem: ChoiceProblem,
    time_attribute: str = "time",
    cost_attribute: str = "cost",
    posterior: bool = False,
    upper_bound: Optional[float] = None,
    keep_samples: bool = False,
) -> AnalysisResult:
    """Route A. ``upper_bound`` (currency/hour) adds the share of VOTs above a plausibility bound."""
    if not isinstance(model, ProvidesMarginalUtilities):
        raise TypeError("Route A needs a model implementing ProvidesMarginalUtilities")
    time_unit, cost_unit = _units(problem, time_attribute, cost_attribute)
    mu = model.marginal_utilities(data, [time_attribute, cost_attribute])
    if list(mu.alternatives) != list(problem.alternatives):
        raise ValueError("MarginalUtilities.alternatives must match the choice problem's alternatives")
    attrs = list(mu.attributes)
    ratio = mu.values[..., attrs.index(time_attribute)] / mu.values[..., attrs.index(cost_attribute)]
    vot, currency = to_currency_per_hour(ratio, time_unit, cost_unit)
    weights = _posterior_weights(mu, data, problem) if posterior else mu.weights

    available = problem.availability_matrix(data).to_numpy()
    n_obs, n_draws, _ = vot.shape
    frames, flags = [], []
    for j, alternative in enumerate(problem.alternatives):
        has_attributes = np.isfinite(mu.values[:, :, j, :]).all(axis=(1, 2))
        keep = available[:, j] & has_attributes & np.isfinite(vot[:, :, j]).all(axis=1)
        undefined = int((available[:, j] & has_attributes & ~keep).sum())
        if undefined:
            flags.append(f"{alternative}: VOT undefined for {undefined} observations (zero marginal utility of cost)")
        if not keep.any():
            continue
        frames.append(pd.DataFrame({
            "observation": np.repeat(data.index[keep], n_draws),
            "draw": np.tile(np.arange(n_draws), int(keep.sum())),
            "alternative": alternative,
            "vot": vot[keep, :, j].ravel(),
            "weight": weights[keep].ravel(),
        }))
    if not frames:
        raise ValueError("no alternative has both attributes with a defined VOT")
    samples = pd.concat(frames, ignore_index=True)
    tables = {"summary": _summarize(samples, upper_bound)}
    if keep_samples:
        tables["samples"] = samples
    route = "A-posterior" if posterior else "A-population"
    return AnalysisResult(
        kind="vot",
        tables=tables,
        settings={"route": route, "currency_per_hour": currency, "time_attribute": time_attribute,
                  "cost_attribute": cost_attribute, "n_draws": n_draws, "panel": mu.panel,
                  "upper_bound": upper_bound},
        flags=flags,
    )


def vot_from_probabilities(
    model,
    data: pd.DataFrame,
    problem: ChoiceProblem,
    time_attribute: str = "time",
    cost_attribute: str = "cost",
    step_fraction: float = 1e-4,
    range_data: Optional[pd.DataFrame] = None,
    upper_bound: Optional[float] = None,
    keep_samples: bool = False,
) -> AnalysisResult:
    """
    Route B, for alternatives with both attributes. Steps are
    ``step_fraction`` × the attribute's range in ``range_data`` (default ``data``).
    VOT is undefined (dropped and counted) where ∂P/∂cost is zero.
    """
    time_unit, cost_unit = _units(problem, time_attribute, cost_attribute)
    range_data = data if range_data is None else range_data
    available = problem.availability_matrix(data)
    alternatives = [a for a in problem.alternatives
                    if a in problem.attributes[time_attribute].columns and a in problem.attributes[cost_attribute].columns]

    def derivative(attribute, alternative):
        column = range_data[attribute_column(problem, attribute, alternative)]
        step = step_fraction * float(column.max() - column.min())
        if step <= 0:
            raise ValueError(f"{attribute} of {alternative} has no range; cannot take derivatives")
        up = predict(model, shift(data, problem, attribute, alternative, step), problem)[alternative]
        down = predict(model, shift(data, problem, attribute, alternative, -step), problem)[alternative]
        return ((up - down) / (2 * step)).to_numpy()

    if not alternatives:
        raise ValueError(f"no alternative has both {time_attribute!r} and {cost_attribute!r}")
    frames, flags = [], []
    for alternative in alternatives:
        d_time, d_cost = derivative(time_attribute, alternative), derivative(cost_attribute, alternative)
        keep = available[alternative].to_numpy() & (np.abs(d_cost) > 1e-12)
        undefined = int((available[alternative].to_numpy() & ~keep).sum())
        if undefined:
            flags.append(f"{alternative}: VOT undefined for {undefined} observations (zero cost derivative)")
        vot, currency = to_currency_per_hour(d_time[keep] / d_cost[keep], time_unit, cost_unit)
        frames.append(pd.DataFrame({
            "observation": data.index[keep], "draw": 0, "alternative": alternative, "vot": vot, "weight": 1.0,
        }))
    samples = pd.concat(frames, ignore_index=True)
    if getattr(model, "piecewise_constant", False):
        flags.append("model is piecewise constant: probability-derivative VOT is unreliable")
    tables = {"summary": _summarize(samples, upper_bound)}
    if keep_samples:
        tables["samples"] = samples
    return AnalysisResult(
        kind="vot",
        tables=tables,
        settings={"route": "B", "currency_per_hour": currency, "time_attribute": time_attribute,
                  "cost_attribute": cost_attribute, "step_fraction": step_fraction, "upper_bound": upper_bound},
        flags=flags,
    )
