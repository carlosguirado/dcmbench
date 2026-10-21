"""
Utilities, choice probabilities and log-likelihoods of a spec, on any backend.

Conventions (identical to Biogeme's):

- Unavailable alternatives get utility ``MASK`` (a large negative number,
  not ``-inf``, so torch gradients stay finite).
- Nested logit: nest parameter μ_m ≥ 1 scales utilities inside the nest;
  P(j) = exp(μ_m V_j) / Σ_{k∈m} exp(μ_m V_k) · exp(L_m) / Σ_n exp(L_n), with
  L_m = log Σ_{k∈m} exp(μ_m V_k) / μ_m. Alternatives in no nest form their own.
- Mixed logit: probabilities average logit over draws; with panel draws, a
  respondent's likelihood is the average over draws of the product over
  their choices.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from dcmbench.problem import ChoiceProblem
from dcmbench.specs.draws import coefficient, variates
from dcmbench.specs.expression import evaluate
from dcmbench.specs.spec import Spec, SpecError

MASK = -1e100


def check_compatible(spec: Spec, problem: ChoiceProblem) -> None:
    if set(spec.alternatives) != set(problem.alternatives):
        raise SpecError(
            f"spec alternatives {sorted(spec.alternatives)} differ from the choice problem's "
            f"{sorted(problem.alternatives)}"
        )
    if spec.draws is not None and spec.draws.panel and problem.group_column is None:
        raise SpecError("panel draws need ChoiceProblem.group_column")


def draw_units(spec: Spec, problem: ChoiceProblem, data: pd.DataFrame):
    """Unit (respondent or observation) index per row, and the number of units."""
    if spec.draws is not None and spec.draws.panel:
        codes, uniques = pd.factorize(data[problem.group_column], sort=True)
        return codes, len(uniques)
    return np.arange(len(data)), len(data)


def random_draws(spec: Spec, problem: ChoiceProblem, data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """``(n_obs, R)`` variates per random parameter, mapped from their units."""
    if not spec.random_parameters:
        return {}
    units, n_units = draw_units(spec, problem, data)
    return {
        name: variates(rp, n_units, spec.draws, index)[units]
        for index, (name, rp) in enumerate(spec.random_parameters.items())
    }


def utilities(spec: Spec, problem: ChoiceProblem, data: Mapping[str, object], n_obs: int,
              values: Mapping[str, object], draws: Mapping[str, object], backend):
    """
    ``(n_obs, R, J)`` utilities, alternatives in ``problem.alternatives`` order.
    ``data`` maps column names to ``(n_obs, 1)`` arrays; ``values`` maps parameter
    names to scalars; ``draws`` maps random parameters to ``(n_obs, R)`` variates.
    """
    env = dict(data)
    env.update(values)
    for name, rp in spec.random_parameters.items():
        env[name] = coefficient(rp, values[rp.mean], values[rp.spread], draws[name], backend)
    n_draws = next(iter(draws.values())).shape[1] if draws else 1
    columns = [backend.broadcast(evaluate(spec.parsed[alt], env, backend), (n_obs, n_draws))
               for alt in problem.alternatives]
    return backend.stack(columns, axis=2)


def _log_logit(v, available, backend):
    v = backend.where(available, v, MASK)
    shifted = v - backend.detach(backend.amax(v, axis=2))
    return shifted - backend.log(backend.total(backend.exp(shifted), axis=2))


def _log_nested(v, available, spec: Spec, problem: ChoiceProblem, values, backend):
    v = backend.where(available, v, MASK)
    position = {alt: j for j, alt in enumerate(problem.alternatives)}
    groups = [(values[n.parameter], [position[a] for a in n.alternatives]) for n in spec.nests]
    nested = {j for _, members in groups for j in members}
    groups += [(1.0, [j]) for j in range(len(problem.alternatives)) if j not in nested]

    log_sums, within = [], {}
    for mu, members in groups:
        scaled = backend.stack([mu * v[..., j] for j in members], axis=2)
        peak = backend.detach(backend.amax(scaled, axis=2))
        log_sum = backend.log(backend.total(backend.exp(scaled - peak), axis=2)) + peak  # (n, R, 1)
        log_sums.append(log_sum / mu)
        for k, j in enumerate(members):
            within[j] = (scaled[..., k:k + 1] - log_sum, len(log_sums) - 1)
    top = backend.stack([ls[..., 0] for ls in log_sums], axis=2)
    top = top - backend.detach(backend.amax(top, axis=2))
    log_top = top - backend.log(backend.total(backend.exp(top), axis=2))
    return backend.stack([within[j][0][..., 0] + log_top[..., within[j][1]] for j in range(len(problem.alternatives))],
                         axis=2)


def log_probabilities_per_draw(spec, problem, data, n_obs, values, draws, available, backend):
    """
    ``(n_obs, R, J)`` log choice probabilities (R = 1 without random
    parameters). Computed in log space, so extreme utilities never underflow
    to log(0); unavailable alternatives get a log-probability of about ``MASK``.
    """
    v = utilities(spec, problem, data, n_obs, values, draws, backend)
    if spec.model == "NL":
        return _log_nested(v, available, spec, problem, values, backend)
    return _log_logit(v, available, backend)


def probabilities_per_draw(spec, problem, data, n_obs, values, draws, available, backend):
    """``(n_obs, R, J)`` choice probabilities; exactly 0 for unavailable alternatives."""
    log_p = log_probabilities_per_draw(spec, problem, data, n_obs, values, draws, available, backend)
    return backend.where(available, backend.exp(log_p), 0.0)


def _log_mean_exp(x, backend):
    """log of the mean over axis 1, stably."""
    peak = backend.detach(backend.amax(x, axis=1))
    return backend.log(backend.total(backend.exp(x - peak), axis=1)[:, 0] / x.shape[1]) + peak[:, 0]


def unit_loglikelihoods(log_per_draw, chosen, units, n_units, panel, backend):
    """
    Log-likelihood per unit from ``(n_obs, R, J)`` log-probabilities: per
    observation, or per respondent when ``panel``. Entirely in log space.
    """
    rows = backend.index(np.arange(log_per_draw.shape[0]))
    log_chosen = log_per_draw[rows, :, backend.index(chosen)]  # (n_obs, R)
    if log_per_draw.shape[1] == 1:
        return log_chosen[:, 0]
    if not panel:
        return _log_mean_exp(log_chosen, backend)
    return _log_mean_exp(_group_sum(log_chosen, units, n_units, backend), backend)


def _group_sum(x, units, n_units, backend):
    if backend.name == "numpy":
        out = np.zeros((n_units, x.shape[1]))
        np.add.at(out, units, x)
        return out
    torch = backend.torch
    out = torch.zeros((n_units, x.shape[1]), dtype=x.dtype)
    return out.index_add(0, torch.as_tensor(units, dtype=torch.long), x)


class Inputs:
    """Data prepared once for repeated evaluation (e.g. during estimation)."""

    def __init__(self, spec: Spec, problem: ChoiceProblem, data: pd.DataFrame, backend,
                 draws: Optional[Mapping[str, np.ndarray]] = None):
        check_compatible(spec, problem)
        missing = sorted(set(spec.data_columns()) - set(data.columns))
        if missing:
            raise SpecError(f"data is missing columns used by the spec: {missing}")
        self.n_obs = len(data)
        self.columns = {c: backend.asarray(data[c].to_numpy(dtype=float)[:, None]) for c in spec.data_columns()}
        draws = random_draws(spec, problem, data) if draws is None else draws
        self.draws = {name: backend.asarray(d) for name, d in draws.items()}
        self.available = problem.availability_matrix(data).to_numpy()[:, None, :]
        self.units, self.n_units = draw_units(spec, problem, data)
        self.panel = bool(spec.draws is not None and spec.draws.panel)
        self.chosen = None
        if problem.choice_column in data.columns:
            position = {alt: j for j, alt in enumerate(problem.alternatives)}
            self.chosen = problem.choices(data).map(position).to_numpy()
