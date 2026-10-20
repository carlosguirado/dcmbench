"""The fitted model shared by every engine that estimates a spec."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from dcmbench.problem import ChoiceProblem
from dcmbench.protocols import MarginalUtilities
from dcmbench.specs import compute
from dcmbench.specs.backends import NumpyBackend
from dcmbench.specs.expression import derivative, evaluate
from dcmbench.specs.draws import coefficient
from dcmbench.specs.spec import Spec

NUMPY = NumpyBackend()


class SpecModel:
    """
    A spec with estimated parameters.

    Engines (Biogeme, PyTorch) only estimate. Predictions, log-likelihoods,
    marginal utilities and posterior inputs are computed here from the spec,
    so every engine's estimates are evaluated identically.

    ``values`` holds every parameter (estimated and fixed); ``std_errors``
    holds robust standard errors of the estimated ones.
    """

    handles_availability = True

    def __init__(
        self,
        spec: Spec,
        problem: ChoiceProblem,
        values: Mapping[str, float],
        std_errors: Optional[Mapping[str, float]] = None,
        converged: bool = True,
        engine: str = "unknown",
        engine_details: Optional[Mapping[str, Any]] = None,
    ):
        compute.check_compatible(spec, problem)
        missing = set(spec.parameters) - set(values)
        if missing:
            raise ValueError(f"values missing for parameters {sorted(missing)}")
        self.spec, self.problem = spec, problem
        self.values = {name: float(values[name]) for name in spec.parameters}
        self.std_errors = dict(std_errors or {})
        self._converged = bool(converged)
        self.engine = engine
        self.engine_details = dict(engine_details or {})

    # --- FittedModel ------------------------------------------------------------------------------

    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        per_draw = self.probabilities_per_draw(data)
        return pd.DataFrame(per_draw.mean(axis=1), columns=list(self.problem.alternatives), index=data.index)

    @property
    def n_params(self) -> int:
        return self.spec.n_free_parameters

    def null_loglikelihood(self, data: pd.DataFrame) -> float:
        """Equal shares over each observation's available alternatives."""
        return float(-np.log(self.problem.availability_matrix(data).sum(axis=1)).sum())

    @property
    def converged(self) -> bool:
        return self._converged

    # --- extras -----------------------------------------------------------------------------------

    def probabilities_per_draw(self, data: pd.DataFrame) -> np.ndarray:
        inputs = compute.Inputs(self.spec, self.problem, data, NUMPY)
        return compute.probabilities_per_draw(self.spec, self.problem, inputs.columns, inputs.n_obs, self.values,
                                              inputs.draws, inputs.available, NUMPY)

    def loglikelihood(self, data: pd.DataFrame) -> float:
        inputs = compute.Inputs(self.spec, self.problem, data, NUMPY)
        per_draw = compute.probabilities_per_draw(self.spec, self.problem, inputs.columns, inputs.n_obs, self.values,
                                                  inputs.draws, inputs.available, NUMPY)
        return float(compute.unit_loglikelihoods(per_draw, inputs.chosen, inputs.units, inputs.n_units,
                                                 inputs.panel, NUMPY).sum())

    def coefficients(self) -> pd.DataFrame:
        rows = []
        for name, parameter in self.spec.parameters.items():
            se = self.std_errors.get(name, np.nan)
            value = self.values[name]
            rows.append({
                "name": name,
                "estimate": value,
                "robust_std_err": np.nan if parameter.fixed else se,
                "t_stat": np.nan if parameter.fixed or not se else value / se,
                "fixed": parameter.fixed,
                "at_bound": (parameter.lower is not None and np.isclose(value, parameter.lower))
                or (parameter.upper is not None and np.isclose(value, parameter.upper)),
            })
        return pd.DataFrame(rows)

    def marginal_utilities(self, data: pd.DataFrame, attributes: Sequence[str]) -> MarginalUtilities:
        """∂V_j/∂x_j for each alternative's column of each attribute, per draw (exact, symbolic)."""
        inputs = compute.Inputs(self.spec, self.problem, data, NUMPY)
        env: Dict[str, Any] = {c: data[c].to_numpy(dtype=float)[:, None] for c in data.columns
                               if pd.api.types.is_numeric_dtype(data[c])}
        env.update(self.values)
        for name, rp in self.spec.random_parameters.items():
            env[name] = coefficient(rp, self.values[rp.mean], self.values[rp.spread], inputs.draws[name], NUMPY)
        n_draws = next(iter(inputs.draws.values())).shape[1] if inputs.draws else 1

        values = np.full((len(data), n_draws, len(self.problem.alternatives), len(attributes)), np.nan)
        for k, attribute in enumerate(attributes):
            for j, alternative in enumerate(self.problem.alternatives):
                column = self.problem.attributes[attribute].columns.get(alternative)
                if column is None:
                    continue
                d = evaluate(derivative(self.spec.parsed[alternative], column), env, NUMPY)
                values[:, :, j, k] = np.broadcast_to(np.asarray(d, dtype=float), (len(data), n_draws))

        chosen_prob = None
        if inputs.chosen is not None:
            per_draw = compute.probabilities_per_draw(self.spec, self.problem, inputs.columns, inputs.n_obs,
                                                      self.values, inputs.draws, inputs.available, NUMPY)
            chosen_prob = per_draw[np.arange(len(data)), :, inputs.chosen]
        return MarginalUtilities(
            values=values,
            weights=np.full((len(data), n_draws), 1.0 / n_draws),
            alternatives=self.problem.alternatives,
            attributes=list(attributes),
            chosen_prob=chosen_prob,
            panel=inputs.panel,
        )

    def describe(self) -> Dict[str, Any]:
        return {"spec": self.spec.name, "spec_sha256": self.spec.sha256(), "model_type": self.spec.model,
                "engine": self.engine, **self.engine_details}

    # Nested-logit introspection kept for validation-module callers.
    @property
    def is_nested(self) -> bool:
        return self.spec.model == "NL"

    @property
    def nests(self):
        return self.spec.nests if self.spec.nests else None
