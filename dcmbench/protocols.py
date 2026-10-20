"""
Model protocols.

The core protocol is ``dcmbench.validation.protocols.FittedModel``: a fitted
model only has to return choice probabilities. Everything else DCMBench
computes itself. Models may also implement the optional capability protocols
below, which are detected at runtime and never required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from dcmbench.problem import ChoiceProblem
from dcmbench.validation.protocols import FittedModel, ModelFitter

__all__ = [
    "FittedModel",
    "ModelFitter",
    "ProvidesCoefficients",
    "ProvidesMarginalUtilities",
    "MarginalUtilities",
    "ProbabilityTableError",
    "check_probabilities",
    "mask_unavailable",
]


@runtime_checkable
class ProvidesCoefficients(Protocol):
    def coefficients(self) -> pd.DataFrame:
        """One row per parameter: estimate, robust standard error, whether fixed."""
        ...


@dataclass
class MarginalUtilities:
    """
    ∂V_ij/∂x_ij for each observation, draw, alternative and attribute, in data units.

    A "draw" is whatever the model's taste heterogeneity is made of: one for
    MNL, K classes for latent class, R simulation draws for mixed logit.
    ``weights`` are prior weights per observation (rows sum to 1);
    ``chosen_prob`` (P(observed choice | draw)) lets DCMBench compute
    posterior weights. ``panel=True`` means draw r is the same person-level
    draw for all of a respondent's observations, so posteriors multiply
    ``chosen_prob`` within ``ChoiceProblem.group_column``. NaN marks an
    alternative without the attribute.
    """

    values: np.ndarray
    weights: np.ndarray
    alternatives: Sequence[str]
    attributes: Sequence[str]
    chosen_prob: Optional[np.ndarray] = None
    panel: bool = False

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.values.ndim != 4:
            raise ValueError("values must have shape (n_obs, n_draws, n_alternatives, n_attributes)")
        n_obs, n_draws, n_alts, n_attrs = self.values.shape
        if (n_alts, n_attrs) != (len(self.alternatives), len(self.attributes)):
            raise ValueError("values shape does not match alternatives and attributes")
        if self.weights.shape != (n_obs, n_draws):
            raise ValueError(f"weights must have shape {(n_obs, n_draws)}, not {self.weights.shape}")
        if not np.allclose(self.weights.sum(axis=1), 1.0):
            raise ValueError("weights must sum to 1 for every observation")
        if self.chosen_prob is not None:
            self.chosen_prob = np.asarray(self.chosen_prob, dtype=float)
            if self.chosen_prob.shape != (n_obs, n_draws):
                raise ValueError(f"chosen_prob must have shape {(n_obs, n_draws)}")


@runtime_checkable
class ProvidesMarginalUtilities(Protocol):
    def marginal_utilities(self, data: pd.DataFrame, attributes: Sequence[str]) -> MarginalUtilities:
        """Marginal utilities of the named ``ChoiceProblem`` attributes on ``data``."""
        ...


class ProbabilityTableError(ValueError):
    """A model returned a probability table that violates the contract."""


def check_probabilities(
    probabilities: pd.DataFrame,
    problem: ChoiceProblem,
    data: pd.DataFrame,
    atol: float = 1e-8,
) -> None:
    """
    Raise ``ProbabilityTableError`` unless ``probabilities`` has one column per
    alternative (in ``problem.alternatives`` order), one row per observation,
    finite values in [0, 1] summing to 1, and zero mass on unavailable alternatives.
    """
    if list(probabilities.columns) != list(problem.alternatives):
        raise ProbabilityTableError(
            f"probability columns {list(probabilities.columns)} must equal "
            f"alternatives {list(problem.alternatives)}, in that order"
        )
    if len(probabilities) != len(data):
        raise ProbabilityTableError(f"{len(probabilities)} probability rows for {len(data)} observations")

    values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ProbabilityTableError("probabilities contain NaN or infinite values")
    if (values < -atol).any() or (values > 1 + atol).any():
        raise ProbabilityTableError("probabilities must lie in [0, 1]")
    bad_rows = np.flatnonzero(np.abs(values.sum(axis=1) - 1.0) > 1e-6)
    if bad_rows.size:
        raise ProbabilityTableError(f"{bad_rows.size} rows do not sum to 1 (first: row {bad_rows[0]})")

    unavailable = ~problem.availability_matrix(data).to_numpy()
    leaked = values[unavailable]
    if leaked.size and (leaked > atol).any():
        raise ProbabilityTableError(
            f"{int((leaked > atol).sum())} probabilities on unavailable alternatives exceed {atol}"
        )


def mask_unavailable(
    probabilities: pd.DataFrame,
    problem: ChoiceProblem,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Zero unavailable alternatives and renormalize each row.

    For engines that cannot know about availability (e.g. scikit-learn
    classifiers). Raises if a row has no probability mass left.
    """
    probabilities = probabilities.reindex(columns=list(problem.alternatives), fill_value=0.0)
    available = problem.availability_matrix(data).to_numpy()
    values = np.where(available, probabilities.to_numpy(dtype=float), 0.0)
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ProbabilityTableError(
            f"{int((totals <= 0).sum())} observations have zero probability on every available alternative"
        )
    return pd.DataFrame(values / totals, index=probabilities.index, columns=probabilities.columns)
