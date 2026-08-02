"""Protocols for internal validation fitters and fitted models."""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FittedModel(Protocol):
    """Model fitted on a training subset and ready for out-of-sample scoring."""

    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of choice probabilities (one column per alternative)."""
        ...

    @property
    def n_params(self) -> int:
        """Number of estimated parameters."""
        ...

    def null_loglikelihood(self, data: pd.DataFrame) -> float:
        """Null-model log-likelihood on ``data`` (same sample used for OOS scoring)."""
        ...

    @property
    def converged(self) -> bool:
        """Whether estimation reported convergence."""
        ...


# Callable that estimates on a training DataFrame and returns a FittedModel
ModelFitter = Callable[[pd.DataFrame], FittedModel]
