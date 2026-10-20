"""The single boundary through which DCMBench obtains probabilities from a model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dcmbench.problem import ChoiceProblem
from dcmbench.protocols import ProbabilityTableError, check_probabilities, mask_unavailable


class NonDeterministicModelError(RuntimeError):
    """A fitted model returned different probabilities for identical data."""


def predict(model, data: pd.DataFrame, problem: ChoiceProblem) -> pd.DataFrame:
    """
    Probabilities from ``model`` on ``data``, checked against the contract.

    Models that set ``handles_availability = False`` (e.g. scikit-learn
    classifiers) are masked and renormalized here; every other model must
    already put zero mass on unavailable alternatives.
    """
    probabilities = model.predict_probabilities(data)
    if not isinstance(probabilities, pd.DataFrame):
        raise ProbabilityTableError(
            f"predict_probabilities must return a DataFrame, not {type(probabilities).__name__}"
        )
    if getattr(model, "handles_availability", True) is False:
        probabilities = mask_unavailable(probabilities, problem, data)
    check_probabilities(probabilities, problem, data)
    return probabilities.set_axis(data.index)


def check_deterministic(model, data: pd.DataFrame, problem: ChoiceProblem, n_rows: int = 50) -> None:
    """
    Raise unless two predictions on the same rows are identical.

    Finite-difference analyses compare predictions on nearly identical data;
    simulated models must reuse their draws (common random numbers).
    """
    sample = data.iloc[:n_rows]
    first = predict(model, sample, problem).to_numpy()
    second = predict(model, sample, problem).to_numpy()
    if not np.array_equal(first, second):
        raise NonDeterministicModelError(
            "predict_probabilities returned different values for identical data. "
            "Simulated models must reuse the same draws on every call."
        )
