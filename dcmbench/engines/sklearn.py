"""
Any scikit-learn classifier with ``predict_proba`` as a choice model.

Classifiers know nothing about availability, so the model declares
``handles_availability = False`` and DCMBench masks and renormalizes its
probabilities. There is no meaningful parameter count (ρ̄² is not applicable).
Tree, ensemble and nearest-neighbour models are piecewise constant in their
inputs, which derivative-based analyses flag.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dcmbench.problem import ChoiceProblem

_PIECEWISE_MODULES = ("sklearn.tree", "sklearn.ensemble", "sklearn.neighbors")


def is_piecewise_constant(estimator) -> bool:
    module = type(estimator).__module__
    return any(module.startswith(prefix) for prefix in _PIECEWISE_MODULES)


class SklearnModel:
    handles_availability = False

    def __init__(self, estimator, problem: ChoiceProblem, features: Sequence[str], piecewise_constant: bool):
        self.estimator, self.problem, self.features = estimator, problem, list(features)
        self.piecewise_constant = piecewise_constant

    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        raw = self.estimator.predict_proba(data[self.features])
        probabilities = pd.DataFrame(raw, columns=list(self.estimator.classes_), index=data.index)
        return probabilities.reindex(columns=list(self.problem.alternatives), fill_value=0.0)

    @property
    def n_params(self) -> None:
        return None

    def null_loglikelihood(self, data: pd.DataFrame) -> float:
        return float(-np.log(self.problem.availability_matrix(data).sum(axis=1)).sum())

    @property
    def converged(self) -> bool:
        return True

    def describe(self):
        return {"engine": "scikit-learn", "estimator": repr(self.estimator), "features": self.features,
                "piecewise_constant": self.piecewise_constant}


class SklearnFitter:
    """
    Fitter: ``SklearnFitter(estimator, problem, features)(train) -> SklearnModel``.

    The estimator is cloned for every fit (one per CV fold). ``features`` are
    the input columns; include every attribute the analyses perturb.
    """

    def __init__(self, estimator, problem: ChoiceProblem, features: Sequence[str],
                 piecewise_constant: Optional[bool] = None):
        if not hasattr(estimator, "predict_proba"):
            raise TypeError("the estimator must implement predict_proba")
        if not features:
            raise ValueError("give the feature columns explicitly")
        self.estimator, self.problem, self.features = estimator, problem, list(features)
        self.piecewise_constant = is_piecewise_constant(estimator) if piecewise_constant is None else piecewise_constant

    def __call__(self, train: pd.DataFrame) -> SklearnModel:
        from sklearn.base import clone

        estimator = clone(self.estimator)
        estimator.fit(train[self.features], self.problem.choices(train))
        return SklearnModel(estimator, self.problem, self.features, self.piecewise_constant)
