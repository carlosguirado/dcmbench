"""
Probabilities computed elsewhere (e.g. by Apollo in R), scored by DCMBench.

The model looks rows up by a key column, so it cannot respond to perturbed
data: ``evaluate()`` refuses behavioral analyses for it (``static = True``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from dcmbench.problem import ChoiceProblem


class PrecomputedModel:
    static = True

    def __init__(
        self,
        probabilities: pd.DataFrame,
        problem: ChoiceProblem,
        key: str,
        n_params: Optional[int] = None,
        converged: bool = True,
        source: str = "precomputed",
    ):
        """
        ``probabilities`` has a ``key`` column identifying observations and one
        column per alternative in ``problem.alternatives``.
        """
        missing = [alt for alt in problem.alternatives if alt not in probabilities.columns]
        if key not in probabilities.columns or missing:
            raise ValueError(f"probabilities need a {key!r} column and columns for alternatives {missing}")
        if probabilities[key].duplicated().any():
            raise ValueError(f"key column {key!r} has duplicates")
        self.table = probabilities.set_index(key)[list(problem.alternatives)]
        self.problem, self.key, self._n_params, self._converged, self.source = problem, key, n_params, converged, source

    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        keys = data[self.key]
        unknown = ~keys.isin(self.table.index)
        if unknown.any():
            raise KeyError(f"{int(unknown.sum())} observations have no precomputed probabilities")
        return self.table.loc[keys.to_numpy()].set_axis(data.index)

    @property
    def n_params(self) -> Optional[int]:
        return self._n_params

    def null_loglikelihood(self, data: pd.DataFrame) -> float:
        return float(-np.log(self.problem.availability_matrix(data).sum(axis=1)).sum())

    @property
    def converged(self) -> bool:
        return self._converged

    def describe(self):
        return {"engine": "precomputed", "source": self.source}
