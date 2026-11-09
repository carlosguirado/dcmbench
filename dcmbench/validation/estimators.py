"""Biogeme helpers that turn a JSON model spec into a per-fold fitter."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _default_converged(results: Any) -> bool:
    """Best-effort convergence check shared by all Biogeme-backed FittedModels."""
    if hasattr(results, "data") and hasattr(results.data, "convergence"):
        flag = results.data.convergence
        if flag is not None:
            return bool(flag)
    if hasattr(results, "convergence"):
        flag = results.convergence
        if flag is not None:
            return bool(flag)
    if hasattr(results, "algorithm_has_converged"):
        flag = results.algorithm_has_converged
        # A method before Biogeme 3.3, a property from 3.3 on.
        return bool(flag() if callable(flag) else flag)
    # Successful estimate() without an explicit flag: treat as converged.
    return True


def _default_n_params(results: Any) -> int:
    """Best-effort parameter count shared by all Biogeme-backed FittedModels."""
    if hasattr(results, "data") and hasattr(results.data, "nparam"):
        return int(results.data.nparam)
    if hasattr(results, "get_beta_values"):
        return len(results.get_beta_values())
    raise AttributeError(
        "Cannot infer n_params from results; pass n_params_fn=... explicitly."
    )


class BiogemeFittedModel:
    """
    FittedModel for a v1 spec estimated with Biogeme.

    Wraps a ``dcmbench.specs.SpecModel``; probability columns are the v1
    alternative ids (e.g. 1, 2, 3).
    """

    def __init__(self, model: Any, choice_column: str = "CHOICE"):
        self._adapter = model  # the SpecModel; kept under this name for existing callers
        self.choice_column = choice_column
        self._columns = {alt: (int(alt) if alt.isdigit() else alt) for alt in model.problem.alternatives}

    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._adapter.predict_probabilities(data).rename(columns=self._columns)

    @property
    def n_params(self) -> int:
        return self._adapter.n_params

    @property
    def converged(self) -> bool:
        return self._adapter.converged

    def null_loglikelihood(self, data: pd.DataFrame) -> float:
        """Availability-aware equal-share null LL."""
        return self._adapter.null_loglikelihood(data)


class BiogemeSpecFitter:
    """
    Fit a Biogeme model from a v1 JSON specification on each training subset.

    Usage::

        fitter = BiogemeSpecFitter(spec, choice_column="CHOICE")
        fitted = fitter(train_df)
        probs = fitted.predict_probabilities(test_df)

    For new work, write a spec v2 and use ``dcmbench.engines.biogeme.BiogemeEstimator``.
    """

    def __init__(
        self,
        spec: Dict[str, Any],
        choice_column: str = "CHOICE",
        database_name: str = "validation_train",
        quiet: bool = True,
    ):
        from dcmbench.specs import v1

        v1.check_supported(spec)
        self.spec = spec
        self.choice_column = choice_column
        self.database_name = database_name
        self.quiet = quiet

    def __call__(self, train_df: pd.DataFrame) -> BiogemeFittedModel:
        from dcmbench.engines.biogeme import BiogemeEstimator
        from dcmbench.specs import v1

        spec, problem = v1.convert(self.spec, choice_column=self.choice_column)
        model = BiogemeEstimator(spec, problem)(train_df)
        return BiogemeFittedModel(model, choice_column=self.choice_column)


def fitter_from_spec(
    spec: Dict[str, Any],
    choice_column: str = "CHOICE",
    **kwargs: Any,
) -> BiogemeSpecFitter:
    """Return a callable fitter built from a Biogeme JSON model specification."""
    return BiogemeSpecFitter(spec, choice_column=choice_column, **kwargs)


class _FunctionFittedModel:
    """FittedModel wrapper around user-supplied build/predict/null-LL callables."""

    def __init__(
        self,
        results: Any,
        predict_fn: Callable[[Any, pd.DataFrame], pd.DataFrame],
        null_loglikelihood_fn: Optional[Callable[[Any, pd.DataFrame], float]],
        n_params_fn: Optional[Callable[[Any], int]],
        converged_fn: Optional[Callable[[Any], bool]],
        choice_column: str,
    ):
        self._results = results
        self._predict_fn = predict_fn
        self._null_loglikelihood_fn = null_loglikelihood_fn
        self._n_params_fn = n_params_fn
        self._converged_fn = converged_fn
        self.choice_column = choice_column

    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._predict_fn(self._results, data)

    @property
    def n_params(self) -> int:
        if self._n_params_fn is not None:
            return int(self._n_params_fn(self._results))
        return _default_n_params(self._results)

    @property
    def converged(self) -> bool:
        if self._converged_fn is not None:
            return bool(self._converged_fn(self._results))
        return _default_converged(self._results)

    def null_loglikelihood(self, data: pd.DataFrame) -> float:
        if self._null_loglikelihood_fn is not None:
            return float(self._null_loglikelihood_fn(self._results, data))
        from dcmbench.validation.metrics import equal_share_null_loglikelihood

        warnings.warn(
            "No null_loglikelihood_fn supplied to FunctionModelFitter; using an "
            "equal-share null LL that ignores per-observation availability. Pass "
            "null_loglikelihood_fn=... (e.g. via biogeme.calculate_null_loglikelihood) "
            "for an availability-aware baseline.",
            UserWarning,
            stacklevel=2,
        )
        return equal_share_null_loglikelihood(data, self.choice_column)


class FunctionModelFitter:
    """
    ModelFitter from user-supplied build/predict callables.

    Use when ``fitter_from_spec`` cannot build the model (notably MXL with
    ``bioDraws``). See ``tutorials/internal_validation_mxl_swissmetro.py``.

    Parameters
    ----------
    build_fn : callable(train_df) -> results
    predict_fn : callable(results, data) -> DataFrame of probabilities
    null_loglikelihood_fn : callable(results, data) -> float, optional
    n_params_fn : callable(results) -> int, optional
    converged_fn : callable(results) -> bool, optional
    choice_column : str
    """

    def __init__(
        self,
        build_fn: Callable[[pd.DataFrame], Any],
        predict_fn: Callable[[Any, pd.DataFrame], pd.DataFrame],
        null_loglikelihood_fn: Optional[Callable[[Any, pd.DataFrame], float]] = None,
        n_params_fn: Optional[Callable[[Any], int]] = None,
        converged_fn: Optional[Callable[[Any], bool]] = None,
        choice_column: str = "CHOICE",
    ):
        self.build_fn = build_fn
        self.predict_fn = predict_fn
        self.null_loglikelihood_fn = null_loglikelihood_fn
        self.n_params_fn = n_params_fn
        self.converged_fn = converged_fn
        self.choice_column = choice_column

    def __call__(self, train_df: pd.DataFrame) -> _FunctionFittedModel:
        results = self.build_fn(train_df)
        return _FunctionFittedModel(
            results=results,
            predict_fn=self.predict_fn,
            null_loglikelihood_fn=self.null_loglikelihood_fn,
            n_params_fn=self.n_params_fn,
            converged_fn=self.converged_fn,
            choice_column=self.choice_column,
        )
