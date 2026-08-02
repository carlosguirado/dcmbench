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
        return bool(results.algorithm_has_converged())
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
    """FittedModel wrapper around UniversalBiogemeAdapter + Biogeme results."""

    def __init__(
        self,
        adapter: Any,
        results: Any,
        availabilities: Optional[Dict[Any, Any]] = None,
        utilities: Optional[Dict[Any, Any]] = None,
        choice_column: str = "CHOICE",
    ):
        self._adapter = adapter
        self._results = results
        self._availabilities = availabilities
        self._utilities = utilities
        self.choice_column = choice_column

    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._adapter.predict_probabilities(data)

    @property
    def n_params(self) -> int:
        return _default_n_params(self._results)

    @property
    def converged(self) -> bool:
        return _default_converged(self._results)

    def null_loglikelihood(self, data: pd.DataFrame) -> float:
        """Availability-aware null LL when possible; else equal-share fallback."""
        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme import models
        from biogeme.expressions import Variable

        from dcmbench.validation.metrics import equal_share_null_loglikelihood

        n_alts = None
        if self._utilities is not None:
            n_alts = len(self._utilities)

        if self._availabilities is None or self._utilities is None:
            return equal_share_null_loglikelihood(
                data, self.choice_column, n_alternatives=n_alts
            )

        try:
            database = db.Database("null_ll", data.copy())
            choice = Variable(self.choice_column)
            logprob = models.loglogit(
                self._utilities, self._availabilities, choice
            )
            biogeme = bio.BIOGEME(database, logprob)
            for attr, value in (
                ("generate_html", False),
                ("generate_pickle", False),
                ("saveIterations", False),
            ):
                if hasattr(biogeme, attr):
                    setattr(biogeme, attr, value)
            return float(
                biogeme.calculate_null_loglikelihood(self._availabilities)
            )
        except Exception as exc:
            warnings.warn(
                f"Availability-aware null log-likelihood failed ({type(exc).__name__}: "
                f"{exc}); falling back to an equal-share null LL that ignores "
                "per-observation availability. This changes the baseline used for "
                "rho_sq/rho_bar_sq on this fold/draw and may not be comparable to "
                "folds where the availability-aware null LL succeeded.",
                UserWarning,
                stacklevel=2,
            )
            return equal_share_null_loglikelihood(
                data, self.choice_column, n_alternatives=n_alts
            )


class BiogemeSpecFitter:
    """
    Fit a Biogeme model from a JSON specification on each training subset.

    Usage::

        fitter = BiogemeSpecFitter(spec, choice_column="CHOICE")
        fitted = fitter(train_df)
        probs = fitted.predict_probabilities(test_df)
    """

    def __init__(
        self,
        spec: Dict[str, Any],
        choice_column: str = "CHOICE",
        database_name: str = "validation_train",
        quiet: bool = True,
    ):
        model_type = str(spec.get("metadata", {}).get("model_type", "")).upper()
        if model_type == "MXL":
            random_params = spec.get("random_parameters", {})
            if random_params:
                raise NotImplementedError(
                    "BiogemeSpecFitter/fitter_from_spec cannot build mixed logit "
                    "(MXL) models with random taste parameters: "
                    "dcmbench.model_specifications.build_model_from_spec does not "
                    "wire 'random_parameters' into the utility formulas (no "
                    "bioDraws are created), so Biogeme raises 'The argument of "
                    "MonteCarlo must contain a bioDraws' as soon as the model is "
                    "constructed. Cross-validation/bootstrap of a real MXL model "
                    "currently requires building the Biogeme model yourself (with "
                    "explicit bioDraws) and wrapping it in a custom ModelFitter, "
                    "rather than using fitter_from_spec."
                )
            warnings.warn(
                "Model spec declares model_type='MXL' but has no "
                "'random_parameters'; build_model_from_spec will silently "
                "estimate a plain MNL for this spec. Set random_parameters or "
                "use a custom ModelFitter for genuine mixed logit.",
                UserWarning,
                stacklevel=2,
            )
        self.spec = spec
        self.choice_column = choice_column
        self.database_name = database_name
        self.quiet = quiet

    def __call__(self, train_df: pd.DataFrame) -> BiogemeFittedModel:
        import biogeme.database as db

        from dcmbench.adapters.universal_biogeme_adapter import UniversalBiogemeAdapter
        from dcmbench.model_specifications import build_model_from_spec

        database = db.Database(self.database_name, train_df.copy())
        model = build_model_from_spec(self.spec, database)

        # Expose structure under the names UniversalBiogemeAdapter prefers.
        if hasattr(model, "_utilities"):
            model._dcmbench_utilities = model._utilities
        if hasattr(model, "_availabilities"):
            model._dcmbench_availability = model._availabilities

        if self.quiet:
            for attr, value in (
                ("generate_html", False),
                ("generate_pickle", False),
                ("saveIterations", False),
            ):
                if hasattr(model, attr):
                    setattr(model, attr, value)

        results = model.estimate()
        adapter = UniversalBiogemeAdapter(
            model, results, database=database, name=getattr(model, "modelName", "model")
        )
        return BiogemeFittedModel(
            adapter=adapter,
            results=results,
            availabilities=getattr(model, "_availabilities", None),
            utilities=getattr(model, "_utilities", None),
            choice_column=self.choice_column,
        )


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
    ModelFitter built from user-supplied build/predict functions.

    Use this for models the JSON-spec path (``BiogemeSpecFitter``/
    ``fitter_from_spec``) cannot build --- most importantly **mixed logit
    (MXL)** with genuine random taste parameters. ``build_model_from_spec``
    only substitutes plain (non-random) ``Beta`` objects into utility
    formulas, so it cannot express a ``bioDraws``-based random coefficient;
    a hand-written Biogeme model is required instead. This mirrors the
    ``build_*``/``predict_*`` function-pair pattern used for MXL models in
    the ``dcm-internal-validation`` experiments repo.

    ``InternalValidator`` itself does not care how a ``FittedModel`` is
    produced -- it only calls ``fitter(train_df)`` and expects back an
    object with ``predict_probabilities``, ``n_params``, ``converged``, and
    ``null_loglikelihood``. ``FunctionModelFitter`` is just a thin adapter
    that packages plain functions into that protocol.

    Parameters
    ----------
    build_fn : callable(train_df) -> results
        Estimate the model on the training subset (e.g. construct a Biogeme
        model with explicit ``bioDraws`` and call ``.estimate()``). Called
        once per fold/bootstrap draw. ``results`` can be any object; it is
        passed through unchanged to the other callables.
    predict_fn : callable(results, data) -> pd.DataFrame
        Predict choice probabilities on ``data`` given ``results`` (e.g.
        rebuild the same random-coefficient expression, wrap it in
        ``MonteCarlo``, and call ``BIOGEME(...).simulate(results.get_beta_values())``
        on a database built from ``data``). Must return columns resolvable
        via ``choice_mapping``.
    null_loglikelihood_fn : callable(results, data) -> float, optional
        Availability-aware null LL on ``data`` (e.g.
        ``biogeme.calculate_null_loglikelihood(av)`` on a fresh BIOGEME
        object built from ``data``). If omitted, falls back to an
        equal-share null LL with a warning.
    n_params_fn : callable(results) -> int, optional
        Defaults to ``results.data.nparam`` / ``len(results.get_beta_values())``.
    converged_fn : callable(results) -> bool, optional
        Defaults to Biogeme's usual convergence-flag locations, else ``True``.
    choice_column : str

    Examples
    --------
    See ``tutorials/internal_validation_mxl_swissmetro.py`` for a full,
    runnable mixed-logit example (build/predict functions with explicit
    ``bioDraws``, plugged into ``InternalValidator.cross_validate``).
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
