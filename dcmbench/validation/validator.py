"""Orchestration for k-fold CV and prediction-error bootstrap."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from dcmbench.validation.metrics import (
    BOOTSTRAP_TRIPLICATE_METRICS,
    PERCENT_METRICS,
    blend_632,
    build_scoring_frame,
    get_performance_metrics,
    get_performance_metrics_per_observation,
    infer_choice_mapping,
    likelihood_metrics,
    validate_choice_mapping,
)
from dcmbench.validation.protocols import FittedModel, ModelFitter
from dcmbench.validation.results import ValidationResult, summarize_numeric_columns
from dcmbench.validation.splitting import iter_bootstrap_indices, iter_kfold_indices


class InternalValidator:
    """
    Re-estimate a model on resampled subsets and score held-out fit.

    Parameters
    ----------
    fitter : callable
        ``fitter(train_df) -> FittedModel``. Use ``fitter_from_spec`` for Biogeme
        JSON specs, or supply a custom callable.
    choice_column : str
        Name of the chosen-alternative column.
    choice_mapping : dict, optional
        Maps choice codes to probability column names. If omitted, inferred once
        from a warm-up fit (identity when codes match columns).
    group_column : str, optional
        If set, folds/bootstrap draws keep all rows for a group together.
    random_state : int, optional
        Seed for reproducible splits.
    clearness_threshold : float
        Threshold for clearness metrics.
    on_error : {'skip', 'raise'}
        How to handle *exceptions* raised during estimation/prediction (e.g. a
        singular Hessian, bad data). Does NOT affect non-convergence: a fold or
        bootstrap draw whose fitter runs to completion but reports
        ``fitted.converged is False`` (e.g. the optimizer hit max iterations)
        is always recorded as ``status="not_converged"`` and skipped, even
        when ``on_error="raise"``. Check ``result.n_not_converged`` /
        ``result.detail`` to catch this.
    verbose : bool
        Print progress per fold/draw.
    """

    def __init__(
        self,
        fitter: ModelFitter,
        choice_column: str = "CHOICE",
        choice_mapping: Optional[Mapping[Any, Any]] = None,
        group_column: Optional[str] = None,
        random_state: Optional[int] = None,
        clearness_threshold: float = 0.5,
        on_error: str = "skip",
        verbose: bool = False,
    ):
        if on_error not in {"skip", "raise"}:
            raise ValueError("on_error must be 'skip' or 'raise'")
        self.fitter = fitter
        self.choice_column = choice_column
        self.choice_mapping = dict(choice_mapping) if choice_mapping is not None else None
        self.group_column = group_column
        self.random_state = random_state
        self.clearness_threshold = clearness_threshold
        self.on_error = on_error
        self.verbose = verbose

    def _resolve_choice_mapping(self, data: pd.DataFrame) -> Dict[Any, Any]:
        if self.choice_mapping is not None:
            validate_choice_mapping(data, self.choice_column, self.choice_mapping)
            return dict(self.choice_mapping)

        # Warm-up fit on a small subset to discover probability column names.
        # This is a full extra estimation (discarded afterwards); pass
        # choice_mapping explicitly to skip it.
        warnings.warn(
            "choice_mapping not provided: running an extra warm-up estimation on "
            "a data subsample to infer it. This is discarded and not counted as "
            "a fold/draw. Pass choice_mapping explicitly to skip this and make "
            "runs reproducible/deterministic regardless of random_state.",
            UserWarning,
            stacklevel=2,
        )
        if self.verbose:
            print("Inferring choice_mapping via a warm-up fit...")
        sample = data
        if len(data) > 200:
            sample = data.sample(n=200, random_state=self.random_state or 0)
        fitted = self.fitter(sample)
        probs = fitted.predict_probabilities(sample.head(min(20, len(sample))))
        mapping = infer_choice_mapping(data, self.choice_column, list(probs.columns))
        validate_choice_mapping(data, self.choice_column, mapping, list(probs.columns))
        self.choice_mapping = mapping
        return mapping

    def _score(
        self,
        fitted: FittedModel,
        eval_df: pd.DataFrame,
        choice_mapping: Mapping[Any, Any],
    ) -> Dict[str, float]:
        probs = fitted.predict_probabilities(eval_df)
        validate_choice_mapping(
            eval_df, self.choice_column, choice_mapping, list(probs.columns)
        )
        frame = build_scoring_frame(eval_df, probs, self.choice_column)
        null_ll = fitted.null_loglikelihood(eval_df)
        metrics = likelihood_metrics(
            frame,
            self.choice_column,
            choice_mapping,
            null_ll=null_ll,
            n_params=fitted.n_params,
        )
        metrics.update(
            get_performance_metrics(
                frame,
                self.choice_column,
                choice_mapping,
                clearness_threshold=self.clearness_threshold,
            )
        )
        return metrics

    def _fit_and_handle(
        self,
        train_df: pd.DataFrame,
        context: str,
    ) -> Union[FittedModel, Dict[str, Any]]:
        try:
            fitted = self.fitter(train_df)
        except Exception as exc:
            if self.on_error == "raise":
                raise
            return {
                "status": "error",
                "converged": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        if not getattr(fitted, "converged", True):
            return {
                "status": "not_converged",
                "converged": False,
                "error": "Estimation reported convergence=False",
                "fitted": fitted,
            }
        return fitted

    def cross_validate(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
    ) -> ValidationResult:
        """
        K-fold cross-validation with re-estimation on each training fold.

        Each fold is a full maximum-likelihood estimation. Prefer small
        ``n_splits`` while testing; use ``group_column`` for panel data.
        """
        if self.group_column is not None and self.group_column not in data.columns:
            raise ValueError(f"group_column '{self.group_column}' not found in data")

        choice_mapping = self._resolve_choice_mapping(data)
        rows: List[Dict[str, Any]] = []

        for fold, (train_idx, test_idx) in enumerate(
            iter_kfold_indices(
                data,
                n_splits=n_splits,
                random_state=self.random_state,
                group_column=self.group_column,
            )
        ):
            if self.verbose:
                print(f"CV fold {fold + 1}/{n_splits}...")
            train_df = data.iloc[train_idx]
            test_df = data.iloc[test_idx]
            base = {
                "method": "cv",
                "fold": fold,
                "n_train": len(train_df),
                "n_eval": len(test_df),
            }

            outcome = self._fit_and_handle(train_df, context=f"fold {fold}")
            if isinstance(outcome, dict) and "fitted" not in outcome:
                rows.append({**base, **outcome})
                continue
            if isinstance(outcome, dict):
                # not_converged with fitted object — still skip scoring
                rows.append(
                    {
                        **base,
                        "status": outcome["status"],
                        "converged": False,
                        "error": outcome.get("error"),
                    }
                )
                continue

            fitted = outcome
            try:
                metrics = self._score(fitted, test_df, choice_mapping)
                rows.append(
                    {
                        **base,
                        "status": "ok",
                        "converged": True,
                        "error": None,
                        **metrics,
                    }
                )
            except Exception as exc:
                if self.on_error == "raise":
                    raise
                rows.append(
                    {
                        **base,
                        "status": "error",
                        "converged": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        detail = pd.DataFrame(rows)
        n_success = int((detail["status"] == "ok").sum()) if not detail.empty else 0
        n_failed = int((detail["status"] == "error").sum()) if not detail.empty else 0
        n_not_converged = (
            int((detail["status"] == "not_converged").sum()) if not detail.empty else 0
        )
        if n_failed or n_not_converged:
            warnings.warn(
                f"cross_validate: {n_success} ok, {n_not_converged} not converged, "
                f"{n_failed} errors out of {len(detail)} folds",
                UserWarning,
                stacklevel=2,
            )

        summary_row = summarize_numeric_columns(detail)
        # Also expose plain means under original metric names for convenience
        ok = detail[detail["status"] == "ok"] if not detail.empty else detail
        for col in list(LIKELIHOOD_AND_PRED):
            if col in ok.columns:
                summary_row[col] = float(np.nanmean(ok[col]))

        return ValidationResult(
            method="cv",
            detail=detail,
            summary_row=summary_row,
            n_success=n_success,
            n_failed=n_failed,
            n_not_converged=n_not_converged,
            params={
                "n_splits": n_splits,
                "group_column": self.group_column,
                "random_state": self.random_state,
                "clearness_threshold": self.clearness_threshold,
            },
        )

    def bootstrap(
        self,
        data: pd.DataFrame,
        n_bootstrap: int = 30,
    ) -> ValidationResult:
        """
        Prediction-error bootstrap with OOB scoring and 0.632 prediction metrics.

        Likelihood metrics (``avg_ll``, ``rho_sq``, ``rho_bar_sq``) are OOB means.
        Prediction metrics are reported as apparent, OOB, and 0.632 blends.

        Each bootstrap draw is a full estimation — this can be expensive for
        mixed logit.
        """
        if self.group_column is not None and self.group_column not in data.columns:
            raise ValueError(f"group_column '{self.group_column}' not found in data")

        choice_mapping = self._resolve_choice_mapping(data)
        n = len(data)
        rows: List[Dict[str, Any]] = []
        apparent_list: List[Dict[str, float]] = []
        oob_metrics_per_obs: Dict[Any, List[Dict[str, float]]] = {}
        ll_rows: List[Dict[str, float]] = []

        draw_iter = list(
            iter_bootstrap_indices(
                data,
                n_bootstrap=n_bootstrap,
                random_state=self.random_state,
                group_column=self.group_column,
            )
        )
        if len(draw_iter) < n_bootstrap:
            warnings.warn(
                f"Requested n_bootstrap={n_bootstrap} draws but only "
                f"{len(draw_iter)} had a non-empty OOB set and were produced "
                "(draws with empty OOB sets are skipped, most likely because "
                "n_bootstrap or the group count is small). Results are based on "
                f"{len(draw_iter)} draws; see ValidationResult.params['n_draws_produced'].",
                UserWarning,
                stacklevel=2,
            )

        for b, (train_idx, oob_idx) in enumerate(draw_iter):
            if self.verbose:
                print(f"Bootstrap draw {b + 1}/{len(draw_iter)}...")
            train_df = data.iloc[train_idx]
            oob_df = data.iloc[oob_idx]
            base = {
                "method": "bootstrap",
                "bootstrap_iter": b,
                "n_train": len(train_df),
                "n_eval": len(oob_df),
            }

            outcome = self._fit_and_handle(train_df, context=f"bootstrap {b}")
            if isinstance(outcome, dict) and "fitted" not in outcome:
                rows.append({**base, **outcome})
                continue
            if isinstance(outcome, dict):
                rows.append(
                    {
                        **base,
                        "status": outcome["status"],
                        "converged": False,
                        "error": outcome.get("error"),
                    }
                )
                continue

            fitted = outcome
            try:
                # OOB likelihood + prediction metrics
                oob_probs = fitted.predict_probabilities(oob_df)
                oob_frame = build_scoring_frame(oob_df, oob_probs, self.choice_column)
                null_ll = fitted.null_loglikelihood(oob_df)
                ll = likelihood_metrics(
                    oob_frame,
                    self.choice_column,
                    choice_mapping,
                    null_ll=null_ll,
                    n_params=fitted.n_params,
                )
                oob_perf = get_performance_metrics(
                    oob_frame,
                    self.choice_column,
                    choice_mapping,
                    clearness_threshold=self.clearness_threshold,
                )

                # Apparent (bootstrap sample) for 0.632
                app_probs = fitted.predict_probabilities(train_df)
                app_frame = build_scoring_frame(train_df, app_probs, self.choice_column)
                app_perf = get_performance_metrics(
                    app_frame,
                    self.choice_column,
                    choice_mapping,
                    clearness_threshold=self.clearness_threshold,
                )
                apparent_list.append(app_perf)

                per_obs = get_performance_metrics_per_observation(
                    oob_frame,
                    self.choice_column,
                    choice_mapping,
                    clearness_threshold=self.clearness_threshold,
                )
                for obs_pos, metrics_obs in zip(oob_idx, per_obs):
                    # Use original index label when available
                    key = data.index[obs_pos]
                    oob_metrics_per_obs.setdefault(key, []).append(metrics_obs)

                row = {
                    **base,
                    "status": "ok",
                    "converged": True,
                    "error": None,
                    **ll,
                    **{f"{k}_oob_draw": v for k, v in oob_perf.items()},
                }
                rows.append(row)
                ll_rows.append(ll)
            except Exception as exc:
                if self.on_error == "raise":
                    raise
                rows.append(
                    {
                        **base,
                        "status": "error",
                        "converged": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        detail = pd.DataFrame(rows)
        n_success = int((detail["status"] == "ok").sum()) if not detail.empty else 0
        n_failed = int((detail["status"] == "error").sum()) if not detail.empty else 0
        n_not_converged = (
            int((detail["status"] == "not_converged").sum()) if not detail.empty else 0
        )
        if n_failed or n_not_converged:
            warnings.warn(
                f"bootstrap: {n_success} ok, {n_not_converged} not converged, "
                f"{n_failed} errors out of {len(detail)} draws",
                UserWarning,
                stacklevel=2,
            )

        summary_row: Dict[str, Any] = {}
        if ll_rows:
            for key in ("avg_ll", "rho_sq", "rho_bar_sq"):
                vals = [r[key] for r in ll_rows]
                summary_row[key] = float(np.nanmean(vals))
                summary_row[f"{key}_std"] = (
                    float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
                )

        # Aggregate OOB prediction metrics (per-observation then mean)
        aggregated_oob: Dict[str, float] = {}
        for key in BOOTSTRAP_TRIPLICATE_METRICS:
            per_obs_aggregate = []
            for metrics_list in oob_metrics_per_obs.values():
                values = [m[key] for m in metrics_list if key in m]
                if not values:
                    continue
                mean_val = float(np.mean(values))
                if key in PERCENT_METRICS:
                    mean_val *= 100.0
                per_obs_aggregate.append(mean_val)
            if key == "RMSE_disaggregate":
                aggregated_oob[key] = (
                    float(np.sqrt(np.mean(per_obs_aggregate)))
                    if per_obs_aggregate
                    else float("nan")
                )
            else:
                aggregated_oob[key] = (
                    float(np.mean(per_obs_aggregate))
                    if per_obs_aggregate
                    else float("nan")
                )

        apparent_agg: Dict[str, float] = {}
        for key in BOOTSTRAP_TRIPLICATE_METRICS:
            values = [m.get(key, np.nan) for m in apparent_list]
            apparent_agg[key] = float(np.nanmean(values)) if values else float("nan")

        metrics_632 = blend_632(apparent_agg, aggregated_oob)

        for key in BOOTSTRAP_TRIPLICATE_METRICS:
            summary_row[f"{key}_apparent"] = apparent_agg.get(key, np.nan)
            summary_row[f"{key}_oob"] = aggregated_oob.get(key, np.nan)
            summary_row[f"{key}_632"] = metrics_632.get(key, np.nan)

        summary_row["clearness_threshold"] = self.clearness_threshold
        summary_row.update(summarize_numeric_columns(detail))

        return ValidationResult(
            method="bootstrap",
            detail=detail,
            summary_row=summary_row,
            n_success=n_success,
            n_failed=n_failed,
            n_not_converged=n_not_converged,
            params={
                "n_bootstrap": n_bootstrap,
                "n_draws_produced": len(draw_iter),
                "group_column": self.group_column,
                "random_state": self.random_state,
                "clearness_threshold": self.clearness_threshold,
            },
        )


# Columns copied as plain means into CV summary
LIKELIHOOD_AND_PRED = [
    "avg_ll",
    "rho_sq",
    "rho_bar_sq",
    "pct_correct_predictions",
    "fitting_factor",
    "MAE_disaggregate",
    "RMSE_disaggregate",
    "brier",
    "pct_clearly_right",
    "pct_clearly_wrong",
    "pct_unclear",
]


def cross_validate(
    fitter: ModelFitter,
    data: pd.DataFrame,
    n_splits: int = 5,
    **validator_kwargs: Any,
) -> ValidationResult:
    """Module-level shortcut for :meth:`InternalValidator.cross_validate`."""
    return InternalValidator(fitter, **validator_kwargs).cross_validate(
        data, n_splits=n_splits
    )


def bootstrap(
    fitter: ModelFitter,
    data: pd.DataFrame,
    n_bootstrap: int = 30,
    **validator_kwargs: Any,
) -> ValidationResult:
    """Module-level shortcut for :meth:`InternalValidator.bootstrap`."""
    return InternalValidator(fitter, **validator_kwargs).bootstrap(
        data, n_bootstrap=n_bootstrap
    )
