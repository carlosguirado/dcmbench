"""Out-of-sample metrics for internal validation (distinct from in-sample benchmarking)."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# {CHOICE code -> probability column name}. Identity {1:1,2:2,3:3} when both
# use the same alternative IDs; use e.g. {"TRAIN": 1, ...} if CHOICE is strings.
ChoiceMapping = Mapping[Any, Any]

PERCENT_METRICS = (
    "pct_correct_predictions",
    "pct_clearly_right",
    "pct_clearly_wrong",
    "pct_unclear",
)

BOOTSTRAP_TRIPLICATE_METRICS = (
    "pct_correct_predictions",
    "fitting_factor",
    "MAE_disaggregate",
    "RMSE_disaggregate",
    "brier",
    "pct_clearly_right",
    "pct_clearly_wrong",
    "pct_unclear",
)

LIKELIHOOD_METRICS = ("avg_ll", "rho_sq", "rho_bar_sq")


def validate_choice_mapping(
    data: pd.DataFrame,
    choice_column: str,
    choice_mapping: ChoiceMapping,
    probability_columns: Optional[Sequence[Any]] = None,
) -> None:
    """Fail fast if choice codes or probability columns are misaligned."""
    if choice_column not in data.columns:
        raise ValueError(f"choice_column '{choice_column}' not found in data")
    if not choice_mapping:
        raise ValueError("choice_mapping must be a non-empty dict")

    observed = set(pd.unique(data[choice_column].dropna()))
    mapped = set(choice_mapping.keys())
    missing = observed - mapped
    if missing:
        raise ValueError(
            f"choice_mapping is missing codes present in data: {sorted(missing, key=str)}"
        )

    if probability_columns is not None:
        prob_cols = set(probability_columns)
        needed = set(choice_mapping.values())
        absent = needed - prob_cols
        if absent:
            raise ValueError(
                f"predict_probabilities output is missing columns for choice_mapping "
                f"values: {sorted(absent, key=str)}. Available: {sorted(prob_cols, key=str)}"
            )


def infer_choice_mapping(
    data: pd.DataFrame,
    choice_column: str,
    probability_columns: Sequence[Any],
) -> Dict[Any, Any]:
    """
    Infer an identity mapping when choice codes match probability column names.

    Raises ValueError if they cannot be aligned uniquely.
    """
    codes = list(pd.unique(data[choice_column].dropna()))
    cols = list(probability_columns)

    # Direct match (same values, possibly different types)
    col_by_str = {str(c): c for c in cols}
    mapping: Dict[Any, Any] = {}
    for code in codes:
        if code in cols:
            mapping[code] = code
        elif str(code) in col_by_str:
            mapping[code] = col_by_str[str(code)]
        else:
            # try int cast both ways
            try:
                as_int = int(code)
                if as_int in cols:
                    mapping[code] = as_int
                    continue
            except (TypeError, ValueError):
                pass
            raise ValueError(
                f"Cannot infer choice_mapping: choice code {code!r} has no matching "
                f"probability column among {cols}. Pass choice_mapping explicitly."
            )
    return mapping


def _chosen_probabilities(
    results: pd.DataFrame,
    choice_column: str,
    choice_mapping: ChoiceMapping,
) -> np.ndarray:
    prob_cols = list(choice_mapping.values())
    prob_matrix = results[prob_cols].to_numpy(dtype=float)
    col_pos = {col: i for i, col in enumerate(prob_cols)}
    indices = np.array([col_pos[choice_mapping[c]] for c in results[choice_column]])
    rows = np.arange(len(results))
    return prob_matrix[rows, indices]


def observation_loglikelihoods(
    results: pd.DataFrame,
    choice_column: str,
    choice_mapping: ChoiceMapping,
) -> np.ndarray:
    """Per-observation log P(chosen) from predicted probabilities."""
    chosen = _chosen_probabilities(results, choice_column, choice_mapping)
    # Avoid log(0)
    chosen = np.clip(chosen, 1e-300, 1.0)
    return np.log(chosen)


def likelihood_metrics(
    results: pd.DataFrame,
    choice_column: str,
    choice_mapping: ChoiceMapping,
    null_ll: float,
    n_params: int,
) -> Dict[str, float]:
    """Compute avg_ll, rho_sq, and rho_bar_sq on an evaluation frame."""
    ll_obs = observation_loglikelihoods(results, choice_column, choice_mapping)
    total_ll = float(np.sum(ll_obs))
    n = len(results)
    if n == 0:
        return {"avg_ll": np.nan, "rho_sq": np.nan, "rho_bar_sq": np.nan}
    avg_ll = total_ll / n
    if null_ll == 0 or not np.isfinite(null_ll):
        rho_sq = np.nan
        rho_bar_sq = np.nan
    else:
        rho_sq = 1.0 - (total_ll / null_ll)
        rho_bar_sq = 1.0 - ((total_ll - n_params) / null_ll)
    return {
        "avg_ll": avg_ll,
        "rho_sq": float(rho_sq),
        "rho_bar_sq": float(rho_bar_sq),
    }


def calculate_prediction_accuracy(
    results: pd.DataFrame,
    choice_column: str,
    choice_mapping: ChoiceMapping,
) -> float:
    """Percent correctly predicted by argmax probability."""
    alt_cols = list(choice_mapping.values())
    reverse = {v: k for k, v in choice_mapping.items()}
    predicted = results[alt_cols].idxmax(axis=1).map(reverse)
    return float((results[choice_column] == predicted).mean() * 100.0)


def calculate_fitting_factor(
    results: pd.DataFrame,
    choice_column: str,
    choice_mapping: ChoiceMapping,
) -> float:
    return float(np.mean(_chosen_probabilities(results, choice_column, choice_mapping)))


def calculate_disaggregate_errors(
    results: pd.DataFrame,
    choice_column: str,
    choice_mapping: ChoiceMapping,
) -> Dict[str, float]:
    errors = _chosen_probabilities(results, choice_column, choice_mapping) - 1.0
    return {
        "MAE_disaggregate": float(np.mean(np.abs(errors))),
        "RMSE_disaggregate": float(np.sqrt(np.mean(errors ** 2))),
    }


def calculate_brier_score(
    results: pd.DataFrame,
    choice_column: str,
    choice_mapping: ChoiceMapping,
) -> float:
    alt_cols = list(choice_mapping.values())
    prob_matrix = results[alt_cols].to_numpy(dtype=float)
    true_matrix = np.zeros_like(prob_matrix)
    col_idx = {alt: i for i, alt in enumerate(alt_cols)}
    rows = np.arange(len(results))
    cols = np.array([col_idx[choice_mapping[c]] for c in results[choice_column]])
    true_matrix[rows, cols] = 1.0
    return float(np.mean((prob_matrix - true_matrix) ** 2))


def calculate_clearness_metrics(
    results: pd.DataFrame,
    choice_column: str,
    choice_mapping: ChoiceMapping,
    threshold: float = 0.5,
) -> Dict[str, float]:
    alt_cols = list(choice_mapping.values())
    n_obs = len(results)
    n_alts = len(alt_cols)
    min_threshold = 1.0 / n_alts if n_alts else 0.0
    if threshold <= min_threshold:
        warnings.warn(
            f"clearness_threshold {threshold} should be considerably larger than "
            f"{min_threshold:.3f}",
            UserWarning,
            stacklevel=2,
        )

    chosen = _chosen_probabilities(results, choice_column, choice_mapping)
    prob_matrix = results[alt_cols].to_numpy(dtype=float)
    col_idx = {alt: i for i, alt in enumerate(alt_cols)}
    chosen_cols = np.array([col_idx[choice_mapping[c]] for c in results[choice_column]])
    max_non_chosen = np.empty(n_obs, dtype=float)
    for i in range(n_obs):
        mask = np.ones(n_alts, dtype=bool)
        mask[chosen_cols[i]] = False
        max_non_chosen[i] = prob_matrix[i, mask].max() if n_alts > 1 else 0.0

    pct_right = float((chosen > threshold).mean() * 100.0)
    pct_wrong = float((max_non_chosen > threshold).mean() * 100.0)
    return {
        "pct_clearly_right": pct_right,
        "pct_clearly_wrong": pct_wrong,
        "pct_unclear": 100.0 - pct_right - pct_wrong,
        "clearness_threshold": threshold,
    }


def get_performance_metrics(
    results: pd.DataFrame,
    choice_column: str,
    choice_mapping: ChoiceMapping,
    clearness_threshold: float = 0.5,
) -> Dict[str, float]:
    """Aggregate prediction metrics on an evaluation DataFrame."""
    metrics = {
        "pct_correct_predictions": calculate_prediction_accuracy(
            results, choice_column, choice_mapping
        ),
        "fitting_factor": calculate_fitting_factor(
            results, choice_column, choice_mapping
        ),
        **calculate_disaggregate_errors(results, choice_column, choice_mapping),
        "brier": calculate_brier_score(results, choice_column, choice_mapping),
        **calculate_clearness_metrics(
            results, choice_column, choice_mapping, clearness_threshold
        ),
    }
    return metrics


def get_performance_metrics_per_observation(
    results: pd.DataFrame,
    choice_column: str,
    choice_mapping: ChoiceMapping,
    clearness_threshold: float = 0.5,
) -> List[Dict[str, float]]:
    """
    Per-row metric components for bootstrap OOB pooling.

    Vectorized (numpy) rather than ``DataFrame.iterrows()``: this is called
    once per bootstrap draw on the OOB set, so row-by-row Series construction
    would dominate runtime for larger datasets/many draws.
    """
    alt_cols = list(choice_mapping.values())
    n_obs = len(results)
    n_alts = len(alt_cols)
    if n_obs == 0:
        return []

    prob_matrix = results[alt_cols].to_numpy(dtype=float)
    col_idx = {alt: i for i, alt in enumerate(alt_cols)}
    rows = np.arange(n_obs)
    chosen_cols = np.array(
        [col_idx[choice_mapping[c]] for c in results[choice_column]]
    )

    p_chosen = prob_matrix[rows, chosen_cols]

    true_matrix = np.zeros_like(prob_matrix)
    true_matrix[rows, chosen_cols] = 1.0
    brier = np.mean((prob_matrix - true_matrix) ** 2, axis=1)

    predicted_cols = np.argmax(prob_matrix, axis=1)
    correct = (predicted_cols == chosen_cols).astype(float)

    err = p_chosen - 1.0

    if n_alts > 1:
        masked = prob_matrix.copy()
        masked[rows, chosen_cols] = -np.inf
        max_non_chosen = masked.max(axis=1)
    else:
        max_non_chosen = np.zeros(n_obs)

    clearly_right = p_chosen > clearness_threshold
    clearly_wrong = max_non_chosen > clearness_threshold
    unclear = (~clearly_right) & (~clearly_wrong)

    return [
        {
            "brier": float(brier[i]),
            "pct_correct_predictions": float(correct[i]),
            "fitting_factor": float(p_chosen[i]),
            "MAE_disaggregate": float(abs(err[i])),
            "RMSE_disaggregate": float(err[i] ** 2),
            "pct_clearly_right": float(clearly_right[i]),
            "pct_clearly_wrong": float(clearly_wrong[i]),
            "pct_unclear": float(unclear[i]),
        }
        for i in range(n_obs)
    ]


def blend_632(
    apparent: Mapping[str, float],
    oob: Mapping[str, float],
    keys: Sequence[str] = BOOTSTRAP_TRIPLICATE_METRICS,
) -> Dict[str, float]:
    """Efron 0.632 blend of apparent and OOB prediction metrics."""
    out: Dict[str, float] = {}
    for key in keys:
        a = apparent.get(key, np.nan)
        o = oob.get(key, np.nan)
        if key == "RMSE_disaggregate":
            out[key] = float(np.sqrt(0.368 * (a ** 2) + 0.632 * (o ** 2)))
        else:
            out[key] = float(0.368 * a + 0.632 * o)
    return out


def build_scoring_frame(
    data: pd.DataFrame,
    probabilities: pd.DataFrame,
    choice_column: str,
) -> pd.DataFrame:
    """Combine evaluation choices with predicted probabilities for metric helpers."""
    if len(data) != len(probabilities):
        raise ValueError(
            f"Length mismatch: data has {len(data)} rows, probabilities has "
            f"{len(probabilities)} rows"
        )
    frame = probabilities.copy()
    frame.index = data.index
    frame[choice_column] = data[choice_column].to_numpy()
    return frame


def equal_share_null_loglikelihood(
    data: pd.DataFrame,
    choice_column: str,
    n_alternatives: Optional[int] = None,
) -> float:
    """
    Fallback null LL: -n * log(J) using the number of unique choices (or n_alternatives).

    Prefer FittedModel.null_loglikelihood when availability-aware null LL is available.
    """
    n = len(data)
    if n_alternatives is None:
        n_alternatives = int(data[choice_column].nunique())
    if n_alternatives < 1:
        return float("nan")
    return float(-n * np.log(n_alternatives))
