"""Containers for internal validation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Results from ``cross_validate`` or ``bootstrap``."""

    method: str
    detail: pd.DataFrame
    summary_row: Dict[str, Any] = field(default_factory=dict)
    n_success: int = 0
    n_failed: int = 0
    n_not_converged: int = 0
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def folds(self) -> pd.DataFrame:
        """Alias for CV fold-level detail."""
        return self.detail

    @property
    def iterations(self) -> pd.DataFrame:
        """Alias for bootstrap iteration-level detail."""
        return self.detail

    def summary(self) -> pd.DataFrame:
        """
        Return a one-row summary DataFrame.

        For CV this is mean/std across successful folds (already aggregated into
        ``summary_row``). For bootstrap it holds OOB likelihood means and
        apparent/oob/632 prediction metrics.
        """
        row = {
            "method": self.method,
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "n_not_converged": self.n_not_converged,
            **self.params,
            **self.summary_row,
        }
        return pd.DataFrame([row])


def summarize_numeric_columns(
    detail: pd.DataFrame,
    metric_columns: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Mean and std of numeric metric columns over successful rows."""
    if detail.empty:
        return {}
    success = detail
    if "status" in detail.columns:
        success = detail[detail["status"] == "ok"]
    if success.empty:
        return {}

    if metric_columns is None:
        skip = {
            "method",
            "fold",
            "bootstrap_iter",
            "status",
            "error",
            "converged",
            "n_train",
            "n_eval",
        }
        metric_columns = [
            c
            for c in success.columns
            if c not in skip and pd.api.types.is_numeric_dtype(success[c])
        ]

    out: Dict[str, float] = {}
    for col in metric_columns:
        values = pd.to_numeric(success[col], errors="coerce")
        out[f"{col}_mean"] = float(np.nanmean(values))
        out[f"{col}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return out
