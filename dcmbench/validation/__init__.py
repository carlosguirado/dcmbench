"""
Internal validation for discrete choice models.

Provides k-fold cross-validation and prediction-error bootstrap (OOB + 0.632).
In-sample estimation metrics remain with ``SimpleBenchmarker``.
"""

from dcmbench.validation.estimators import (
    BiogemeSpecFitter,
    FunctionModelFitter,
    fitter_from_spec,
)
from dcmbench.validation.results import ValidationResult
from dcmbench.validation.validator import InternalValidator, bootstrap, cross_validate

__all__ = [
    "InternalValidator",
    "ValidationResult",
    "BiogemeSpecFitter",
    "FunctionModelFitter",
    "fitter_from_spec",
    "cross_validate",
    "bootstrap",
]
