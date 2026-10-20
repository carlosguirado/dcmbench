"""
Result objects and provenance.

Every result can be written to and read back from plain JSON. A metric that
does not apply to a model is recorded as ``None`` with a reason in
``not_applicable``, never as zero.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Step:
    """One preparation step applied to a dataset."""

    action: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    rows_before: Optional[int] = None
    rows_after: Optional[int] = None


@dataclass
class Provenance:
    """Where a result came from: data, its preparation, and the software that produced it."""

    dataset: Dict[str, Any] = field(default_factory=dict)
    steps: List[Step] = field(default_factory=list)
    model: Dict[str, Any] = field(default_factory=dict)
    software: Dict[str, str] = field(default_factory=dict)
    seeds: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Provenance":
        data = dict(data)
        data["steps"] = [Step(**s) for s in data.get("steps", [])]
        return cls(**data)


@dataclass
class FitResult:
    """In-sample statistical fit."""

    n_obs: int
    loglikelihood: float
    null_loglikelihood: Optional[float]
    n_params: Optional[int]
    converged: bool
    rho_sq: Optional[float] = None
    rho_bar_sq: Optional[float] = None
    not_applicable: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_loglikelihoods(
        cls,
        n_obs: int,
        loglikelihood: float,
        null_loglikelihood: Optional[float],
        n_params: Optional[int],
        converged: bool,
    ) -> "FitResult":
        not_applicable = {}
        rho_sq = rho_bar_sq = None
        if null_loglikelihood is None:
            not_applicable["rho_sq"] = not_applicable["rho_bar_sq"] = "no null log-likelihood"
        else:
            rho_sq = 1.0 - loglikelihood / null_loglikelihood
            if n_params is None:
                not_applicable["rho_bar_sq"] = "model has no meaningful parameter count"
            else:
                rho_bar_sq = 1.0 - (loglikelihood - n_params) / null_loglikelihood
        return cls(n_obs, loglikelihood, null_loglikelihood, n_params, converged, rho_sq, rho_bar_sq, not_applicable)


@dataclass
class EvaluationResult:
    """
    Everything DCMBench computed for one model on one dataset.

    Phase 1 holds fit and provenance; prediction and behavioral sections are
    added with the evaluation engine (Phase 2).
    """

    model_name: str
    problem_name: str
    provenance: Provenance
    fit: Optional[FitResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        data = dict(data)
        data["provenance"] = Provenance.from_dict(data["provenance"])
        if data.get("fit") is not None:
            data["fit"] = FitResult(**data["fit"])
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "EvaluationResult":
        return cls.from_dict(json.loads(text))
