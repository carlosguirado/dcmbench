"""
Result objects and provenance.

Every result can be written to and read back from plain JSON. A metric that
does not apply to a model is recorded as ``None`` with a reason in
``not_applicable``, never as zero.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


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
    notes: List[str] = field(default_factory=list)

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


def _frame_to_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    records = frame.to_dict(orient="records")
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in record.items()}
        for record in records
    ]


@dataclass(eq=False)
class AnalysisResult:
    """
    One analysis (prediction, elasticities, VOT, scenario, regularity).

    ``tables`` are tidy DataFrames (e.g. ``"summary"``, ``"individual"``);
    ``settings`` record every choice that affects the numbers (steps,
    thresholds, route, evaluation data); ``flags`` are human-readable caveats
    and deviations that must travel with the numbers.
    """

    kind: str
    tables: Dict[str, pd.DataFrame]
    settings: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    @property
    def summary(self) -> pd.DataFrame:
        return self.tables["summary"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "tables": {name: _frame_to_records(frame) for name, frame in self.tables.items()},
            "settings": self.settings,
            "flags": list(self.flags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        tables = {name: pd.DataFrame.from_records(rows) for name, rows in data["tables"].items()}
        return cls(data["kind"], tables, dict(data.get("settings", {})), list(data.get("flags", [])))

    def __eq__(self, other) -> bool:
        if not isinstance(other, AnalysisResult) or self.tables.keys() != other.tables.keys():
            return False
        return (
            self.kind == other.kind
            and self.settings == other.settings
            and self.flags == other.flags
            and all(_frames_equal(self.tables[k], other.tables[k]) for k in self.tables)
        )


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
    except AssertionError:
        return False
    return True


@dataclass
class EvaluationResult:
    """Everything DCMBench computed for one model on one dataset."""

    model_name: str
    problem_name: str
    provenance: Provenance
    fit: Optional[FitResult] = None
    analyses: Dict[str, AnalysisResult] = field(default_factory=dict)

    def __getitem__(self, name: str) -> AnalysisResult:
        return self.analyses[name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "problem_name": self.problem_name,
            "provenance": asdict(self.provenance),
            "fit": None if self.fit is None else asdict(self.fit),
            "analyses": {name: result.to_dict() for name, result in self.analyses.items()},
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, default=_json_default, **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        return cls(
            model_name=data["model_name"],
            problem_name=data["problem_name"],
            provenance=Provenance.from_dict(data["provenance"]),
            fit=None if data.get("fit") is None else FitResult(**data["fit"]),
            analyses={name: AnalysisResult.from_dict(r) for name, r in data.get("analyses", {}).items()},
        )

    @classmethod
    def from_json(cls, text: str) -> "EvaluationResult":
        return cls.from_dict(json.loads(text))


def _json_default(value):
    """numpy scalars and similar."""
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
