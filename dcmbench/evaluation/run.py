"""``evaluate()``: one call from a model and a dataset to the full multi-dimensional result."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib import metadata
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dcmbench.data.dataset import Dataset
from dcmbench.evaluation.elasticity import point_elasticities
from dcmbench.evaluation.predict import check_deterministic, predict
from dcmbench.evaluation.regularity import behavioral_regularity
from dcmbench.evaluation.scenarios import Scenario, run_scenario
from dcmbench.evaluation.vot import vot_from_marginal_utilities, vot_from_probabilities
from dcmbench.protocols import FittedModel, ProvidesMarginalUtilities
from dcmbench.results import AnalysisResult, EvaluationResult, FitResult, Provenance
from dcmbench.validation.metrics import (
    build_scoring_frame,
    get_performance_metrics,
    likelihood_metrics,
    observation_loglikelihoods,
)
from dcmbench.validation.validator import InternalValidator

ANALYSES = ("elasticities", "vot", "scenarios", "regularity")


@dataclass(frozen=True)
class Holdout:
    """Single train/test split (``sklearn.model_selection.train_test_split``)."""

    test_size: float = 0.2
    seed: int = 42
    stratify: bool = True
    grouped: bool = False

    def __post_init__(self):
        if self.stratify and self.grouped:
            raise ValueError("Holdout cannot both stratify on the choice and keep respondents together")


@dataclass(frozen=True)
class CrossValidation:
    """K-fold cross-validation with re-estimation per fold (``InternalValidator.cross_validate``)."""

    n_splits: int = 5
    seed: int = 0
    grouped: bool = True


@dataclass(frozen=True)
class Bootstrap:
    """Prediction-error bootstrap, OOB and .632 (``InternalValidator.bootstrap``)."""

    n_draws: int = 30
    seed: int = 0
    grouped: bool = True


def _split(protocol: Holdout, data: pd.DataFrame, problem):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    if protocol.grouped:
        if problem.group_column is None:
            raise ValueError("grouped holdout needs ChoiceProblem.group_column")
        splitter = GroupShuffleSplit(n_splits=1, test_size=protocol.test_size, random_state=protocol.seed)
        train_idx, test_idx = next(splitter.split(data, groups=data[problem.group_column]))
        return data.iloc[train_idx], data.iloc[test_idx]
    # Stratify on the alternative's position in the choice problem, so the split
    # does not depend on how a dataset happens to code its choices.
    position = {alt: i for i, alt in enumerate(problem.alternatives)}
    stratify = problem.choices(data).map(position).to_numpy() if protocol.stratify else None
    return train_test_split(data, test_size=protocol.test_size, stratify=stratify, random_state=protocol.seed)


def _fit(model, data: pd.DataFrame, problem) -> FitResult:
    probabilities = predict(model, data, problem)
    frame = build_scoring_frame(data, probabilities, problem.choice_column)
    loglikelihood = float(np.sum(observation_loglikelihoods(frame, problem.choice_column, problem.choice_mapping)))
    return FitResult.from_loglikelihoods(
        n_obs=len(data),
        loglikelihood=loglikelihood,
        null_loglikelihood=float(model.null_loglikelihood(data)),
        n_params=model.n_params,
        converged=bool(model.converged),
    )


def _holdout_prediction(model, test: pd.DataFrame, problem, protocol: Holdout) -> AnalysisResult:
    probabilities = predict(model, test, problem)
    frame = build_scoring_frame(test, probabilities, problem.choice_column)
    metrics = {
        "loglikelihood": float(np.sum(observation_loglikelihoods(frame, problem.choice_column, problem.choice_mapping))),
        **likelihood_metrics(frame, problem.choice_column, problem.choice_mapping,
                             null_ll=float(model.null_loglikelihood(test)), n_params=model.n_params),
        **get_performance_metrics(frame, problem.choice_column, problem.choice_mapping),
        "n_obs": len(test),
    }
    return AnalysisResult(kind="prediction", tables={"summary": pd.DataFrame([metrics])},
                          settings={"protocol": "holdout", **protocol.__dict__})


def _validation_prediction(fitter, data: pd.DataFrame, problem, protocol) -> AnalysisResult:
    validator = InternalValidator(
        fitter,
        choice_column=problem.choice_column,
        choice_mapping=problem.choice_mapping,
        group_column=problem.group_column if protocol.grouped else None,
        random_state=protocol.seed,
        on_error="raise",
    )
    if isinstance(protocol, CrossValidation):
        result = validator.cross_validate(data, n_splits=protocol.n_splits)
        name = "cross_validation"
    else:
        result = validator.bootstrap(data, n_bootstrap=protocol.n_draws)
        name = "bootstrap"
    return AnalysisResult(
        kind="prediction",
        tables={"summary": result.summary(), "detail": result.detail.drop(columns=["fitted"], errors="ignore")},
        settings={"protocol": name, **protocol.__dict__},
    )


def _provenance(dataset: Dataset, model, name: str, seeds: dict) -> Provenance:
    def version(package):
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return "not installed"

    describe = getattr(model, "describe", None)
    return Provenance(
        dataset={"name": dataset.name, **dataset.info.__dict__},
        steps=list(dataset.provenance),
        model={"name": name, "class": type(model).__qualname__, **(describe() if callable(describe) else {})},
        software={"python": platform.python_version(), **{p: version(p) for p in ("dcmbench", "numpy", "pandas", "scikit-learn")}},
        seeds=seeds,
    )


def evaluate(
    model_or_fitter,
    dataset: Dataset,
    *,
    name: str,
    protocol: Optional[object] = None,
    analyses: Sequence[str] = ANALYSES,
    scenarios: Sequence[Scenario] = (),
    time_attribute: str = "time",
    cost_attribute: str = "cost",
    vot_route: str = "auto",
    vot_posterior: bool = False,
    vot_upper_bound: Optional[float] = None,
) -> EvaluationResult:
    """
    Evaluate one model on one dataset along every dimension.

    ``model_or_fitter`` is a fitted model (``FittedModel``) or a fitter
    (``Callable[[DataFrame], FittedModel]``). With ``protocol=None`` the model
    (or ``fitter(full data)``) is evaluated in-sample. ``Holdout`` fits on the
    training part; prediction and behavioral analyses use the test part.
    ``CrossValidation`` and ``Bootstrap`` give prediction metrics via the
    validation module; behavioral analyses then use the full-sample fit
    in-sample.

    ``vot_route``: ``"auto"`` (Route A when the model provides marginal
    utilities, else B), ``"A"`` or ``"B"``. Nothing falls back silently:
    any failure raises.
    """
    unknown = set(analyses) - set(ANALYSES)
    if unknown:
        raise ValueError(f"unknown analyses {sorted(unknown)}; choose from {ANALYSES}")
    dataset.validate()
    data, problem = dataset.frame, dataset.problem
    is_model = isinstance(model_or_fitter, FittedModel)
    if not is_model and not callable(model_or_fitter):
        raise TypeError("pass a fitted model or a fitter (callable returning a fitted model)")
    if is_model and protocol is not None:
        raise ValueError("validation protocols re-estimate the model; pass a fitter, not a fitted model")

    report = {}
    if isinstance(protocol, Holdout):
        train, test = _split(protocol, data, problem)
        model, fit_data, analysis_data = model_or_fitter(train), train, test
        report["prediction"] = _holdout_prediction(model, test, problem, protocol)
        evaluated_on = "holdout test data"
    elif protocol is None or isinstance(protocol, (CrossValidation, Bootstrap)):
        model = model_or_fitter if is_model else model_or_fitter(data)
        fit_data = analysis_data = data
        if protocol is not None:
            report["prediction"] = _validation_prediction(model_or_fitter, data, problem, protocol)
        evaluated_on = "full sample (in-sample)"
    else:
        raise TypeError(f"unknown protocol {protocol!r}; use Holdout, CrossValidation or Bootstrap")

    behavioral = [a for a in analyses if a != "scenarios" or scenarios]
    if getattr(model, "static", False) and behavioral:
        raise ValueError(
            "this model's probabilities are precomputed and cannot respond to changed data; "
            f"run it with analyses=() instead of {tuple(behavioral)}"
        )
    check_deterministic(model, analysis_data, problem)
    fit = _fit(model, fit_data, problem)

    has_attribute = lambda a: a in problem.attributes  # noqa: E731
    if "elasticities" in analyses:
        for attribute in (time_attribute, cost_attribute):
            if has_attribute(attribute):
                report[f"elasticities_{attribute}"] = point_elasticities(model, analysis_data, problem, attribute)
    if "vot" in analyses and has_attribute(time_attribute) and has_attribute(cost_attribute):
        route = vot_route
        if route == "auto":
            route = "A" if isinstance(model, ProvidesMarginalUtilities) else "B"
        if route == "A":
            report["vot"] = vot_from_marginal_utilities(model, analysis_data, problem, time_attribute, cost_attribute,
                                                        posterior=vot_posterior, upper_bound=vot_upper_bound)
        elif route == "B":
            report["vot"] = vot_from_probabilities(model, analysis_data, problem, time_attribute, cost_attribute,
                                                   range_data=data, upper_bound=vot_upper_bound)
        else:
            raise ValueError("vot_route must be 'auto', 'A' or 'B'")
    if "scenarios" in analyses:
        for scenario in scenarios:
            report[f"scenario: {scenario.name}"] = run_scenario(model, analysis_data, problem, scenario)
    if "regularity" in analyses:
        report["regularity"] = behavioral_regularity(model, analysis_data, problem, range_data=data)

    for result in report.values():
        result.settings.setdefault("evaluated_on", evaluated_on if result.kind != "prediction" else result.settings["protocol"])

    seeds = {"protocol": protocol.seed} if protocol is not None else {}
    return EvaluationResult(
        model_name=name,
        problem_name=problem.name,
        provenance=_provenance(dataset, model, name, seeds),
        fit=fit,
        analyses=report,
    )
