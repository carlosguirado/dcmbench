"""Unit tests for dcmbench.results."""

import numpy as np
import pytest

from dcmbench.results import EvaluationResult, FitResult, Provenance, Step


def test_fit_result_reports_not_applicable_rho_bar():
    fit = FitResult.from_loglikelihoods(n_obs=100, loglikelihood=-60.0, null_loglikelihood=-100.0,
                                        n_params=None, converged=True)
    assert fit.rho_sq == pytest.approx(0.4)
    assert fit.rho_bar_sq is None
    assert "rho_bar_sq" in fit.not_applicable


def test_fit_result_with_parameter_count():
    fit = FitResult.from_loglikelihoods(100, -60.0, -100.0, n_params=5, converged=True)
    assert fit.rho_bar_sq == pytest.approx(1 - 65 / 100)
    assert fit.not_applicable == {}


def test_evaluation_result_json_round_trip():
    result = EvaluationResult(
        model_name="mnl",
        problem_name="swissmetro",
        provenance=Provenance(
            dataset={"name": "swissmetro", "sha256": "abc"},
            steps=[Step("filter", "commuters", {"query": "PURPOSE == 1"}, 10, 5)],
            software={"dcmbench": "1.0.0"},
            seeds={"split": 42},
        ),
        fit=FitResult.from_loglikelihoods(5, -3.0, -5.0, n_params=None, converged=True),
    )
    restored = EvaluationResult.from_json(result.to_json())
    assert restored == result
    assert isinstance(restored.provenance.steps[0], Step)
    assert not np.isnan(restored.fit.rho_sq)
