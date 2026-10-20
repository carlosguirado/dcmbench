"""Unit tests for dcmbench.protocols and the n_params=None path through validation."""

import numpy as np
import pandas as pd
import pytest

from dcmbench.problem import ChoiceProblem
from dcmbench.protocols import (
    FittedModel,
    MarginalUtilities,
    ProbabilityTableError,
    ProvidesMarginalUtilities,
    check_probabilities,
    mask_unavailable,
)
from dcmbench.validation.metrics import likelihood_metrics
from dcmbench.validation.validator import InternalValidator

PROBLEM = ChoiceProblem(
    name="toy",
    alternatives=("a", "b", "c"),
    choice_column="CHOICE",
    choice_codes={1: "a", 2: "b", 3: "c"},
    availability={"c": "C_AV"},
)
DATA = pd.DataFrame({"CHOICE": [1, 2, 1], "C_AV": [1, 1, 0]})


def _probs(rows):
    return pd.DataFrame(rows, columns=list(PROBLEM.alternatives))


def test_check_probabilities_accepts_valid_table():
    check_probabilities(_probs([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8], [0.6, 0.4, 0.0]]), PROBLEM, DATA)


@pytest.mark.parametrize(
    "table, message",
    [
        (pd.DataFrame([[0.3, 0.2, 0.5]] * 3, columns=["b", "a", "c"]), "in that order"),
        (_probs([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]), "probability rows"),
        (_probs([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8], [0.6, 0.4, np.nan]]), "NaN"),
        (_probs([[0.2, 0.3, 0.6], [0.1, 0.1, 0.8], [0.6, 0.4, 0.0]]), "do not sum to 1"),
        (_probs([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8], [0.5, 0.4, 0.1]]), "unavailable"),
    ],
)
def test_check_probabilities_rejects_contract_violations(table, message):
    with pytest.raises(ProbabilityTableError, match=message):
        check_probabilities(table, PROBLEM, DATA)


def test_mask_unavailable_zeroes_and_renormalizes():
    masked = mask_unavailable(_probs([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8], [0.25, 0.25, 0.5]]), PROBLEM, DATA)
    np.testing.assert_allclose(masked.iloc[2].to_numpy(), [0.5, 0.5, 0.0])
    np.testing.assert_allclose(masked.iloc[0].to_numpy(), [0.2, 0.3, 0.5])
    check_probabilities(masked, PROBLEM, DATA)


def test_mask_unavailable_raises_when_no_mass_left():
    with pytest.raises(ProbabilityTableError, match="zero probability"):
        mask_unavailable(_probs([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8], [0.0, 0.0, 1.0]]), PROBLEM, DATA)


def test_marginal_utilities_shape_checks():
    mu = MarginalUtilities(
        values=np.zeros((4, 2, 3, 2)),
        weights=np.full((4, 2), 0.5),
        alternatives=PROBLEM.alternatives,
        attributes=("time", "cost"),
    )
    assert mu.values.shape == (4, 2, 3, 2)
    with pytest.raises(ValueError, match="sum to 1"):
        MarginalUtilities(np.zeros((4, 2, 3, 2)), np.ones((4, 2)), PROBLEM.alternatives, ("time", "cost"))
    with pytest.raises(ValueError, match="does not match"):
        MarginalUtilities(np.zeros((4, 2, 3, 1)), np.full((4, 2), 0.5), PROBLEM.alternatives, ("time", "cost"))


class _ConstantModel:
    """Fitted model predicting the training shares; no meaningful parameter count."""

    def __init__(self, shares):
        self.shares = shares

    def predict_probabilities(self, data):
        return pd.DataFrame([self.shares] * len(data), columns=list(PROBLEM.alternatives), index=data.index)

    @property
    def n_params(self):
        return None

    def null_loglikelihood(self, data):
        return len(data) * np.log(1 / 3)

    @property
    def converged(self):
        return True


def test_capability_protocols_are_runtime_checkable():
    model = _ConstantModel([1 / 3] * 3)
    assert isinstance(model, FittedModel)
    assert not isinstance(model, ProvidesMarginalUtilities)


def test_likelihood_metrics_without_parameter_count():
    frame = pd.DataFrame({"CHOICE": [1, 2], "a": [0.5, 0.25], "b": [0.25, 0.5], "c": [0.25, 0.25]})
    metrics = likelihood_metrics(frame, "CHOICE", PROBLEM.choice_mapping, null_ll=2 * np.log(1 / 3), n_params=None)
    assert np.isfinite(metrics["rho_sq"])
    assert np.isnan(metrics["rho_bar_sq"])


def test_validator_accepts_choice_problem_mapping_and_no_parameter_count():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"CHOICE": rng.choice([1, 2, 3], size=60, p=[0.5, 0.3, 0.2]), "C_AV": 1})

    def fit(train):
        shares = PROBLEM.choices(train).value_counts(normalize=True).reindex(PROBLEM.alternatives, fill_value=0)
        return _ConstantModel(list(shares))

    validator = InternalValidator(fit, choice_column="CHOICE", choice_mapping=PROBLEM.choice_mapping, random_state=0)
    result = validator.cross_validate(data, n_splits=3)
    assert result.n_success == 3
    assert result.detail["rho_bar_sq"].isna().all()
    assert result.detail["rho_sq"].notna().all()
