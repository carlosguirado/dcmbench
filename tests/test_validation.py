"""Unit tests for dcmbench.validation (synthetic data; no Biogeme required)."""

import numpy as np
import pandas as pd
import pytest

from dcmbench.validation.metrics import (
    blend_632,
    build_scoring_frame,
    get_performance_metrics,
    infer_choice_mapping,
    likelihood_metrics,
    validate_choice_mapping,
)
from dcmbench.validation.splitting import iter_bootstrap_indices, iter_kfold_indices
from dcmbench.validation.validator import InternalValidator


def _synthetic_choice_data(n=60, seed=0):
    """
    3-alternative synthetic panel. Columns 1/2/3 hold each alternative's
    "predicted" probability, deliberately named to match the CHOICE codes
    (1/2/3) -- i.e. this fakes a model whose predict_probabilities() output
    columns already coincide with the CHOICE encoding. That's why tests
    below use choice_mapping={1: 1, 2: 2, 3: 3} (an identity map: {CHOICE
    code: probability column name}), not because choice_mapping is always
    a no-op -- see tutorials/internal_validation_swissmetro.py for a comment
    on when the two sides of the mapping differ.
    """
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(3), size=n)
    choice = np.array([rng.choice([1, 2, 3], p=p) for p in probs])
    df = pd.DataFrame(
        {
            "ID": np.arange(n) // 3,
            "CHOICE": choice,
            1: probs[:, 0],
            2: probs[:, 1],
            3: probs[:, 2],
        }
    )
    return df


def test_infer_and_validate_choice_mapping():
    data = _synthetic_choice_data()
    mapping = infer_choice_mapping(data, "CHOICE", [1, 2, 3])
    assert mapping == {1: 1, 2: 2, 3: 3}
    validate_choice_mapping(data, "CHOICE", mapping, [1, 2, 3])


def test_likelihood_and_performance_metrics():
    data = _synthetic_choice_data()
    mapping = {1: 1, 2: 2, 3: 3}
    frame = build_scoring_frame(data[["CHOICE"]], data[[1, 2, 3]], "CHOICE")
    null_ll = -len(frame) * np.log(3)
    ll = likelihood_metrics(frame, "CHOICE", mapping, null_ll=null_ll, n_params=4)
    assert np.isfinite(ll["avg_ll"])
    assert ll["rho_sq"] <= 1.0
    perf = get_performance_metrics(frame, "CHOICE", mapping)
    assert 0 <= perf["pct_correct_predictions"] <= 100
    assert 0 <= perf["fitting_factor"] <= 1
    assert perf["brier"] >= 0


def test_blend_632():
    apparent = {"fitting_factor": 0.8, "RMSE_disaggregate": 0.4, "brier": 0.2}
    oob = {"fitting_factor": 0.5, "RMSE_disaggregate": 0.6, "brier": 0.4}
    out = blend_632(apparent, oob, keys=["fitting_factor", "RMSE_disaggregate", "brier"])
    assert out["fitting_factor"] == pytest.approx(0.368 * 0.8 + 0.632 * 0.5)
    expected_rmse = np.sqrt(0.368 * 0.4**2 + 0.632 * 0.6**2)
    assert out["RMSE_disaggregate"] == pytest.approx(expected_rmse)


def test_kfold_covers_all_rows():
    data = _synthetic_choice_data(n=30)
    seen = np.zeros(len(data), dtype=bool)
    for train_idx, test_idx in iter_kfold_indices(data, n_splits=5, random_state=1):
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        seen[test_idx] = True
    assert seen.all()


def test_group_kfold_keeps_groups_intact():
    data = _synthetic_choice_data(n=30)
    for train_idx, test_idx in iter_kfold_indices(
        data, n_splits=3, random_state=2, group_column="ID"
    ):
        train_ids = set(data.iloc[train_idx]["ID"])
        test_ids = set(data.iloc[test_idx]["ID"])
        assert train_ids.isdisjoint(test_ids)


def test_bootstrap_oob_nonempty():
    data = _synthetic_choice_data(n=40)
    draws = list(iter_bootstrap_indices(data, n_bootstrap=10, random_state=3))
    assert len(draws) == 10
    for train_idx, oob_idx in draws:
        assert len(oob_idx) > 0
        assert len(train_idx) == len(data)


class _DummyFitted:
    def __init__(self, probs_template: pd.DataFrame):
        self._probs = probs_template
        self.n_params = 2
        self.converged = True

    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        # Constant class shares matching template columns
        out = pd.DataFrame(
            np.tile(self._probs.iloc[0].to_numpy(), (len(data), 1)),
            columns=self._probs.columns,
            index=data.index,
        )
        return out

    def null_loglikelihood(self, data: pd.DataFrame) -> float:
        return float(-len(data) * np.log(3))


def test_function_model_fitter_basic():
    """FunctionModelFitter should wire build_fn/predict_fn into the FittedModel
    protocol correctly, using explicit null_ll/n_params/converged callables
    (the path a hand-built MXL model would use -- see
    tutorials/internal_validation_mxl_swissmetro.py)."""
    from dcmbench.validation.estimators import FunctionModelFitter

    data = _synthetic_choice_data(n=30)
    template = data[[1, 2, 3]]
    build_calls = {"n": 0}

    def build_fn(train_df):
        build_calls["n"] += 1
        return {"beta": 0.5}  # stand-in for e.g. Biogeme estimation results

    def predict_fn(results, data):
        out = pd.DataFrame(
            np.tile(template.iloc[0].to_numpy(), (len(data), 1)),
            columns=template.columns,
            index=data.index,
        )
        return out

    fitter = FunctionModelFitter(
        build_fn=build_fn,
        predict_fn=predict_fn,
        null_loglikelihood_fn=lambda results, data: -len(data) * np.log(3),
        n_params_fn=lambda results: 4,
        converged_fn=lambda results: True,
        choice_column="CHOICE",
    )

    fitted = fitter(data)
    assert build_calls["n"] == 1
    assert fitted.n_params == 4
    assert fitted.converged is True
    probs = fitted.predict_probabilities(data.head(5))
    assert list(probs.columns) == [1, 2, 3]
    assert fitted.null_loglikelihood(data) == pytest.approx(-30 * np.log(3))

    # And it should work end-to-end through InternalValidator, same as any
    # other ModelFitter.
    validator = InternalValidator(
        fitter, choice_column="CHOICE", choice_mapping={1: 1, 2: 2, 3: 3}, random_state=0
    )
    result = validator.cross_validate(data, n_splits=3)
    assert result.n_success == 3


def test_function_model_fitter_defaults_without_explicit_callables():
    """Without null_loglikelihood_fn/n_params_fn/converged_fn, sensible
    defaults kick in (equal-share null LL with a warning; results.data.nparam
    or get_beta_values() for n_params; convergence-flag lookup else True)."""
    from dcmbench.validation.estimators import FunctionModelFitter

    class _FakeResults:
        def get_beta_values(self):
            return {"a": 1.0, "b": 2.0}

    data = _synthetic_choice_data(n=12)
    template = data[[1, 2, 3]]

    def predict_fn(results, data):
        return pd.DataFrame(
            np.tile(template.iloc[0].to_numpy(), (len(data), 1)),
            columns=template.columns,
            index=data.index,
        )

    fitter = FunctionModelFitter(build_fn=lambda train_df: _FakeResults(), predict_fn=predict_fn)
    fitted = fitter(data)

    assert fitted.n_params == 2  # len(get_beta_values())
    assert fitted.converged is True  # no convergence flag found -> default True
    with pytest.warns(UserWarning, match="equal-share null LL"):
        null_ll = fitted.null_loglikelihood(data)
    n_observed_alts = data["CHOICE"].nunique()  # fallback uses observed choices, not J=3
    assert null_ll == pytest.approx(-len(data) * np.log(n_observed_alts))


def test_biogeme_spec_fitter_rejects_mxl_with_random_parameters():
    """
    build_model_from_spec does not wire 'random_parameters' into bioDraws, so
    Biogeme itself would raise a cryptic 'MonteCarlo must contain a bioDraws'
    error. BiogemeSpecFitter should fail fast with an explicit message instead.
    """
    pytest.importorskip("biogeme")
    from dcmbench.validation.estimators import BiogemeSpecFitter

    spec = {
        "metadata": {"model_type": "MXL"},
        "random_parameters": {"B_TIME": {"distribution": "normal"}},
        "parameters": {},
        "utilities": {},
        "availability": {},
        "data_mapping": {"choice_variable": "CHOICE", "variables": {}},
    }
    with pytest.raises(NotImplementedError, match="bioDraws"):
        BiogemeSpecFitter(spec, choice_column="CHOICE")


def test_biogeme_spec_fitter_warns_on_mxl_without_random_parameters():
    pytest.importorskip("biogeme")
    from dcmbench.validation.estimators import BiogemeSpecFitter

    spec = {
        "metadata": {"model_type": "MXL"},
        "parameters": {},
        "utilities": {},
        "availability": {},
        "data_mapping": {"choice_variable": "CHOICE", "variables": {}},
    }
    with pytest.warns(UserWarning, match="silently estimate a plain MNL"):
        BiogemeSpecFitter(spec, choice_column="CHOICE")


_SWISSMETRO_LIKE_SPEC = {
    "metadata": {"name": "synthetic_mnl", "model_type": "MNL"},
    "parameters": {
        "ASC_CAR": {"initial_value": 0, "fixed": True},
        "ASC_TRAIN": {"initial_value": 0, "fixed": False},
        "B_TIME": {"initial_value": 0, "fixed": False},
        "B_COST": {"initial_value": 0, "fixed": False},
    },
    "utilities": {
        "1": {"name": "Train", "formula": "ASC_TRAIN + B_TIME * TRAIN_TT + B_COST * TRAIN_CO"},
        "2": {"name": "Car", "formula": "ASC_CAR + B_TIME * CAR_TT + B_COST * CAR_CO"},
    },
    "availability": {"1": "1", "2": "1"},
    "data_mapping": {
        "choice_variable": "CHOICE",
        "variables": {"TRAIN_TT": "", "TRAIN_CO": "", "CAR_TT": "", "CAR_CO": ""},
    },
}


def _synthetic_two_alt_data(n=120, seed=7):
    """
    Small 2-alternative dataset with a real behavioral signal (utility +
    Gumbel noise -> argmax), so a real Biogeme MNL estimation on it actually
    converges to sensible, non-degenerate coefficients rather than just
    "running without crashing".
    """
    rng = np.random.default_rng(seed)
    train_tt = rng.uniform(20, 120, n)
    train_co = rng.uniform(5, 40, n)
    car_tt = rng.uniform(15, 100, n)
    car_co = rng.uniform(5, 60, n)
    true_b_time, true_b_cost, true_asc_train = -0.02, -0.03, 0.3
    u_train = true_asc_train + true_b_time * train_tt + true_b_cost * train_co
    u_car = true_b_time * car_tt + true_b_cost * car_co
    gumbel = rng.gumbel(size=(n, 2))
    choice = np.where(u_train + gumbel[:, 0] > u_car + gumbel[:, 1], 1, 2)
    return pd.DataFrame(
        {
            "CHOICE": choice,
            "TRAIN_TT": train_tt,
            "TRAIN_CO": train_co,
            "CAR_TT": car_tt,
            "CAR_CO": car_co,
        }
    )


def test_biogeme_spec_fitter_real_mnl_cv_and_bootstrap_happy_path():
    """
    Real end-to-end estimation (not a dummy/mock fitter): fitter_from_spec ->
    real Biogeme MNL estimation -> InternalValidator.cross_validate/bootstrap,
    on data with genuine behavioral signal. This is the "happy path" that the
    tutorials exercise manually but that wasn't previously covered by an
    automated regression test.
    """
    pytest.importorskip("biogeme")
    from dcmbench.validation.estimators import fitter_from_spec

    data = _synthetic_two_alt_data()
    fitter = fitter_from_spec(_SWISSMETRO_LIKE_SPEC, choice_column="CHOICE")
    validator = InternalValidator(
        fitter, choice_column="CHOICE", choice_mapping={1: 1, 2: 2}, random_state=0
    )

    cv = validator.cross_validate(data, n_splits=3)
    assert cv.n_success == 3
    assert cv.n_failed == 0
    assert cv.n_not_converged == 0
    assert np.isfinite(cv.detail["rho_sq"]).all()
    # With real signal and n=120, folds should show clearly-better-than-chance
    # fit, not just "some finite number".
    assert cv.summary()["rho_sq"].iloc[0] > 0.05

    boot = validator.bootstrap(data, n_bootstrap=3)
    assert boot.n_success == 3
    assert np.isfinite(boot.summary()["rho_sq"].iloc[0])


def _synthetic_four_alt_partial_availability_data(n=150, seed=11):
    """
    ModeCanada-shaped synthetic data: 4 alternatives, non-identity CHOICE
    codes vs. availability that isn't uniformly 1 (~15% of rows drop the
    'bus'/'air'-like alternatives 3 and 4), unlike the always-3-alternatives
    Swissmetro fixtures used elsewhere in this file. Exercises availability
    handling and >2 alternatives that the other tests don't touch.
    """
    rng = np.random.default_rng(seed)
    time_ = rng.uniform(10, 200, size=(n, 4))
    cost = rng.uniform(5, 80, size=(n, 4))
    avail = np.ones((n, 4))
    avail[:, 2] = rng.random(n) > 0.15  # alt 3 sometimes unavailable
    avail[:, 3] = rng.random(n) > 0.15  # alt 4 sometimes unavailable

    true_asc = np.array([0.0, 0.4, -0.2, 0.1])
    true_b_time, true_b_cost = -0.015, -0.02
    utility = true_asc + true_b_time * time_ + true_b_cost * cost
    utility = np.where(avail == 1, utility, -1e6)
    gumbel = rng.gumbel(size=(n, 4))
    choice = np.argmax(utility + gumbel, axis=1) + 1  # 1-based alternative ids

    df = pd.DataFrame({"CHOICE": choice})
    for i, alt in enumerate(["1", "2", "3", "4"]):
        df[f"TIME_{alt}"] = time_[:, i]
        df[f"COST_{alt}"] = cost[:, i]
        df[f"AV_{alt}"] = avail[:, i].astype(int)
    return df


_FOUR_ALT_SPEC = {
    "metadata": {"name": "synthetic_four_alt_mnl", "model_type": "MNL"},
    "parameters": {
        "ASC_1": {"initial_value": 0, "fixed": True},
        "ASC_2": {"initial_value": 0, "fixed": False},
        "ASC_3": {"initial_value": 0, "fixed": False},
        "ASC_4": {"initial_value": 0, "fixed": False},
        "B_TIME": {"initial_value": 0, "fixed": False},
        "B_COST": {"initial_value": 0, "fixed": False},
    },
    "utilities": {
        "1": {"name": "Alt1", "formula": "ASC_1 + B_TIME * TIME_1 + B_COST * COST_1"},
        "2": {"name": "Alt2", "formula": "ASC_2 + B_TIME * TIME_2 + B_COST * COST_2"},
        "3": {"name": "Alt3", "formula": "ASC_3 + B_TIME * TIME_3 + B_COST * COST_3"},
        "4": {"name": "Alt4", "formula": "ASC_4 + B_TIME * TIME_4 + B_COST * COST_4"},
    },
    "availability": {"1": "AV_1", "2": "AV_2", "3": "AV_3", "4": "AV_4"},
    "data_mapping": {
        "choice_variable": "CHOICE",
        "variables": {
            v: ""
            for v in [
                "TIME_1", "COST_1", "AV_1",
                "TIME_2", "COST_2", "AV_2",
                "TIME_3", "COST_3", "AV_3",
                "TIME_4", "COST_4", "AV_4",
            ]
        },
    },
}


def test_biogeme_spec_fitter_generalizes_to_four_alternatives_with_partial_availability():
    """
    Structural smoke test (modeled on ModeCanada's shape: 4 alternatives,
    partial availability) verified manually against the real ModeCanada
    dataset in-session (rho_sq ~0.4, matching known literature values for a
    basic ModeCanada MNL). This synthetic version keeps that structural
    coverage -- 4 alternatives, non-trivial availability -- in the fast,
    network-free automated test suite.
    """
    pytest.importorskip("biogeme")
    from dcmbench.validation.estimators import fitter_from_spec

    data = _synthetic_four_alt_partial_availability_data()
    fitter = fitter_from_spec(_FOUR_ALT_SPEC, choice_column="CHOICE")
    validator = InternalValidator(
        fitter,
        choice_column="CHOICE",
        choice_mapping={1: 1, 2: 2, 3: 3, 4: 4},
        random_state=0,
    )
    cv = validator.cross_validate(data, n_splits=3)
    assert cv.n_success == 3
    assert np.isfinite(cv.detail["rho_sq"]).all()
    assert cv.summary()["rho_sq"].iloc[0] > 0.05


def test_internal_validator_cv_with_dummy_fitter():
    data = _synthetic_choice_data(n=45)
    template = data[[1, 2, 3]]

    def fitter(train_df):
        return _DummyFitted(template)

    validator = InternalValidator(
        fitter,
        choice_column="CHOICE",
        choice_mapping={1: 1, 2: 2, 3: 3},
        random_state=0,
    )
    result = validator.cross_validate(data, n_splits=3)
    assert result.method == "cv"
    assert result.n_success == 3
    assert len(result.folds) == 3
    summary = result.summary()
    assert "avg_ll" in summary.columns
    assert "avg_ll_mean" in summary.columns


def test_choice_mapping_inference_warns_about_warmup_fit():
    data = _synthetic_choice_data(n=45)
    template = data[[1, 2, 3]]
    calls = {"n": 0}

    def fitter(train_df):
        calls["n"] += 1
        return _DummyFitted(template)

    validator = InternalValidator(fitter, choice_column="CHOICE", random_state=0)
    with pytest.warns(UserWarning, match="warm-up estimation"):
        validator.cross_validate(data, n_splits=3)
    # 1 warm-up fit + 3 folds
    assert calls["n"] == 4


def test_bootstrap_warns_when_fewer_draws_produced_than_requested():
    data = _synthetic_choice_data(n=6)
    data["ID"] = 0  # single group -> OOB is always empty -> zero usable draws

    template = data[[1, 2, 3]]

    def fitter(train_df):
        return _DummyFitted(template)

    validator = InternalValidator(
        fitter,
        choice_column="CHOICE",
        choice_mapping={1: 1, 2: 2, 3: 3},
        group_column="ID",
        random_state=0,
    )
    with pytest.warns(UserWarning, match="only"):
        result = validator.bootstrap(data, n_bootstrap=20)
    assert result.params["n_draws_produced"] < 20


def test_internal_validator_bootstrap_with_dummy_fitter():
    data = _synthetic_choice_data(n=40)
    template = data[[1, 2, 3]]

    def fitter(train_df):
        return _DummyFitted(template)

    validator = InternalValidator(
        fitter,
        choice_column="CHOICE",
        choice_mapping={1: 1, 2: 2, 3: 3},
        random_state=1,
    )
    result = validator.bootstrap(data, n_bootstrap=5)
    assert result.method == "bootstrap"
    assert result.n_success == 5
    summary = result.summary()
    assert "avg_ll" in summary.columns
    assert "fitting_factor_632" in summary.columns
    assert "fitting_factor_oob" in summary.columns
    assert "fitting_factor_apparent" in summary.columns


class _AlwaysFailsFitter:
    """ModelFitter stand-in that always raises, to exercise on_error handling."""

    def __call__(self, train_df: pd.DataFrame):
        raise RuntimeError("simulated estimation failure")


def test_cross_validate_all_folds_failing_does_not_crash():
    """
    on_error='skip' (the default): every fold fails with a real exception.
    cross_validate should degrade gracefully -- no crash, n_success == 0,
    a UserWarning summarizing the failures, and a still-valid (if
    metric-less) summary() -- rather than raising or returning garbage.
    """
    data = _synthetic_choice_data(n=30)

    validator = InternalValidator(
        _AlwaysFailsFitter(),
        choice_column="CHOICE",
        choice_mapping={1: 1, 2: 2, 3: 3},
        random_state=0,
    )
    with pytest.warns(UserWarning, match="0 ok"):
        result = validator.cross_validate(data, n_splits=3)

    assert result.n_success == 0
    assert result.n_failed == 3
    summary = result.summary()
    assert len(summary) == 1
    assert summary["n_success"].iloc[0] == 0


def test_on_error_raise_propagates_real_exceptions():
    """
    on_error='raise' must propagate *real* exceptions from the fitter (not
    just record non-convergence, which is always skipped regardless of
    on_error -- see InternalValidator's on_error docstring).
    """
    data = _synthetic_choice_data(n=30)

    validator = InternalValidator(
        _AlwaysFailsFitter(),
        choice_column="CHOICE",
        choice_mapping={1: 1, 2: 2, 3: 3},
        random_state=0,
        on_error="raise",
    )
    with pytest.raises(RuntimeError, match="simulated estimation failure"):
        validator.cross_validate(data, n_splits=3)
