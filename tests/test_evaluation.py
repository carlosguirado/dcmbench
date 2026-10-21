"""
Known-truth tests for dcmbench.evaluation.

Oracle models with known coefficients make every quantity available in closed
form: MNL elasticities, VOT = β_time/β_cost, the negative-VOT share of a
normal mixed logit, IIA diversion, and regularity counts.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from dcmbench.data.dataset import Dataset, DatasetInfo
from dcmbench.evaluation import (
    CrossValidation,
    Holdout,
    NonDeterministicModelError,
    Perturbation,
    Scenario,
    behavioral_regularity,
    check_deterministic,
    evaluate,
    point_elasticities,
    predict,
    run_scenario,
    sensitivity,
    vot_from_marginal_utilities,
    vot_from_probabilities,
    weighted_quantiles,
)
from dcmbench.evaluation.vot import _posterior_weights
from dcmbench.problem import Attribute, ChoiceProblem
from dcmbench.protocols import MarginalUtilities, ProbabilityTableError
from dcmbench.results import EvaluationResult

ALTS = ("train", "sm", "car")
PROBLEM = ChoiceProblem(
    name="synthetic",
    alternatives=ALTS,
    choice_column="CHOICE",
    choice_codes={1: "train", 2: "sm", 3: "car"},
    availability={"car": "CAR_AV"},
    attributes={
        "time": Attribute({a: f"{a.upper()}_TT" for a in ALTS}, unit="minute", kind="time"),
        "cost": Attribute({a: f"{a.upper()}_CO" for a in ALTS}, unit="CHF", kind="cost"),
    },
    group_column="ID",
)
B_TIME, B_COST = -0.02, -0.1  # VOT = 0.2 CHF/min = 12 CHF/h


def _data(n=400, seed=0, car_unavailable_share=0.2):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"ID": np.arange(n) // 4})
    for alt in ALTS:
        frame[f"{alt.upper()}_TT"] = rng.uniform(20, 200, n)
        frame[f"{alt.upper()}_CO"] = rng.uniform(5, 100, n)
    frame["CAR_AV"] = (rng.random(n) > car_unavailable_share).astype(int)
    model = LinearLogit()
    probs = model.predict_probabilities(frame).to_numpy()
    frame["CHOICE"] = [rng.choice([1, 2, 3], p=p) for p in probs]
    return frame


class LinearLogit:
    """MNL with known coefficients: V_j = asc_j + β_time·time_j + β_cost·cost_j."""

    def __init__(self, b_time=B_TIME, b_cost=B_COST, asc=(0.5, 0.0, 0.3), cost_sign=None):
        self.b_time, self.b_cost, self.asc = b_time, b_cost, np.asarray(asc)
        self.cost_sign = cost_sign  # optional column name holding ±1 multipliers on β_cost

    def utilities(self, data):
        b_cost = self.b_cost * (data[self.cost_sign].to_numpy()[:, None] if self.cost_sign else 1.0)
        time = np.column_stack([data[f"{a.upper()}_TT"] for a in ALTS])
        cost = np.column_stack([data[f"{a.upper()}_CO"] for a in ALTS])
        v = self.asc + self.b_time * time + b_cost * cost
        return np.where(PROBLEM.availability_matrix(data).to_numpy(), v, -np.inf)

    def predict_probabilities(self, data):
        v = self.utilities(data)
        e = np.exp(v - v.max(axis=1, keepdims=True))
        return pd.DataFrame(e / e.sum(axis=1, keepdims=True), columns=list(ALTS), index=data.index)

    @property
    def n_params(self):
        return 4

    def null_loglikelihood(self, data):
        return float(-np.log(PROBLEM.availability_matrix(data).sum(axis=1)).sum())

    @property
    def converged(self):
        return True

    def marginal_utilities(self, data, attributes):
        coefficient = {"time": self.b_time, "cost": self.b_cost}
        values = np.empty((len(data), 1, len(ALTS), len(attributes)))
        for k, attribute in enumerate(attributes):
            values[..., k] = coefficient[attribute]
        return MarginalUtilities(values, np.ones((len(data), 1)), ALTS, attributes)


class ProbabilityOnly:
    """Wrap a model, exposing only the core protocol (forces VOT Route B)."""

    def __init__(self, model):
        self.model = model

    def predict_probabilities(self, data):
        return self.model.predict_probabilities(data)

    n_params = property(lambda self: self.model.n_params)
    converged = property(lambda self: self.model.converged)

    def null_loglikelihood(self, data):
        return self.model.null_loglikelihood(data)


class MixedLogit(LinearLogit):
    """β_time ~ N(mu, sigma) with R fixed draws per observation (common random numbers)."""

    def __init__(self, mu=-0.02, sigma=0.01, draws=2000, seed=1):
        super().__init__(b_time=mu)
        self.mu, self.sigma, self.draws, self.seed = mu, sigma, draws, seed

    def _b_time(self, data):
        rng = np.random.default_rng(self.seed)
        return self.mu + self.sigma * rng.standard_normal((len(data), self.draws))

    def _probabilities_per_draw(self, data):
        b_time = self._b_time(data)[:, :, None]
        time = np.column_stack([data[f"{a.upper()}_TT"] for a in ALTS])[:, None, :]
        cost = np.column_stack([data[f"{a.upper()}_CO"] for a in ALTS])[:, None, :]
        v = self.asc + b_time * time + self.b_cost * cost
        v = np.where(PROBLEM.availability_matrix(data).to_numpy()[:, None, :], v, -np.inf)
        e = np.exp(v - v.max(axis=2, keepdims=True))
        return e / e.sum(axis=2, keepdims=True)

    def predict_probabilities(self, data):
        return pd.DataFrame(self._probabilities_per_draw(data).mean(axis=1), columns=list(ALTS), index=data.index)

    def marginal_utilities(self, data, attributes):
        b_time = self._b_time(data)
        per_draw = self._probabilities_per_draw(data)
        chosen = PROBLEM.choices(data).map({a: i for i, a in enumerate(ALTS)}).to_numpy()
        chosen_prob = per_draw[np.arange(len(data)), :, chosen]
        values = np.empty((len(data), self.draws, len(ALTS), len(attributes)))
        for k, attribute in enumerate(attributes):
            values[..., k] = b_time[:, :, None] if attribute == "time" else self.b_cost
        return MarginalUtilities(values, np.full((len(data), self.draws), 1 / self.draws), ALTS, attributes,
                                 chosen_prob=chosen_prob)


class SteppedCost(LinearLogit):
    """Piecewise-constant in cost: utilities use cost rounded down to multiples of 10."""

    piecewise_constant = True

    def utilities(self, data):
        rounded = data.copy()
        for alt in ALTS:
            rounded[f"{alt.upper()}_CO"] = np.floor(data[f"{alt.upper()}_CO"] / 10) * 10
        return super().utilities(rounded)


def _dataset(frame):
    info = DatasetInfo("synthetic", "none", "none", None, "test", "0" * 64, "none")
    return Dataset("synthetic", frame, PROBLEM, info)


# --- prediction boundary --------------------------------------------------------------------------

def test_predict_rejects_contract_violations_and_masks_when_asked():
    data = _data(20)

    class Leaky(LinearLogit):
        def predict_probabilities(self, d):
            return pd.DataFrame(1 / 3, columns=list(ALTS), index=d.index)

    with pytest.raises(ProbabilityTableError, match="unavailable"):
        predict(Leaky(), data, PROBLEM)
    leaky = Leaky()
    leaky.handles_availability = False
    masked = predict(leaky, data, PROBLEM)
    assert np.allclose(masked.loc[data["CAR_AV"] == 0, "car"], 0)
    assert np.allclose(masked.sum(axis=1), 1)


def test_nondeterministic_models_are_rejected():
    class Noisy(LinearLogit):
        def predict_probabilities(self, d):
            p = super().predict_probabilities(d) + np.random.default_rng().uniform(0, 1e-9, (len(d), 3))
            return p.div(p.sum(axis=1), axis=0).where(PROBLEM.availability_matrix(d), 0.0)

    with pytest.raises(NonDeterministicModelError):
        check_deterministic(Noisy(), _data(20), PROBLEM)
    check_deterministic(MixedLogit(draws=50), _data(20), PROBLEM)


# --- elasticities ---------------------------------------------------------------------------------

def test_point_elasticities_match_mnl_closed_form():
    data, model = _data(), LinearLogit()
    result = point_elasticities(model, data, PROBLEM, "cost", keep_individual=True)
    p = model.predict_probabilities(data)
    individual = result.tables["individual"]
    for changed in ALTS:
        x = data[f"{changed.upper()}_CO"]
        for responding in ALTS:
            rows = individual[(individual.changed == changed) & (individual.responding == responding)]
            obs = rows["observation"]
            if responding == changed:
                expected = B_COST * x[obs] * (1 - p.loc[obs, changed])
            else:
                expected = -B_COST * x[obs] * p.loc[obs, changed]
            np.testing.assert_allclose(rows["elasticity"].to_numpy(), expected.to_numpy(), rtol=1e-5, atol=1e-9)
    own_car = result.summary.query("changed == 'car' and responding == 'car'")
    assert own_car["n_obs"].iloc[0] == int(data["CAR_AV"].sum())


# --- VOT ------------------------------------------------------------------------------------------

def test_vot_routes_agree_for_linear_logit():
    data, model = _data(), LinearLogit()
    route_a = vot_from_marginal_utilities(model, data, PROBLEM)
    route_b = vot_from_probabilities(ProbabilityOnly(model), data, PROBLEM)
    expected = 60 * B_TIME / B_COST  # CHF per hour
    for result in (route_a, route_b):
        summary = result.summary.set_index("alternative")
        np.testing.assert_allclose(summary["mean"], expected, rtol=1e-5)
        np.testing.assert_allclose(summary["q50"], expected, rtol=1e-5)
        assert (summary["share_negative"] == 0).all()
        assert result.settings["currency_per_hour"] == "CHF"
    assert route_a.settings["route"] == "A-population" and route_b.settings["route"] == "B"
    assert route_a.summary.set_index("alternative").loc["car", "n_obs"] == data["CAR_AV"].sum()


def test_mixed_logit_negative_vot_share_matches_closed_form():
    mu, sigma = -0.02, 0.01
    model, data = MixedLogit(mu, sigma), _data(300)
    result = vot_from_marginal_utilities(model, data, PROBLEM, upper_bound=30.0, keep_samples=True)
    summary = result.summary.set_index("alternative")
    assert summary.loc["train", "share_negative"] == pytest.approx(norm.cdf(-abs(mu) / sigma), abs=0.005)
    # VOT = 60 β_time / β_cost ~ N(12, 6) CHF/h; P(VOT > 30) = 1 - Φ(3)
    assert summary.loc["train", "share_above_bound"] == pytest.approx(1 - norm.cdf(3), abs=0.002)
    assert summary.loc["train", "mean"] == pytest.approx(60 * mu / B_COST, rel=0.01)


def test_posterior_vot_shifts_towards_the_observed_choices():
    data = _data(300)
    time = np.column_stack([data[f"{a.upper()}_TT"] for a in ALTS])
    available = PROBLEM.availability_matrix(data).to_numpy()
    fastest = np.where(available, time, np.inf).argmin(axis=1)
    data["CHOICE"] = fastest + 1  # everyone chose their fastest mode: strong time sensitivity
    model = MixedLogit(-0.02, 0.01, draws=500)
    population = vot_from_marginal_utilities(model, data, PROBLEM)
    posterior = vot_from_marginal_utilities(model, data, PROBLEM, posterior=True)
    assert posterior.settings["route"] == "A-posterior"
    assert posterior.summary["mean"].iloc[0] > population.summary["mean"].iloc[0]


def test_panel_posterior_multiplies_within_respondent():
    data = pd.DataFrame({"ID": [7, 7]})
    mu = MarginalUtilities(np.zeros((2, 2, 3, 1)), np.full((2, 2), 0.5), ALTS, ["time"],
                           chosen_prob=np.array([[0.9, 0.1], [0.9, 0.1]]))
    np.testing.assert_allclose(_posterior_weights(mu, data, PROBLEM), [[0.9, 0.1], [0.9, 0.1]])
    mu.panel = True
    np.testing.assert_allclose(_posterior_weights(mu, data, PROBLEM)[0], [0.81 / 0.82, 0.01 / 0.82])


def test_weighted_quantiles():
    values, weights = np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0])
    assert weighted_quantiles(values, weights, [0.5])[0] == pytest.approx(2.5)
    assert weighted_quantiles(np.arange(101.0), np.ones(101), [0.5])[0] == pytest.approx(50.0)


# --- scenarios ------------------------------------------------------------------------------------

def test_mnl_diversion_is_proportional_to_shares_for_identical_travellers():
    row = _data(1).assign(CAR_AV=1)
    data = pd.concat([row] * 10, ignore_index=True)
    model = LinearLogit()
    base = model.predict_probabilities(data).iloc[0]
    scenario = Scenario("car cost x1.5", [Perturbation("cost", ("car",), multiply=1.5)])
    table = run_scenario(model, data, PROBLEM, scenario).summary.set_index("alternative")
    for alt in ("train", "sm"):
        assert table.loc[alt, "diversion"] == pytest.approx(base[alt] / (base["train"] + base["sm"]), rel=1e-9)
    assert np.isnan(table.loc["car", "diversion"])
    s0, s1 = table.loc["car", "share_base"], table.loc["car", "share_scenario"]
    assert table.loc["car", "arc_elasticity"] == pytest.approx(((s1 - s0) / ((s1 + s0) / 2)) / (0.5 / 1.25))
    assert table["change"].sum() == pytest.approx(0, abs=1e-12)


def test_sensitivity_curve_includes_baseline():
    data, model = _data(), LinearLogit()
    curve = sensitivity(model, data, PROBLEM, "cost", ("car",), [1.0, 2.0]).summary
    baseline = model.predict_probabilities(data).mean()
    at_one = curve[curve.multiplier == 1.0].set_index("alternative")["share"]
    np.testing.assert_allclose(at_one[list(ALTS)], baseline[list(ALTS)])
    at_two = curve[curve.multiplier == 2.0].set_index("alternative")["share"]
    assert at_two["car"] < at_one["car"]


# --- behavioral regularity ------------------------------------------------------------------------

MODERATE = dict(b_time=-0.005, b_cost=-0.02)  # no near-degenerate probabilities in the synthetic data


def test_regularity_of_a_correctly_signed_logit():
    data = _data()
    result = behavioral_regularity(LinearLogit(**MODERATE), data, PROBLEM)
    summary = result.summary.set_index(["alternative", "attribute"])
    assert (summary["strong"] == 1).all() and (summary["weak"] == 1).all()
    assert summary.loc[("car", "cost"), "n_excluded_unavailable"] == int((data["CAR_AV"] == 0).sum())
    spread = data["TRAIN_CO"].max() - data["TRAIN_CO"].min()
    assert summary.loc[("train", "cost"), "step"] == pytest.approx(spread / 200)
    assert summary.loc[("train", "cost"), "tolerance"] == pytest.approx(spread / 1e6)
    assert summary.loc[("train", "cost"), "definition"] == "feng2024"
    assert summary.loc[("train", "time"), "definition"] == "extension"


def test_regularity_detects_wrongly_signed_responses():
    data = _data().assign(SIGN=np.where(np.arange(400) % 4 == 0, -1.0, 1.0))  # 25% respond the wrong way
    summary = behavioral_regularity(LinearLogit(cost_sign="SIGN", **MODERATE), data, PROBLEM, attributes=["cost"]).summary
    train = summary.set_index("alternative").loc["train"]
    assert train["strong"] == pytest.approx(0.75)
    assert train["weak"] == pytest.approx(0.75)


def test_near_degenerate_probabilities_fail_strong_but_pass_weak():
    """With P close to 0 or 1, |ΔP| falls below the tolerance (Feng et al.: MNL strong can be < 100%)."""
    data, model = _data(), LinearLogit()
    summary = behavioral_regularity(model, data, PROBLEM, attributes=["cost"]).summary.set_index("alternative")
    p = model.predict_probabilities(data)["train"].to_numpy()
    spread = data["TRAIN_CO"].max() - data["TRAIN_CO"].min()
    responsive = abs(B_COST) * (spread / 200) * p * (1 - p) > spread / 1e6  # first-order |ΔP| > τ
    assert summary.loc["train", "weak"] == 1.0
    assert summary.loc["train", "strong"] == pytest.approx(responsive.mean(), abs=0.01)
    assert summary.loc["train", "strong"] < 1.0


def test_regularity_flags_attributes_the_model_ignores():
    problem = ChoiceProblem(
        name="synthetic", alternatives=ALTS, choice_column="CHOICE", choice_codes=PROBLEM.choice_codes,
        availability=PROBLEM.availability,
        attributes={**PROBLEM.attributes, "headway": Attribute({"train": "TRAIN_HE"}, unit="minute", kind="time")},
    )
    data = _data().assign(TRAIN_HE=lambda d: np.linspace(10, 60, len(d)))
    result = behavioral_regularity(LinearLogit(**MODERATE), data, problem, attributes=["headway"])
    assert result.summary["strong"].iloc[0] == 0.0
    assert any("does not use it" in flag for flag in result.flags)


def test_regularity_of_a_constant_attribute_is_not_applicable():
    data = _data().assign(TRAIN_CO=7.0)
    result = behavioral_regularity(LinearLogit(**MODERATE), data, PROBLEM, attributes=["cost"])
    train = result.summary.set_index("alternative").loc["train"]
    assert np.isnan(train["strong"]) and train["n_obs"] == 0
    assert any("constant in the data" in flag for flag in result.flags)


def test_regularity_of_a_piecewise_constant_model_is_flagged():
    result = behavioral_regularity(SteppedCost(), _data(), PROBLEM, attributes=["cost"])
    train = result.summary.set_index("alternative").loc["train"]
    assert train["n_zero_change"] > 0.8 * train["n_obs"]
    assert train["weak"] == 1.0 and train["strong"] < 0.2
    assert any("piecewise constant" in flag for flag in result.flags)


# --- evaluate() -----------------------------------------------------------------------------------

def test_evaluate_in_sample_with_route_a_and_json_round_trip():
    dataset, model = _dataset(_data()), LinearLogit()
    scenario = Scenario("car cost x1.5", [Perturbation("cost", ("car",), multiply=1.5)])
    result = evaluate(model, dataset, name="oracle mnl", scenarios=[scenario])
    assert set(result.analyses) == {"elasticities_time", "elasticities_cost", "vot", "scenario: car cost x1.5", "regularity"}
    p = model.predict_probabilities(dataset.frame)
    chosen = PROBLEM.choices(dataset.frame)
    expected_ll = np.log(p.to_numpy()[np.arange(len(p)), p.columns.get_indexer(chosen)]).sum()
    assert result.fit.loglikelihood == pytest.approx(expected_ll)
    assert result.fit.rho_bar_sq is not None
    assert result["vot"].settings["route"] == "A-population"
    assert result["regularity"].settings["evaluated_on"] == "full sample (in-sample)"
    assert EvaluationResult.from_json(result.to_json()).to_dict() == result.to_dict()


def test_evaluate_probability_only_model_uses_route_b():
    result = evaluate(ProbabilityOnly(LinearLogit()), _dataset(_data()), name="black box", analyses=("vot",))
    assert result["vot"].settings["route"] == "B"
    np.testing.assert_allclose(result["vot"].summary["mean"], 60 * B_TIME / B_COST, rtol=1e-5)


def test_evaluate_with_holdout_and_cross_validation():
    dataset = _dataset(_data())
    fitter = lambda train: LinearLogit()  # noqa: E731  (oracle: ignores the data)

    holdout = evaluate(fitter, dataset, name="oracle", protocol=Holdout(), analyses=("regularity",))
    prediction = holdout["prediction"].summary.iloc[0]
    assert prediction["n_obs"] == 80 and {"macro_f1", "macro_recall", "rho_sq"} <= set(prediction.index)
    assert holdout.fit.n_obs == 320
    assert holdout["regularity"].settings["evaluated_on"] == "holdout test data"

    cv = evaluate(fitter, dataset, name="oracle", protocol=CrossValidation(n_splits=4), analyses=())
    assert cv["prediction"].settings["protocol"] == "cross_validation"
    assert cv["prediction"].tables["detail"]["status"].eq("ok").all()
    assert "macro_f1_mean" in cv["prediction"].summary.columns
    assert EvaluationResult.from_json(cv.to_json()).to_dict() == cv.to_dict()


def test_evaluate_rejects_invalid_requests():
    dataset = _dataset(_data())
    with pytest.raises(ValueError, match="re-estimate"):
        evaluate(LinearLogit(), dataset, name="x", protocol=Holdout())
    with pytest.raises(ValueError, match="unknown analyses"):
        evaluate(LinearLogit(), dataset, name="x", analyses=("fit",))
    with pytest.raises(ValueError, match="stratify"):
        Holdout(grouped=True)
