"""Unit tests for dcmbench.specs: expressions, spec validation, draws and the shared likelihood code."""

import json

import numpy as np
import pandas as pd
import pytest

from dcmbench.problem import Attribute, ChoiceProblem
from dcmbench.specs import ExpressionError, SpecError, SpecModel, load_spec, parse
from dcmbench.specs.backends import NumpyBackend
from dcmbench.specs.draws import standard_draws, variates
from dcmbench.specs.expression import derivative, evaluate, symbols
from dcmbench.specs.spec import Draws, RandomParameter
from dcmbench.specs import v1

NP = NumpyBackend()


# --- expressions ------------------------------------------------------------------------------------

@pytest.mark.parametrize("formula", [
    "a + b * c", "-a / (b - 2)", "log(1 + x) * exp(-y)", "x ** 2 + y ** b", "(GA == 0) * c", "+a", "2.5e-3 * x",
])
def test_allowed_formulas_parse(formula):
    parse(formula)


@pytest.mark.parametrize("formula, message", [
    ("__import__('os')", "only log, exp may be called"),
    ("x.real", "Attribute is not allowed"),
    ("x[0]", "Subscript is not allowed"),
    ("lambda: 1", "Lambda is not allowed"),
    ("1 < x < 2", "one comparison at a time"),
    ("log(x, 2)", "exactly one argument"),
    ("True * x", "Constant is not allowed"),
    ("x and y", "BoolOp is not allowed"),
    ("log", "must be called"),
    ("a +", "cannot parse"),
])
def test_disallowed_formulas_are_rejected(formula, message):
    with pytest.raises(ExpressionError, match=message):
        parse(formula)


def test_symbols_and_evaluation():
    node = parse("B * log(1 + COST) * (GA == 0) + A ** 2")
    assert symbols(node) == {"B", "COST", "GA", "A"}
    env = {"B": -0.5, "A": 3.0, "COST": np.array([0.0, np.e - 1]), "GA": np.array([0, 1])}
    np.testing.assert_allclose(evaluate(node, env, NP), [9.0, 9.0])


def test_symbolic_derivatives_match_finite_differences():
    node = parse("B * log(1 + X) * (G == 0) + exp(A * X) / (1 + X ** 2) + X ** A - 3 * X")
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 3.0, 50)
    env = {"B": -0.7, "A": 1.3, "G": rng.integers(0, 2, 50).astype(float), "X": x}
    exact = evaluate(derivative(node, "X"), env, NP)
    h = 1e-6
    numeric = (evaluate(node, {**env, "X": x + h}, NP) - evaluate(node, {**env, "X": x - h}, NP)) / (2 * h)
    np.testing.assert_allclose(exact, numeric, rtol=1e-6, atol=1e-8)
    assert derivative(node, "nothing") == parse("0")


def test_torch_backend_matches_numpy():
    torch = pytest.importorskip("torch")
    from dcmbench.specs.backends import TorchBackend

    node = parse("B * log(1 + X) * (G == 0) + exp(-X)")
    x, g = np.array([0.5, 1.0, 2.0]), np.array([0.0, 1.0, 0.0])
    expected = evaluate(node, {"B": 2.0, "X": x, "G": g}, NP)
    backend = TorchBackend()
    got = evaluate(node, {"B": torch.tensor(2.0, dtype=torch.float64), "X": backend.asarray(x), "G": backend.asarray(g)}, backend)
    np.testing.assert_allclose(got.numpy(), expected)


# --- spec validation --------------------------------------------------------------------------------

def _mnl(**overrides):
    spec = {
        "spec_version": 2, "name": "toy", "model": "MNL", "alternatives": ["a", "b"],
        "parameters": {"ASC": {"start": 0}, "B": {"start": 0}},
        "utilities": {"a": "ASC + B * XA", "b": "B * XB"},
    }
    spec.update(overrides)
    return spec


def test_spec_round_trip_and_hash():
    spec = load_spec(_mnl())
    again = load_spec(json.dumps(spec.to_dict()))
    assert again.to_dict() == spec.to_dict() and again.sha256() == spec.sha256()
    assert spec.data_columns() == {"XA", "XB"} and spec.n_free_parameters == 2


@pytest.mark.parametrize("overrides, message", [
    ({"spec_version": 1}, "spec_version must be 2"),
    ({"utilities": {"a": "ASC + B * XA"}}, "one formula per alternative"),
    ({"model": "PROBIT"}, "model must be one of"),
    ({"parameters": {"ASC": {"start": 0}, "B": {"start": 0}, "UNUSED": {"start": 0}}}, "never used"),
    ({"nests": [{"name": "n", "parameter": "B", "alternatives": ["a"]}]}, "only allowed in NL"),
    ({"model": "NL"}, "needs nests"),
    ({"model": "MXL"}, "needs random_parameters and draws"),
    ({"extra": 1}, "unknown spec fields"),
    ({"parameters": {"ASC": {"start": 5, "upper": 1}, "B": {"start": 0}}}, "outside its bounds"),
    ({"parameters": {"ASC": {"begin": 0}, "B": {"start": 0}}}, "invalid spec field"),
])
def test_invalid_specs(overrides, message):
    with pytest.raises(SpecError, match=message):
        load_spec(_mnl(**overrides))


def test_mxl_spec_validation():
    spec = _mnl(model="MXL", utilities={"a": "ASC + B_RND * XA", "b": "B_RND * XB"},
                parameters={"ASC": {"start": 0}, "B": {"start": 0}, "S": {"start": 1}},
                random_parameters={"B_RND": {"distribution": "lognormal", "mean": "B", "spread": "S", "sign": -1}},
                draws={"number": 10, "seed": 1})
    assert load_spec(spec).random_parameters["B_RND"].sign == -1
    with pytest.raises(SpecError, match="undefined parameters"):
        load_spec({**spec, "random_parameters": {"B_RND": {"distribution": "normal", "mean": "B", "spread": "Q"}}})
    with pytest.raises(SpecError, match="sign applies to lognormal"):
        RandomParameter("R", "normal", "B", "S", sign=-1)


# --- draws ------------------------------------------------------------------------------------------

def test_draws_are_deterministic_and_well_formed():
    settings = Draws(number=500, seed=3)
    a, b = standard_draws("normal", 40, settings, 0), standard_draws("normal", 40, settings, 0)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, standard_draws("normal", 40, settings, 1))  # independent per random parameter
    halton = standard_draws("normal", 200, Draws(number=50, seed=3, type="halton"), 0)
    assert abs(halton.mean()) < 0.01 and abs(halton.std() - 1) < 0.02
    tri = variates(RandomParameter("T", "triangular", "B", "S"), 200, settings, 0)
    assert tri.min() >= -1 and tri.max() <= 1 and abs(tri.mean()) < 0.02


# --- shared likelihood code -------------------------------------------------------------------------

PROBLEM = ChoiceProblem(
    name="toy", alternatives=("a", "b", "c"), choice_column="CHOICE", choice_codes={1: "a", 2: "b", 3: "c"},
    availability={"c": "C_AV"}, group_column="ID",
    attributes={"x": Attribute({"a": "XA", "b": "XB", "c": "XC"}, unit="minute", kind="time"),
                "cost": Attribute({"a": "CA", "b": "CB", "c": "CC"}, unit="CHF", kind="cost")},
)


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ID": np.arange(n) // 3, "CHOICE": rng.integers(1, 3, n), "C_AV": (rng.random(n) > 0.3).astype(int),
        "XA": rng.normal(size=n), "XB": rng.normal(size=n), "XC": rng.normal(size=n),
        "CA": rng.uniform(1, 5, n), "CB": rng.uniform(1, 5, n), "CC": rng.uniform(1, 5, n),
    })


def _spec(model="MNL", **extra):
    parameters = {"A": {"start": 0}, "B": {"start": 0}, "C": {"start": 0}}
    utilities = {"a": "A + B * XA + C * log(CA)", "b": "B * XB + C * log(CB)", "c": "B * XC + C * log(CC)"}
    if model == "NL":
        parameters["MU"] = {"start": 1, "lower": 1, "upper": 10}
        extra["nests"] = [{"name": "ab", "parameter": "MU", "alternatives": ["a", "b"]}]
    if model == "MXL":
        parameters["S"] = {"start": 0}
        utilities = {k: v.replace("B *", "B_RND *") for k, v in utilities.items()}
        extra["random_parameters"] = {"B_RND": {"distribution": "normal", "mean": "B", "spread": "S"}}
        extra.setdefault("draws", {"number": 200, "seed": 0, "panel": False})
    return load_spec({"spec_version": 2, "name": f"toy_{model}", "model": model, "alternatives": ["a", "b", "c"],
                      "parameters": parameters, "utilities": utilities, **extra})


def _mnl_probabilities(values, data):
    v = np.column_stack([values["A"] + values["B"] * data.XA + values["C"] * np.log(data.CA),
                         values["B"] * data.XB + values["C"] * np.log(data.CB),
                         values["B"] * data.XC + values["C"] * np.log(data.CC)])
    v = np.where(PROBLEM.availability_matrix(data).to_numpy(), v, -np.inf)
    e = np.exp(v - v.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_mnl_probabilities_and_loglikelihood():
    data, values = _data(), {"A": 0.3, "B": -0.8, "C": -1.2}
    model = SpecModel(_spec(), PROBLEM, values)
    expected = _mnl_probabilities(values, data)
    np.testing.assert_allclose(model.predict_probabilities(data).to_numpy(), expected, rtol=1e-12)
    chosen = data.CHOICE.to_numpy() - 1
    assert model.loglikelihood(data) == pytest.approx(np.log(expected[np.arange(len(data)), chosen]).sum())


def test_nested_logit_reduces_to_mnl_at_mu_one_and_is_a_distribution():
    data, values = _data(), {"A": 0.3, "B": -0.8, "C": -1.2}
    mnl = SpecModel(_spec(), PROBLEM, values).predict_probabilities(data)
    nl_one = SpecModel(_spec("NL"), PROBLEM, {**values, "MU": 1.0}).predict_probabilities(data)
    np.testing.assert_allclose(nl_one, mnl, rtol=1e-10)
    nl = SpecModel(_spec("NL"), PROBLEM, {**values, "MU": 2.5}).predict_probabilities(data)
    np.testing.assert_allclose(nl.sum(axis=1), 1.0)
    assert (nl.loc[data.C_AV == 0, "c"] == 0).all()
    assert not np.allclose(nl, mnl)


def test_mixed_logit_with_zero_spread_equals_mnl():
    data, values = _data(), {"A": 0.3, "B": -0.8, "C": -1.2}
    mnl = SpecModel(_spec(), PROBLEM, values)
    mxl = SpecModel(_spec("MXL"), PROBLEM, {**values, "S": 0.0})
    np.testing.assert_allclose(mxl.predict_probabilities(data), mnl.predict_probabilities(data), rtol=1e-12)
    assert mxl.loglikelihood(data) == pytest.approx(mnl.loglikelihood(data))


def test_panel_likelihood_multiplies_within_respondent():
    data = _data()
    spec = _spec("MXL", draws={"number": 50, "seed": 2, "panel": True})
    model = SpecModel(spec, PROBLEM, {"A": 0.3, "B": -0.8, "C": -1.2, "S": 0.9})
    per_draw = model.probabilities_per_draw(data)  # (n, R, J)
    chosen = per_draw[np.arange(len(data)), :, data.CHOICE.to_numpy() - 1]
    expected = sum(np.log(np.prod(chosen[data.ID.to_numpy() == i], axis=0).mean()) for i in data.ID.unique())
    assert model.loglikelihood(data) == pytest.approx(expected)
    # panel draws are shared within a respondent, and differ between respondents
    mu = model.marginal_utilities(data, ["x"])
    assert mu.panel
    assert data.ID[0] == data.ID[1] != data.ID[3]
    np.testing.assert_array_equal(mu.values[0, :, 0, 0], mu.values[1, :, 0, 0])
    assert not np.array_equal(mu.values[0, :, 0, 0], mu.values[3, :, 0, 0])


def test_marginal_utilities_are_exact_derivatives():
    data, values = _data(), {"A": 0.3, "B": -0.8, "C": -1.2}
    mu = SpecModel(_spec(), PROBLEM, values).marginal_utilities(data, ["x", "cost"])
    np.testing.assert_allclose(mu.values[:, 0, 0, 0], -0.8)
    np.testing.assert_allclose(mu.values[:, 0, 1, 1], -1.2 / data.CB)  # ∂(C log CB)/∂CB
    mxl = SpecModel(_spec("MXL"), PROBLEM, {**values, "S": 0.5}).marginal_utilities(data, ["x"])
    assert mxl.values.shape == (60, 200, 3, 1) and mxl.chosen_prob.shape == (60, 200)
    assert mxl.values[:, :, 0, 0].std() == pytest.approx(0.5, rel=0.1)


def test_specs_must_match_the_problem():
    other = ChoiceProblem(name="x", alternatives=("a", "b"), choice_column="CHOICE", choice_codes={1: "a", 2: "b"})
    with pytest.raises(SpecError, match="differ"):
        SpecModel(_spec(), other, {"A": 0, "B": 0, "C": 0})
    with pytest.raises(SpecError, match="missing columns"):
        SpecModel(_spec(), PROBLEM, {"A": 0, "B": 0, "C": 0}).predict_probabilities(_data().drop(columns="XA"))


# --- v1 specs ---------------------------------------------------------------------------------------

def test_v1_conversion():
    spec_v1 = {
        "metadata": {"name": "old", "model_type": "NL"},
        "parameters": {"ASC_1": {"initial_value": 0}, "B": {"initial_value": 0},
                       "MU": {"initial_value": 1.5, "lower_bound": 1, "upper_bound": 10}},
        "utilities": {"1": {"formula": "ASC_1 + B * X1"}, "2": {"formula": "B * X2"}, "3": {"formula": "B * X3"}},
        "availability": {"1": "1", "2": "AV_2", "3": "1"},
        "nests": {"n": {"nest_param": "MU", "alternatives": [1, 3]}},
    }
    spec, problem = v1.convert(spec_v1)
    assert spec.model == "NL" and spec.nests[0].alternatives == ("1", "3")
    assert problem.choice_codes == {1: "1", 2: "2", 3: "3"} and problem.availability == {"1": None, "2": "AV_2", "3": None}
    with pytest.raises(NotImplementedError, match="bioDraws"):
        v1.check_supported({"metadata": {"model_type": "MXL"}, "random_parameters": {"B": {}}})
