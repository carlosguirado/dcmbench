"""
Engines: Biogeme and PyTorch parity on the same spec, known-truth recovery,
scikit-learn and precomputed models, and evaluate() end to end.
"""

import os

import numpy as np
import pandas as pd
import pytest

from dcmbench.data.dataset import Dataset, DatasetInfo
from dcmbench.engines.precomputed import PrecomputedModel
from dcmbench.engines.sklearn import SklearnFitter
from dcmbench.evaluation import CrossValidation, evaluate
from dcmbench.problem import Attribute, ChoiceProblem
from dcmbench.specs import load_spec

ALTS = ("train", "sm", "car")
PROBLEM = ChoiceProblem(
    name="synthetic", alternatives=ALTS, choice_column="CHOICE", choice_codes={1: "train", 2: "sm", 3: "car"},
    availability={"car": "CAR_AV"}, group_column="ID",
    attributes={"time": Attribute({a: f"{a.upper()}_TT" for a in ALTS}, unit="minute", kind="time"),
                "cost": Attribute({a: f"{a.upper()}_CO" for a in ALTS}, unit="CHF", kind="cost")},
)
TRUE = {"ASC_TRAIN": -0.4, "ASC_CAR": 0.3, "B_TIME": -0.02, "B_COST": -0.05, "B_TIME_S": 0.012}


def _synthetic(n_people=600, choices_each=5, mixed=False, seed=0):
    """Choices simulated from an MNL, or a panel MXL with β_time ~ N(B_TIME, B_TIME_S) per respondent."""
    rng = np.random.default_rng(seed)
    n = n_people * choices_each
    data = pd.DataFrame({"ID": np.repeat(np.arange(n_people), choices_each)})
    for alt in ALTS:
        data[f"{alt.upper()}_TT"] = rng.uniform(20, 150, n)
        data[f"{alt.upper()}_CO"] = rng.uniform(5, 60, n)
    data["CAR_AV"] = (rng.random(n) > 0.15).astype(int)
    b_time = TRUE["B_TIME"] + (TRUE["B_TIME_S"] * rng.standard_normal(n_people)[data.ID] if mixed else 0)
    v = np.column_stack([
        TRUE["ASC_TRAIN"] + b_time * data.TRAIN_TT + TRUE["B_COST"] * data.TRAIN_CO,
        b_time * data.SM_TT + TRUE["B_COST"] * data.SM_CO,
        TRUE["ASC_CAR"] + b_time * data.CAR_TT + TRUE["B_COST"] * data.CAR_CO,
    ]) + rng.gumbel(size=(n, 3))
    v[data.CAR_AV.to_numpy() == 0, 2] = -np.inf
    data["CHOICE"] = v.argmax(axis=1) + 1
    return data


def _spec(model="MNL", draws=None):
    b = "B_TIME_RND" if model == "MXL" else "B_TIME"
    spec = {
        "spec_version": 2, "name": f"synthetic_{model.lower()}", "model": model, "alternatives": list(ALTS),
        "parameters": {"ASC_TRAIN": {"start": 0}, "ASC_CAR": {"start": 0}, "B_TIME": {"start": 0}, "B_COST": {"start": 0}},
        "utilities": {"train": f"ASC_TRAIN + {b} * TRAIN_TT + B_COST * TRAIN_CO", "sm": f"{b} * SM_TT + B_COST * SM_CO",
                      "car": f"ASC_CAR + {b} * CAR_TT + B_COST * CAR_CO"},
    }
    if model == "NL":
        spec["parameters"]["MU"] = {"start": 1.5, "lower": 1, "upper": 10}
        spec["nests"] = [{"name": "existing", "parameter": "MU", "alternatives": ["train", "car"]}]
    if model == "MXL":
        spec["parameters"]["B_TIME_S"] = {"start": 0.01}
        spec["random_parameters"] = {"B_TIME_RND": {"distribution": "normal", "mean": "B_TIME", "spread": "B_TIME_S"}}
        spec["draws"] = draws or {"number": 100, "seed": 1, "type": "halton", "panel": True}
    return load_spec(spec)


def _dataset(frame):
    return Dataset("synthetic", frame, PROBLEM, DatasetInfo("synthetic", "none", "none", None, "test", "0" * 64, "none"))


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("model", ["MNL", "NL", "MXL"])
def test_biogeme_and_torch_agree(model, in_tmp_dir):
    pytest.importorskip("biogeme")
    pytest.importorskip("torch")
    from dcmbench.engines.biogeme import BiogemeEstimator
    from dcmbench.engines.torch import TorchEstimator

    data = _synthetic(n_people=300, mixed=(model == "MXL"))
    spec = _spec(model)
    biogeme_fit = BiogemeEstimator(spec, PROBLEM)(data)
    torch_fit = TorchEstimator(spec, PROBLEM)(data)
    assert biogeme_fit.converged and torch_fit.converged
    assert torch_fit.loglikelihood(data) == pytest.approx(biogeme_fit.loglikelihood(data), abs=1e-4)
    a, b = biogeme_fit.coefficients().set_index("name"), torch_fit.coefficients().set_index("name")
    np.testing.assert_allclose(a.estimate, b.estimate, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(a.robust_std_err, b.robust_std_err, rtol=1e-2)
    assert os.listdir(in_tmp_dir) == []  # no biogeme.toml, HTML or YAML files


def test_torch_recovers_true_parameters_mnl_and_panel_mixed_logit():
    pytest.importorskip("torch")
    from dcmbench.engines.torch import TorchEstimator

    for model, data in (("MNL", _synthetic(mixed=False, seed=3)), ("MXL", _synthetic(n_people=1500, mixed=True, seed=4))):
        fit = TorchEstimator(_spec(model), PROBLEM)(data).coefficients().set_index("name")
        for name in fit.index:
            z = (abs(fit.loc[name, "estimate"]) - TRUE[name]) / fit.loc[name, "robust_std_err"] if name == "B_TIME_S" \
                else (fit.loc[name, "estimate"] - TRUE[name]) / fit.loc[name, "robust_std_err"]
            assert abs(z) < 3.5, (model, name, fit.loc[name].to_dict())


def test_evaluate_with_a_spec_model_uses_exact_marginal_utilities():
    pytest.importorskip("torch")
    from dcmbench.engines.torch import TorchEstimator

    dataset = _dataset(_synthetic(n_people=200, seed=5))
    result = evaluate(TorchEstimator(_spec(), PROBLEM), dataset, name="mnl", protocol=CrossValidation(n_splits=3))
    model_info = result.provenance.model
    assert model_info["engine"] == "pytorch" and model_info["model_type"] == "MNL"
    vot = result["vot"].summary.set_index("alternative")
    assert result["vot"].settings["route"] == "A-population"
    assert np.ptp(vot["mean"]) < 1e-9  # generic coefficients: same VOT for every alternative
    assert result["prediction"].summary["n_success"].iloc[0] == 3


def test_sklearn_models_are_masked_and_flagged():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    dataset = _dataset(_synthetic(n_people=200, seed=6))
    features = [c for c in dataset.frame.columns if c.endswith(("_TT", "_CO"))]
    forest = SklearnFitter(RandomForestClassifier(n_estimators=20, random_state=0), PROBLEM, features)
    assert forest.piecewise_constant
    logit = SklearnFitter(LogisticRegression(max_iter=1000), PROBLEM, features)
    assert not logit.piecewise_constant
    result = evaluate(forest, dataset, name="forest", protocol=CrossValidation(n_splits=3), analyses=("regularity",))
    assert result.fit.n_params is None and result.fit.rho_bar_sq is None
    assert "rho_bar_sq" in result.fit.not_applicable
    assert any("piecewise constant" in f for f in result["regularity"].flags)
    probabilities = forest(dataset.frame).predict_probabilities(dataset.frame)
    assert list(probabilities.columns) == list(ALTS)


def test_precomputed_model_scores_but_refuses_behavioral_analyses():
    dataset = _dataset(_synthetic(n_people=50, seed=7).assign(KEY=lambda d: np.arange(len(d))))
    table = pd.DataFrame({"KEY": dataset.frame.KEY, "train": 0.3, "sm": 0.5, "car": 0.2})
    table.loc[dataset.frame.CAR_AV.to_numpy() == 0, ["train", "sm", "car"]] = [0.4, 0.6, 0.0]
    model = PrecomputedModel(table, PROBLEM, key="KEY", n_params=4, source="apollo")
    result = evaluate(model, dataset, name="apollo mnl", analyses=())
    assert result.fit.n_obs == len(dataset.frame)
    with pytest.raises(ValueError, match="precomputed"):
        evaluate(model, dataset, name="apollo mnl")
    with pytest.raises(KeyError, match="no precomputed"):
        model.predict_probabilities(dataset.frame.assign(KEY=-1))
