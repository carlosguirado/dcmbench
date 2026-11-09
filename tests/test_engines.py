"""
Engines: Biogeme and PyTorch parity on the same spec, known-truth recovery,
scikit-learn and precomputed models, and evaluate() end to end.
"""

import os

import numpy as np
import pandas as pd
import pytest

from dcmbench.engines.precomputed import PrecomputedModel
from dcmbench.engines.sklearn import SklearnFitter
from dcmbench.evaluation import CrossValidation, evaluate
from synthetic import ALTS, PROBLEM, TRUE
from synthetic import dataset as _dataset
from synthetic import spec as _spec
from synthetic import synthetic as _synthetic

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
