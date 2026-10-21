"""The generic PyTorch module engine: training, autograd marginal utilities, mixtures."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from dcmbench.engines.torch import TorchEstimator  # noqa: E402
from dcmbench.engines.torch_module import Mixture, TorchModuleFitter, Training  # noqa: E402
from dcmbench.evaluation import evaluate, vot_from_marginal_utilities  # noqa: E402
from synthetic import ALTS, PROBLEM  # noqa: E402
from synthetic import dataset as _dataset  # noqa: E402
from synthetic import spec as _spec  # noqa: E402
from synthetic import synthetic as _synthetic  # noqa: E402

COLUMNS = [f"{a.upper()}_{x}" for a in ALTS for x in ("TT", "CO")]


class LinearLogit(nn.Module):
    """The MNL of tests.test_engines._spec("MNL"), written as a module."""

    def __init__(self):
        super().__init__()
        self.asc = nn.Parameter(torch.zeros(2))  # train, car (sm is the reference)
        self.b = nn.Parameter(torch.zeros(2))  # time, cost

    def forward(self, x, available):
        zero = torch.zeros_like(x["SM_TT"])
        asc = torch.stack([self.asc[0] + zero, zero, self.asc[1] + zero], dim=1)
        time = torch.stack([x[f"{a.upper()}_TT"] for a in ALTS], dim=1)
        cost = torch.stack([x[f"{a.upper()}_CO"] for a in ALTS], dim=1)
        return asc + self.b[0] * time + self.b[1] * cost


class TwoClassLogit(nn.Module):
    """Latent class: two classes with their own time and cost coefficients, constant shares."""

    def __init__(self):
        super().__init__()
        self.classes = nn.ModuleList([LinearLogit(), LinearLogit()])
        with torch.no_grad():
            self.classes[1].b.fill_(-0.01)  # break symmetry
        self.share = nn.Parameter(torch.zeros(2))

    def forward(self, x, available):
        utilities = torch.stack([c(x, available) for c in self.classes], dim=1)  # (n, K, J)
        return Mixture(utilities, self.share.expand(utilities.shape[0], 2))


LBFGS = Training(optimizer="lbfgs", learning_rate=1.0, batch_size=None, max_epochs=30, patience=3, validation_fraction=0.0)


def test_linear_module_matches_the_spec_engine():
    data = _synthetic(n_people=300, seed=11)
    module_fit = TorchModuleFitter(LinearLogit, PROBLEM, COLUMNS, training=LBFGS)(data)
    spec_fit = TorchEstimator(_spec("MNL"), PROBLEM)(data)
    spec_probs = spec_fit.predict_probabilities(data)
    np.testing.assert_allclose(module_fit.predict_probabilities(data), spec_probs, atol=1e-5)
    b = module_fit.module.b.detach().numpy()
    np.testing.assert_allclose(b, [spec_fit.values["B_TIME"], spec_fit.values["B_COST"]], rtol=1e-3)
    assert module_fit.n_params == 4 and module_fit.converged


def test_marginal_utilities_by_autograd_are_in_raw_units():
    data = _synthetic(n_people=200, seed=12)
    fit = TorchModuleFitter(LinearLogit, PROBLEM, COLUMNS, standardize=COLUMNS, training=LBFGS)(data)
    mu = fit.marginal_utilities(data, ["time", "cost"])
    b_time_raw = fit.module.b[0].item() / fit.scaling["TRAIN_TT"][1]
    np.testing.assert_allclose(mu.values[:, 0, 0, 0], b_time_raw, rtol=1e-10)
    vot = vot_from_marginal_utilities(fit, data, PROBLEM).summary.set_index("alternative")
    b_cost = mu.values[:, 0, :, 1]
    np.testing.assert_allclose(vot.loc["sm", "mean"], 60 * mu.values[0, 0, 1, 0] / b_cost[0, 1], rtol=1e-8)


def test_unavailable_alternatives_get_zero_probability_and_predictions_are_deterministic():
    data = _synthetic(n_people=100, seed=13)
    fit = TorchModuleFitter(LinearLogit, PROBLEM, COLUMNS, training=LBFGS)(data)
    probs = fit.predict_probabilities(data)
    assert (probs.loc[data.CAR_AV == 0, "car"] == 0).all()
    pd.testing.assert_frame_equal(probs, fit.predict_probabilities(data))


def test_latent_class_mixture():
    data = _synthetic(n_people=300, seed=14)
    fit = TorchModuleFitter(TwoClassLogit, PROBLEM, COLUMNS, training=LBFGS, seed=1)(data)
    probs = fit.predict_probabilities(data)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    mu = fit.marginal_utilities(data, ["time", "cost"])
    assert mu.values.shape == (len(data), 2, 3, 2) and mu.chosen_prob.shape == (len(data), 2)
    np.testing.assert_allclose(mu.weights.sum(axis=1), 1.0)
    for k in range(2):
        np.testing.assert_allclose(mu.values[:, k, 0, 0], fit.module.classes[k].b[0].item())
    population = vot_from_marginal_utilities(fit, data, PROBLEM)
    posterior = vot_from_marginal_utilities(fit, data, PROBLEM, posterior=True)
    assert population.settings["n_draws"] == 2 and posterior.settings["route"] == "A-posterior"
    assert fit.n_params == 10


def test_early_stopping_restores_the_best_epoch_and_evaluate_runs():
    dataset = _dataset(_synthetic(n_people=200, seed=15))
    training = Training(optimizer="adam", learning_rate=0.05, batch_size=128, max_epochs=40, patience=3)
    fitter = TorchModuleFitter(LinearLogit, PROBLEM, COLUMNS, standardize=COLUMNS, training=training, name="mnl module")
    model = fitter(dataset.frame)
    assert model.report["monitored_on"] == "validation split" and 1 <= model.report["best_epoch"] <= model.report["epochs"]
    result = evaluate(fitter, dataset, name="mnl module", analyses=("vot", "regularity"))
    assert result["vot"].settings["route"] == "A-population"
    assert result.provenance.model["engine"] == "pytorch module"
