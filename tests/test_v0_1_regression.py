"""
The 1.0 stack (data layer, spec v2, Biogeme engine, evaluation) against the
0.1.x regression baseline in tests/fixtures/v0_1_baseline/.

Needs Biogeme and the datasets: opt in with DCMBENCH_NETWORK_TESTS=1.
"""

import json
import os
from pathlib import Path

import pytest

import dcmbench.data as data
from dcmbench.evaluation import Holdout, Perturbation, Scenario, evaluate, run_scenario
from dcmbench.specs import load_spec

pytestmark = pytest.mark.skipif(
    os.environ.get("DCMBENCH_NETWORK_TESTS") != "1", reason="set DCMBENCH_NETWORK_TESTS=1 to download datasets"
)
BASELINE = Path(__file__).parent / "fixtures" / "v0_1_baseline"
FULL_SAMPLE = json.loads((BASELINE / "full_sample_estimates.json").read_text())
HOLDOUT = {(r["dataset"], r["model"]): r for r in json.loads((BASELINE / "holdout_80_20.json").read_text())}
VOT_POLICY = json.loads((BASELINE / "vot_and_policy.json").read_text())

# spec file -> (0.1.x model name, holdout key). ModeCanada NL is excluded: its
# nest parameter is bounded to [0.001, 1] and the optimum is not reliably reached.
CLOSED_FORM = {
    "swissmetro_mnl": ("swissmetro_mnl", ("Swissmetro", "MNL")),
    "swissmetro_nl": ("swissmetro_nested_logit", ("Swissmetro", "NL")),
    "swissmetro_custom": ("swissmetro_custom_heterogeneity", ("Swissmetro", "CUSTOM")),
    "modecanada_mnl": ("modecanada_mnl", ("ModeCanada", "MNL")),
    "lpmc_mnl": ("ltds_mnl", ("LPMC", "MNL")),
    "lpmc_nl": ("ltds_nl", ("LPMC", "NL")),
}


def _dataset(spec_name):
    ds = data.load(spec_name.split("_")[0])
    return ds.prepare("benchmark") if "benchmark" in ds.preparations else ds


@pytest.mark.parametrize("spec_name", sorted(CLOSED_FORM))
def test_closed_form_models_reproduce_the_baseline(spec_name, tmp_path, monkeypatch):
    pytest.importorskip("biogeme")
    from dcmbench.engines.biogeme import BiogemeEstimator

    monkeypatch.chdir(tmp_path)
    old_name, holdout_key = CLOSED_FORM[spec_name]
    dataset, spec = _dataset(spec_name), load_spec(BASELINE / "specs" / f"{spec_name}.json")
    fitter = BiogemeEstimator(spec, dataset.problem)

    full = evaluate(fitter, dataset, name=spec_name, analyses=())
    assert full.fit.loglikelihood == pytest.approx(FULL_SAMPLE[old_name]["final_loglikelihood"], abs=0.01)

    holdout = evaluate(fitter, dataset, name=spec_name, protocol=Holdout(test_size=0.2, seed=42), analyses=())
    expected, prediction = HOLDOUT[holdout_key], holdout["prediction"].summary.iloc[0]
    assert holdout.fit.loglikelihood == pytest.approx(expected["is_ll_estimation"], abs=0.01)
    assert prediction["loglikelihood"] == pytest.approx(expected["oos_loglikelihood"], abs=0.2)
    assert prediction["macro_f1"] == pytest.approx(expected["oos_macro_f1"], abs=0.005)
    assert prediction["macro_recall"] == pytest.approx(expected["oos_macro_recall"], abs=0.005)


def test_swissmetro_car_cost_scenario_reproduces_the_baseline():
    pytest.importorskip("biogeme")
    from dcmbench.engines.biogeme import BiogemeEstimator

    dataset = _dataset("swissmetro_mnl")
    model = BiogemeEstimator(load_spec(BASELINE / "specs" / "swissmetro_mnl.json"), dataset.problem)(dataset.frame)
    scenario = Scenario("car cost x1.5", [Perturbation("cost", ("car",), multiply=1.5)])
    table = run_scenario(model, dataset.frame, dataset.problem, scenario).summary.set_index("alternative")
    expected = VOT_POLICY["swissmetro_car_cost_x1_5"]
    assert table.loc["car", "pct_change"] == pytest.approx(expected["car_share_pct_change"]["MNL"], abs=0.05)
    assert table.loc["train", "diversion"] == pytest.approx(expected["share_of_car_loss_diverted_to_train"]["MNL"], abs=0.005)
    assert table.loc["sm", "diversion"] == pytest.approx(expected["share_of_car_loss_diverted_to_swissmetro"]["MNL"], abs=0.005)
