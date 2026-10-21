"""
The example models on user-supplied data: build a Dataset with from_frame,
describe it with ModelInputs, and evaluate TasteNet, ResLogit and latent class.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import latent_class  # noqa: E402
import reslogit  # noqa: E402
import tastenet  # noqa: E402
from common import ModelInputs  # noqa: E402

import dcmbench.data  # noqa: E402
from dcmbench.evaluation import CrossValidation, Perturbation, Scenario, evaluate  # noqa: E402
from synthetic import PROBLEM, synthetic  # noqa: E402


@pytest.fixture(scope="module")
def own_dataset():
    frame = synthetic(n_people=150, seed=31)
    rng = np.random.default_rng(0)
    frame["AGE"] = rng.integers(18, 80, len(frame))
    frame["FEMALE"] = rng.integers(0, 2, len(frame))
    return dcmbench.data.from_frame(frame, PROBLEM, name="my survey", citation="synthetic")


@pytest.mark.parametrize("module", [tastenet, reslogit, latent_class])
def test_example_models_run_on_user_data(module, own_dataset):
    inputs = ModelInputs.from_dataset(own_dataset, reference="sm", characteristics=["AGE", "FEMALE"])
    scenario = Scenario("car cost x1.5", [Perturbation("cost", ("car",), multiply=1.5)])
    result = evaluate(module.fitter(own_dataset, inputs), own_dataset, name=module.__name__,
                      protocol=CrossValidation(n_splits=2), scenarios=[scenario])
    assert result.problem_name == "synthetic" and result.provenance.dataset["data_version"] == "user-supplied"
    assert result["vot"].settings["route"].startswith("A-")
    assert result["prediction"].summary["n_success"].iloc[0] == 2
    assert np.isfinite(result.fit.loglikelihood)


def test_model_inputs_scales_are_shared_across_alternatives(own_dataset):
    inputs = ModelInputs.from_dataset(own_dataset, reference="sm")
    times = np.concatenate([own_dataset.frame[c] for c in ("TRAIN_TT", "SM_TT", "CAR_TT")])
    assert inputs.time_scale == pytest.approx(np.abs(times).mean())
    with pytest.raises(ValueError, match="not an alternative"):
        ModelInputs.from_dataset(own_dataset, reference="bus")
