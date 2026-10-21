"""The RUMBoost wrapper (skipped unless the RUMBoost source is importable)."""

import numpy as np
import pytest

import importlib.util

# Only check availability: RUMBoost itself runs in a worker process, never imported here.
if importlib.util.find_spec("lightgbm") is None or importlib.util.find_spec("rumboost") is None:
    pytest.skip("RUMBoost source not on PYTHONPATH (see dcmbench.engines.rumboost)", allow_module_level=True)

from dcmbench.engines.rumboost import RumboostFitter  # noqa: E402
from dcmbench.evaluation import behavioral_regularity, evaluate  # noqa: E402
from synthetic import ALTS, PROBLEM, dataset, synthetic  # noqa: E402

VARIABLES = {a: [f"{a.upper()}_TT", f"{a.upper()}_CO"] for a in ALTS}
MONOTONE = {c: -1 for cols in VARIABLES.values() for c in cols}


def test_rumboost_fits_masks_and_respects_monotonicity():
    data = synthetic(n_people=300, seed=21)
    model = RumboostFitter(PROBLEM, VARIABLES, MONOTONE, num_iterations=300, validation_fraction=0.2)(data)
    probs = model.predict_probabilities(data)
    assert list(probs.columns) == list(ALTS) and model.piecewise_constant and model.n_params is None
    np.testing.assert_allclose(probs.to_numpy(), model.predict_probabilities(data.copy()).to_numpy())
    regularity = behavioral_regularity(model, data, PROBLEM, attributes=["cost"]).summary
    assert (regularity["weak"] == 1.0).all()  # monotone utilities: never an increase with own cost
    result = evaluate(model, dataset(data), name="rumboost", analyses=("vot",))
    assert result.fit.rho_bar_sq is None
    assert result["vot"].settings["step_fraction"] == pytest.approx(1 / 200)
    assert model.describe()["availability_in_training"] is False


def test_rumboost_specification_checks():
    with pytest.raises(ValueError, match="exactly the alternatives"):
        RumboostFitter(PROBLEM, {"train": ["TRAIN_TT"]})
    with pytest.raises(ValueError, match="-1, 0 or 1"):
        RumboostFitter(PROBLEM, VARIABLES, {"TRAIN_TT": 2})
