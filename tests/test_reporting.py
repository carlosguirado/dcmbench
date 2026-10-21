"""Specification cards, comparisons and figures."""

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import jsonschema  # noqa: E402

from dcmbench.engines.sklearn import SklearnFitter  # noqa: E402
from dcmbench.evaluation import Bootstrap, CrossValidation, Holdout, Perturbation, Scenario, compare, evaluate  # noqa: E402
from dcmbench.reporting import build_card, compare_cards, load_card, render_markdown, save_card, validate_card  # noqa: E402
from dcmbench.reporting.plots import plot_fit_vs_prediction, plot_sensitivity, plot_vot  # noqa: E402
from dcmbench.specs import SpecModel  # noqa: E402
from synthetic import PROBLEM, TRUE, dataset, spec, synthetic  # noqa: E402

SCENARIO = Scenario("car cost x1.5", [Perturbation("cost", ("car",), multiply=1.5)])
MNL_VALUES = {k: TRUE[k] for k in ("ASC_TRAIN", "ASC_CAR", "B_TIME", "B_COST")}


def oracle(model="MNL"):
    """A fitter that ignores the data and returns the true model."""
    values = dict(MNL_VALUES, **({"B_TIME_S": TRUE["B_TIME_S"]} if model == "MXL" else {}))
    draws = {"number": 200, "seed": 3, "type": "halton", "panel": True} if model == "MXL" else None
    return lambda train: SpecModel(spec(model, draws), PROBLEM, values)


@pytest.fixture(scope="module")
def data():
    return dataset(synthetic(n_people=200, seed=41))


@pytest.mark.parametrize("protocol", [None, Holdout(), CrossValidation(n_splits=3), Bootstrap(n_draws=3)])
def test_cards_are_schema_valid_under_every_protocol(protocol, data):
    model = oracle() if protocol is not None else oracle()(data.frame)
    result = evaluate(model, data, name="mnl", protocol=protocol, scenarios=[SCENARIO], vot_upper_bound=60.0)
    card = build_card(result)
    assert card["card_version"] == "0.1.0" and card["data"]["sha256"] == data.info.sha256
    assert card["value_of_time"]["route"] == "A-population"
    np.testing.assert_allclose([r["q50"] for r in card["value_of_time"]["by_alternative"]],
                               60 * TRUE["B_TIME"] / TRUE["B_COST"])
    assert card["policy"][0]["scenario"] == "car cost x1.5"
    assert {p["definition"] for p in card["behavioral_regularity"]["pairs"]} == {"feng2024", "extension"}
    if protocol is None:
        assert card["prediction"] is None and card["protocol"]["prediction"] is None
    else:
        metrics = card["prediction"]["metrics"]
        assert {"avg_ll", "accuracy"} <= set(metrics) and 0 < metrics["accuracy"]["estimate"] <= 1
        if isinstance(protocol, CrossValidation):
            assert metrics["macro_f1"]["spread"] is not None and metrics["macro_f1"]["how"] == "mean (sd) over folds"


def test_card_round_trip_rendering_and_rejection(data, tmp_path):
    card = build_card(evaluate(oracle(), data, name="mnl", protocol=CrossValidation(n_splits=3), scenarios=[SCENARIO]))
    path = save_card(card, tmp_path / "mnl.card.json")
    assert load_card(path) == json.loads(path.read_text()) == card
    text = render_markdown(card)
    for heading in ("## Statistical fit", "## Prediction", "## Value of time", "## Policy: car cost x1.5",
                    "## Behavioral regularity"):
        assert heading in text
    broken = dict(card)
    del broken["fit"]
    with pytest.raises(jsonschema.ValidationError):
        validate_card(broken)


def test_card_of_a_model_without_parameter_count(data):
    from sklearn.linear_model import LogisticRegression

    features = [c for c in data.frame.columns if c.endswith(("_TT", "_CO"))]
    result = evaluate(SklearnFitter(LogisticRegression(max_iter=2000), PROBLEM, features), data, name="logit",
                      protocol=CrossValidation(n_splits=3))
    card = build_card(result)
    assert card["fit"]["n_params"] is None and card["fit"]["rho_bar_sq"] is None
    assert "rho_bar_sq" in card["fit"]["not_applicable"]
    assert card["value_of_time"]["route"] == "B" and card["model"]["engine"] == "scikit-learn"


def test_compare_uses_identical_folds_and_tabulates(data, tmp_path):
    comparison = compare({"MNL": oracle(), "MXL": oracle("MXL")}, data, protocol=CrossValidation(n_splits=3),
                         scenarios=[SCENARIO])
    folds = {name: r["prediction"].tables["detail"][["fold", "n_train", "n_eval"]] for name, r in comparison.results.items()}
    assert folds["MNL"].equals(folds["MXL"])
    table = comparison.table()
    assert list(table["model"]) == ["MNL", "MXL"]
    assert {"oos_avg_ll", "oos_avg_ll_sd", "vot_median[train]", "regularity_strong_cost[car]", "pct_change[car]"} <= set(table.columns)
    assert table.loc["MXL", "vot_share_negative[train]"] > 0 == table.loc["MNL", "vot_share_negative[train]"]
    out = comparison.save(tmp_path / "cards")
    assert sorted(p.name for p in out.iterdir()) == ["synthetic_MNL.card.json", "synthetic_MNL.card.md",
                                                     "synthetic_MXL.card.json", "synthetic_MXL.card.md",
                                                     "synthetic_comparison.csv"]
    assert compare_cards([load_card(out / "synthetic_MNL.card.json")]).shape[0] == 1


def test_figures(data):
    options = dict(protocol=CrossValidation(n_splits=3), keep_vot_samples=True,
                   sensitivity=("cost", ["car"], [0.5, 1.0, 1.5, 2.0]))
    comparison = compare({"MNL": oracle(), "MXL": oracle("MXL")}, data, **options)
    vot_figure = plot_vot(comparison.results, "train")
    density_axis, share_axis = vot_figure.axes
    assert len(density_axis.patches) >= 1 and len(share_axis.collections) == 1  # MXL density, MNL spike
    assert len(density_axis.get_legend().get_texts()) == 2
    curve = plot_sensitivity(comparison.results, "car")
    shares = curve.axes[0].lines[0].get_ydata()
    assert np.all(np.diff(shares) < 0)  # car share falls as car cost rises
    figure = plot_fit_vs_prediction(list(comparison.cards().values()))
    assert len(figure.axes[0].collections) == 2
    with pytest.raises(ValueError, match="keep_vot_samples"):
        plot_vot({"MNL": evaluate(oracle(), data, name="x", analyses=("vot",))}, "train")
