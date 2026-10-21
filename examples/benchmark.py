"""
The full benchmark: every model family on each dataset under identical
5-fold, respondent-grouped cross-validation, with a specification card per
model, comparison tables and figures.

    PYTHONPATH=third_party/rumboost python examples/benchmark.py [--datasets swissmetro] [--out results]
        [--models MNL NL MXL TasteNet] [--protocol holdout]

Writes to ``--out`` (default ``results/``):

- ``<dataset>_<model>.card.json`` / ``.card.md``: one specification card per model;
- ``<dataset>_comparison.csv``: the models side by side;
- ``vot_<dataset>.png``: VOT distributions for the car alternative;
- ``sensitivity_<dataset>.png``: car share as car cost is scaled;
- ``fit_vs_prediction.png``: in-sample against out-of-sample log-likelihood, all datasets.

Without the RUMBoost source on PYTHONPATH, RUMBoost is skipped.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import common  # noqa: E402
import latent_class  # noqa: E402
import reslogit  # noqa: E402
import rumboost_benchmark  # noqa: E402
import tastenet  # noqa: E402
from dcmbench.engines.sklearn import SklearnFitter  # noqa: E402
from dcmbench.engines.torch import TorchEstimator  # noqa: E402
from dcmbench.evaluation import CrossValidation, Holdout, compare  # noqa: E402
from dcmbench.reporting.plots import plot_fit_vs_prediction, plot_sensitivity, plot_vot  # noqa: E402
from dcmbench.specs import load_spec  # noqa: E402

SPECS = Path(__file__).parent / "specs"
MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


def model_families(dataset):
    """Name -> fitter for every model family available on this dataset."""
    problem, inputs = dataset.problem, common.benchmark_inputs(dataset)
    models = {}
    for kind in ("mnl", "nl", "mxl", "custom"):
        path = SPECS / f"{dataset.name}_{kind}.json"
        if path.exists():
            models[kind.upper()] = TorchEstimator(load_spec(path), problem)
    models["TasteNet"] = tastenet.fitter(dataset, inputs)
    models["ResLogit"] = reslogit.fitter(dataset, inputs)
    models["Latent class (2)"] = latent_class.fitter(dataset, inputs)
    if importlib.util.find_spec("rumboost") is not None:
        models["RUMBoost"] = rumboost_benchmark.fitter(dataset, inputs)
    features = inputs.columns
    models["Logistic regression"] = SklearnFitter(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                                  problem, features, piecewise_constant=False)
    models["MLP"] = SklearnFitter(make_pipeline(StandardScaler(), MLPClassifier((64, 64), early_stopping=True, max_iter=500,
                                                                                random_state=0)),
                                  problem, features, piecewise_constant=False)
    models["Random forest"] = SklearnFitter(RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=-1),
                                            problem, features, piecewise_constant=True)
    models["Gradient boosting"] = SklearnFitter(HistGradientBoostingClassifier(random_state=0), problem, features,
                                                piecewise_constant=True)
    return models


def main():
    parser = argparse.ArgumentParser(description="DCMBench benchmark")
    parser.add_argument("--datasets", nargs="+", choices=common.DATASETS, default=list(common.DATASETS))
    parser.add_argument("--models", nargs="+", help="subset of model names (default: all)")
    parser.add_argument("--protocol", choices=("cv", "holdout"), default="cv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    protocol = CrossValidation(n_splits=5) if args.protocol == "cv" else Holdout()

    all_cards = []
    for name in args.datasets:
        dataset = common.benchmark_dataset(name)
        models = model_families(dataset)
        if args.models:
            models = {k: v for k, v in models.items() if k in args.models}
        car = common.CAR[name]
        start = time.time()
        comparison = compare(models, dataset, protocol=protocol, scenarios=[common.car_cost_scenario(name)],
                             keep_vot_samples=True, sensitivity=("cost", [car], MULTIPLIERS))
        comparison.save(out)
        all_cards += list(comparison.cards().values())
        # Parameter-based (Route A) VOT distributions; probability-derivative VOTs of ML models are on their cards.
        route_a = {k: r for k, r in comparison.results.items() if r["vot"].settings["route"].startswith("A")}
        plot_vot(route_a, car).savefig(out / f"vot_{name}.png", dpi=150, bbox_inches="tight")
        plot_sensitivity(comparison.results, car).savefig(out / f"sensitivity_{name}.png", dpi=150, bbox_inches="tight")
        print(f"{name}: {len(models)} models in {time.time() - start:.0f}s", flush=True)
        print(comparison.table()[["fit_ll", "oos_avg_ll", "oos_macro_f1", "vot_route", f"vot_median[{car}]",
                                  f"regularity_strong_cost[{car}]", f"pct_change[{car}]", "n_flags"]].round(3).to_string(),
              flush=True)
    if all_cards:
        plot_fit_vs_prediction(all_cards).savefig(out / "fit_vs_prediction.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
