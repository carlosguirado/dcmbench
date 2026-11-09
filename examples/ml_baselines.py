"""
Machine-learning baselines (scikit-learn) on the three benchmark datasets.

Classifiers see the alternatives' times and costs and the individual
characteristics as plain features. DCMBench masks unavailable alternatives
and renormalizes their probabilities, reports ρ̄² as not applicable (no
meaningful parameter count) and flags tree models as piecewise constant.
VOT comes from probability derivatives (Route B) for every model.

    python examples/ml_baselines.py [--protocol cv] [--datasets lpmc]
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from dcmbench.engines.sklearn import SklearnFitter
from dcmbench.evaluation import evaluate

MODELS = {
    "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True,
                                                                 max_iter=500, random_state=0)),
    "random forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1),
    "gradient boosting": lambda: HistGradientBoostingClassifier(random_state=0),
}
# Pipelines hide the estimator class, so mark the tree models explicitly.
PIECEWISE = {"random forest": True, "gradient boosting": True, "logistic regression": False, "MLP": False}


def main():
    protocol, datasets = common.protocol_from_args(__doc__.splitlines()[1])
    for name in datasets:
        dataset = common.benchmark_dataset(name)
        features = common.benchmark_inputs(dataset).columns
        for label, build in MODELS.items():
            fitter = SklearnFitter(build(), dataset.problem, features, piecewise_constant=PIECEWISE[label])
            result = evaluate(fitter, dataset, name=label, protocol=protocol, vot_route="B",
                              scenarios=[common.car_cost_scenario(name)])
            common.report(result)


if __name__ == "__main__":
    main()
