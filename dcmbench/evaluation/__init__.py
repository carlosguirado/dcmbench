"""
Evaluation from probabilities alone: fit, prediction, elasticities, VOT,
policy scenarios and behavioral regularity.

    result = evaluate(fitter, dcmbench.data.load("swissmetro").prepare("benchmark"),
                      name="mnl", protocol=CrossValidation(n_splits=5),
                      scenarios=[Scenario("car cost x1.5", [Perturbation("cost", ("car",), multiply=1.5)])])
"""

from dcmbench.evaluation.compare import Comparison, compare
from dcmbench.evaluation.elasticity import point_elasticities
from dcmbench.evaluation.perturb import Perturbation, shift
from dcmbench.evaluation.predict import NonDeterministicModelError, check_deterministic, predict
from dcmbench.evaluation.regularity import behavioral_regularity
from dcmbench.evaluation.run import ANALYSES, Bootstrap, CrossValidation, Holdout, evaluate
from dcmbench.evaluation.scenarios import Scenario, market_shares, run_scenario, sensitivity
from dcmbench.evaluation.vot import vot_from_marginal_utilities, vot_from_probabilities, weighted_quantiles

__all__ = [
    "ANALYSES",
    "Bootstrap",
    "Comparison",
    "CrossValidation",
    "Holdout",
    "NonDeterministicModelError",
    "Perturbation",
    "Scenario",
    "behavioral_regularity",
    "check_deterministic",
    "compare",
    "evaluate",
    "market_shares",
    "point_elasticities",
    "predict",
    "run_scenario",
    "sensitivity",
    "shift",
    "vot_from_marginal_utilities",
    "vot_from_probabilities",
    "weighted_quantiles",
]
