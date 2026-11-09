"""
Spec v2: engine-neutral model specifications.

    spec = load_spec("swissmetro_mnl.json")
    fitted = BiogemeEstimator(spec, dataset.problem)(dataset.frame)   # or TorchEstimator
"""

from dcmbench.specs.expression import ExpressionError, parse
from dcmbench.specs.model import SpecModel
from dcmbench.specs.spec import Draws, Nest, Parameter, RandomParameter, Spec, SpecError, load_spec

__all__ = [
    "Draws",
    "ExpressionError",
    "Nest",
    "Parameter",
    "RandomParameter",
    "Spec",
    "SpecError",
    "SpecModel",
    "load_spec",
    "parse",
]
