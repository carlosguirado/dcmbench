"""
DCMBench: evaluate discrete choice models from their choice probabilities.

- ``dcmbench.data``: versioned datasets with their choice-problem descriptions
- ``dcmbench.specs``: spec v2 model specifications
- ``dcmbench.engines``: Biogeme, PyTorch, scikit-learn and precomputed models
- ``dcmbench.evaluation``: fit, prediction, elasticities, VOT, scenarios, regularity
- ``dcmbench.validation``: cross-validation and bootstrap
- ``dcmbench.reporting``: specification cards, comparison tables and figures
"""

from importlib.metadata import PackageNotFoundError, version

from dcmbench import data, engines, evaluation, reporting, specs, validation
from dcmbench.problem import Attribute, ChoiceProblem

try:
    __version__ = version("dcmbench")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "unknown"

__all__ = ["Attribute", "ChoiceProblem", "__version__", "data", "engines", "evaluation", "reporting", "specs",
           "validation"]
