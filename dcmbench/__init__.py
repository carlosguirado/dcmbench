"""
DCMBench: evaluate discrete choice models from their choice probabilities.

- ``dcmbench.data``: versioned datasets with their choice-problem descriptions
- ``dcmbench.specs``: spec v2 model specifications
- ``dcmbench.engines``: Biogeme, PyTorch, scikit-learn and precomputed models
- ``dcmbench.evaluation``: fit, prediction, elasticities, VOT, scenarios, regularity
- ``dcmbench.validation``: cross-validation and bootstrap
"""

from dcmbench import data, engines, evaluation, specs, validation
from dcmbench.problem import Attribute, ChoiceProblem

__all__ = ["Attribute", "ChoiceProblem", "data", "engines", "evaluation", "specs", "validation"]
