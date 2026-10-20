"""
Engines estimate models; DCMBench evaluates them.

- ``BiogemeEstimator`` and ``TorchEstimator`` estimate a spec v2 and return a ``SpecModel``.
- ``SklearnFitter`` wraps any classifier with ``predict_proba``.
- ``PrecomputedModel`` scores probabilities produced elsewhere (e.g. Apollo).

Engine modules import their libraries lazily, so only the ones you use need installing.
"""
