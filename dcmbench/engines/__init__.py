"""
Engines estimate models; DCMBench evaluates them.

- ``biogeme.BiogemeEstimator`` and ``torch.TorchEstimator`` estimate a spec v2 and return a ``SpecModel``.
- ``torch_module.TorchModuleFitter`` trains any PyTorch utility module (e.g. TasteNet, ResLogit, latent class).
- ``rumboost.RumboostFitter`` wraps RUMBoost (gradient-boosted utilities; runs in a worker process).
- ``sklearn.SklearnFitter`` wraps any classifier with ``predict_proba``.
- ``precomputed.PrecomputedModel`` scores probabilities produced elsewhere (e.g. Apollo).

Engine modules import their libraries lazily, so only the ones you use need installing.
"""
