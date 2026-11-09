# Phase 4: hybrid and machine-learning models

Status: **implemented 2026-10-20**. The models live in `examples/`; the library gained two engines. Section numbers like "Plan §6" refer to `docs/PLAN.md`; see also `examples/README.md`.

## New engines

**`TorchModuleFitter`** (`dcmbench/engines/torch_module.py`) trains any PyTorch module that maps data columns to utilities `(n, J)`, or to a `Mixture` of class-specific utilities `(n, K, J)` with membership log-weights `(n, K)`.
- **Inputs.** The module receives a dict of column tensors and the availability mask.
- **Standardization.** Columns listed in `standardize` are standardized inside the autograd graph.
- **Training:**
  - Adam, RMSprop or L-BFGS;
  - mini-batches or the full batch;
  - early stopping on a validation split (respondent-grouped when the problem has a group column), with the best weights restored;
  - or an explicit validation set via `fit(train, validation)`.
- **Marginal utilities** come from autograd with respect to the *raw* columns, per latent class for mixtures. So VOT Route A, and posterior VOTs for latent classes, work for any module.
- **Checks.** A linear-logit module trained through it reproduces the spec engine's probabilities to 10⁻⁵ (`tests/test_torch_module.py`).

**`RumboostFitter`** (`dcmbench/engines/rumboost.py`) builds one boosted ensemble per alternative, with one variable per tree and optional monotone constraints.
- **Process isolation.** LightGBM and PyTorch each load an OpenMP runtime, and on macOS, using both in one process segfaults or deadlocks, in either import order. Each fitted model therefore lives in its own worker subprocess (`_rumboost_worker.py`, pickled requests over stdin and stdout). The worker imports only numpy, pandas, lightgbm and RUMBoost.
- **Availability.** RUMBoost cannot use availability during training; predictions are masked afterwards (`handles_availability = False`), and `describe()` records this.
- **Installation.** RUMBoost must be checked out at commit `068070f` and placed on `PYTHONPATH` (see the module docstring).

## Evaluation changes made along the way

- **Log-space likelihood.** Spec likelihoods are now computed entirely in log space (log-softmax and log-mean-exp over draws, including panel products). Previously, extreme lognormal draws underflowed P(chosen) to 0, and gradients became NaN; that is what broke PyTorch on the LPMC panel MXL.
- **Constant attributes.** Behavioral regularity reports a constant attribute (e.g. ModeCanada car out-of-vehicle time) as not applicable, with a flag, instead of failing.
- **Determinism check.** It allows 10⁻¹² of floating-point noise (multi-threaded random forests), far below simulation noise.
- **Route B step for piecewise-constant models.** VOT Route B uses a step of 1/200 of the range for them, and records it.
- **Derived columns.** `Dataset.derive()` adds derived columns (e.g. dummies), with a provenance step.

## Fidelity

| Model | Check | Result |
|---|---|---|
| TasteNet-MNL | the paper's Swissmetro protocol (`examples/tastenet_fidelity.py`) | test NLL 0.650 mean / 0.642 best, against 0.652 / 0.645 in the paper; same sample and split sizes |
| RUMBoost | the paper's LPMC protocol (`examples/rumboost_fidelity.py`) | test cross-entropy 0.6737, against 0.6737 |
| ResLogit | architecture as in the authors' code | no comparable published dataset |

## Corrected benchmark specs

The corrected specs are in `examples/specs/`. With nest parameters bounded to [1, 10], the ModeCanada and LPMC nested logits converge to μ = 1, i.e. they reduce to their MNL. The panel mixed logits for Swissmetro and LPMC fit far better than the cross-sectional ones (Swissmetro LL −4358.9 against about −5221). Biogeme and PyTorch agree on all of them.
