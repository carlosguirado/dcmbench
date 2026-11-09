# Changelog

All notable changes to DCMBench are documented here.

## [1.0.0]

A rewrite around one idea: **a model only has to supply choice probabilities**.
DCMBench computes fit, prediction, elasticities, VOT, scenarios and
regularity itself, identically for every model class. The 0.1.x API is
replaced (see "Migrating from 0.1.x" below).

### Added

- **`dcmbench.data`.** Versioned, checksummed datasets, each described by a `ChoiceProblem` (alternatives, choice codes, availability, attributes with units, respondent column).
  - Explicit, recorded preparation steps, and named sample definitions (`prepare("benchmark")`).
  - `from_frame` for your own data.
  - Each dataset records its source, terms of use and citation.
- **Spec v2 (`dcmbench.specs`).** One JSON specification for MNL, nested logit and mixed logit that both Biogeme and PyTorch can estimate.
  - Mixed logit: normal, lognormal, uniform and triangular random parameters; pseudo-random or Halton draws; panel structure.
  - DCMBench generates the draws, so engines agree exactly.
  - Formulas are parsed safely, never `eval`'d.
- **Engines (`dcmbench.engines`).**
  - `BiogemeEstimator` (Biogeme ≥ 3.3; writes no files; checks Biogeme's log-likelihood against its own).
  - `TorchEstimator` (L-BFGS-B, autograd Hessian, robust standard errors).
  - `TorchModuleFitter`, for any PyTorch module, including latent class via `Mixture`.
  - `SklearnFitter`, `RumboostFitter` (worker process) and `PrecomputedModel`.
- **`dcmbench.evaluation`.** `evaluate()` and `compare()`.
  - Holdout, respondent-grouped cross-validation, bootstrap.
  - Point elasticities.
  - VOT by Route A (marginal utilities; population or posterior, panel-aware) or Route B (probability derivatives), always labelled.
  - Policy scenarios with diversion and arc elasticities, and sensitivity curves.
  - The behavioural regularity checks of Feng et al. (2024).
  - Every model's probability table is checked (shape, sums, availability, determinism).
- **`dcmbench.reporting`.** Specification cards (prototype JSON Schema 0.1.0, plus Markdown), comparison tables, and VOT / sensitivity / fit-vs-prediction figures.
- **Examples.** Dataset-agnostic TasteNet-MNL, ResLogit, latent class, RUMBoost, scikit-learn baselines and an MNL with covariates. Each hybrid has a fidelity check that reproduces its original paper's result. `examples/benchmark.py` runs everything.
- **Documentation.** User guide, adding a model, adding a dataset, specification cards.

### Changed

- **Requirements.** Python ≥ 3.12. Biogeme (≥ 3.3) and PyTorch are optional extras (`dcmbench[biogeme]`, `dcmbench[torch]`, `dcmbench[all]`).
- **LPMC** now uses the full file (81,086 trips). The 0.1.x 5,000-trip subsample is `prepare("sample_5000")`.
- **Benchmark specs (`examples/specs/`) correct the 0.1.x specs:**
  - nest parameters are bounded to [1, 10], the Biogeme normalization;
  - the Swissmetro business indicator uses `PURPOSE == 3`;
  - income enters through bracket dummies (bracket 4 is "unknown", not a level);
  - mixed logits use 1,000 Halton draws, with panel structure where the data are panel.
- **`dcmbench.validation`** (internal validation, contributed in 0.1.3) is kept with its API. `BiogemeSpecFitter` is rebuilt on the new Biogeme engine.

### Removed

- `dcmbench.datasets`, `dcmbench.adapters`, `dcmbench.model_benchmarker` and `dcmbench.utils`, all replaced as described below.
- ASC calibration in policy analysis. Models estimated with a full set of constants reproduce sample shares at their maximum-likelihood estimates.

### Fixed (relative to 0.1.x)

- **Nested-logit policy simulations.** In 0.1.x, `PolicyAnalyzer` produced baseline shares that did not match the estimated model (e.g. a Swissmetro train share of 9.5% where Biogeme's own simulation of the same model gives 13.2%). ASC calibration then masked the mismatch and distorted the substitution pattern. 1.0 computes scenario shares from the model's own probabilities.

### Migrating from 0.1.x

| 0.1.x | 1.0 |
|---|---|
| `dcmbench.datasets.fetch_data("swissmetro_dataset")` | `dcmbench.data.load("swissmetro").prepare("benchmark")` (`ltds_dataset` is now `lpmc`) |
| JSON specs with numbered alternatives | Spec v2. Convert old ones with `dcmbench.specs.v1.convert(old_spec)` |
| `UniversalBiogemeAdapter`, `PyTorchModelAdapter` | `BiogemeEstimator`, `TorchEstimator`, `TorchModuleFitter`, or any object implementing the `FittedModel` protocol |
| `SimpleBenchmarker` | `dcmbench.evaluation.compare` |
| `PolicyAnalyzer` (scenarios, sensitivity, ASC calibration) | `evaluate(..., scenarios=[Scenario(...)], sensitivity=(attribute, alternatives, multipliers))` |
| VOT helpers | The `vot` analysis of `evaluate` (Route A or B) |
| `dcmbench.validation` | Unchanged API |

## [0.1.3]: merged on GitHub, not published to PyPI

### Added

- **Internal validation** via `dcmbench.validation`:
  - `InternalValidator.cross_validate`: k-fold CV with re-estimation per fold
  - `InternalValidator.bootstrap`: prediction-error bootstrap (OOB + 0.632)
  - `fitter_from_spec` / `BiogemeSpecFitter` for Biogeme JSON specs (MNL / NL)
  - `FunctionModelFitter` for hand-built models
  - Tutorials and unit tests

### Fixed

- Nested logit OOS prediction now uses the nest structure (previously scored with MNL formulas).

## [0.1.2]

Initial public release.
