# Changelog

All notable changes to DCMBench are documented here.

## [0.1.3] — Unreleased

### Added

- **Internal validation** via `dcmbench.validation`:
  - `InternalValidator.cross_validate` — k-fold CV with re-estimation per fold
  - `InternalValidator.bootstrap` — prediction-error bootstrap (OOB + 0.632)
  - `fitter_from_spec` / `BiogemeSpecFitter` for Biogeme JSON specs (MNL / NL)
  - `FunctionModelFitter` for hand-built models (required for genuine MXL with `bioDraws`)
  - Tutorials: `tutorials/internal_validation_swissmetro.py`,
    `tutorials/internal_validation_mxl_swissmetro.py`
  - Unit tests in `tests/test_validation.py`

### Fixed

- Nested logit OOS prediction: `build_model_from_spec` / `UniversalBiogemeAdapter`
  now preserve nest structure and use `models.nested` for probabilities
  (previously estimated NL but scored with MNL logit formulas).

## [0.1.2] — Initial public release
