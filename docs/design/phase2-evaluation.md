# Phase 2: the evaluation engine

Status: **implemented 2026-10-20** (`dcmbench/evaluation/`). This note records the conventions behind the numbers. Section numbers like "Plan §5.2" refer to `docs/PLAN.md`.

## Entry point

```python
from dcmbench.evaluation import evaluate, CrossValidation, Holdout, Bootstrap, Scenario, Perturbation

result = evaluate(fitter, dataset, name="mnl", protocol=CrossValidation(n_splits=5),
                  scenarios=[Scenario("car cost x1.5", [Perturbation("cost", ("car",), multiply=1.5)])])
result.fit                                  # FitResult (in-sample)
result["prediction"].summary                # CV / bootstrap / holdout metrics
result["vot"].summary, result["regularity"].summary, ...
result.to_json()                            # every number, setting and flag
```

- **What `evaluate()` accepts.** A fitted model (`FittedModel`) or a fitter (`Callable[[DataFrame], FittedModel]`). Validation protocols need a fitter, because they re-estimate the model.
- **Which data each part uses:**

| Protocol | Fit | Prediction metrics | Behavioral analyses |
|---|---|---|---|
| none | full sample | — | full sample (in-sample) |
| `Holdout(test_size=0.2, seed=42, stratify=True)` | training part | test part | test part |
| `CrossValidation(n_splits, seed, grouped=True)` | full sample | `InternalValidator.cross_validate` | full sample (in-sample) |
| `Bootstrap(n_draws, seed, grouped=True)` | full sample | `InternalValidator.bootstrap` | full sample (in-sample) |

  Every analysis records which data it was computed on (`settings["evaluated_on"]`). `grouped=True` keeps a respondent's observations together, using `ChoiceProblem.group_column`.
- **Fit.** In-sample fit is computed by DCMBench from the model's probabilities. It does not rely on the log-likelihood the engine reports.
- **Analyses.** `analyses` selects among `"elasticities"`, `"vot"`, `"scenarios"` and `"regularity"`.

## Rules at the model boundary (`evaluation/predict.py`)
- **Every probability table is checked.** Every call goes through `predict()`, which enforces the probability-table contract. Models declaring `handles_availability = False` are masked and renormalized centrally.
- **Predictions must be deterministic.** `evaluate()` predicts twice on the same rows and raises `NonDeterministicModelError` if the results differ. Finite differences compare predictions on nearly identical data, so simulated models must reuse their draws.
- **Optional model attributes:**
  - `piecewise_constant = True` adds caveat flags to the derivative-based analyses.
  - `describe() -> dict` adds model details to provenance.

## Analyses

### Elasticities (`point_elasticities`)
- **Method.** Central differences with step `1e-4 · |x|`, giving E_nij = ∂P_ni/∂x_nj · x_nj / P_ni. Where x = 0, the elasticity is 0 by definition, and those observations are counted.
- **Exclusions.** Observations where either alternative is unavailable are excluded.
- **Aggregate.** Probability-weighted, Σ P_ni E_nij / Σ P_ni. The mean of the individual elasticities is also reported.

### VOT (`vot_from_marginal_utilities`, `vot_from_probabilities`)
- **Route A (`ProvidesMarginalUtilities`).** VOT = (∂V/∂time)/(∂V/∂cost) per observation and draw.
  - Weights are the model's prior weights (`A-population`), or posterior weights ∝ prior × P(observed choice | draw) (`A-posterior`).
  - For panel draws (`MarginalUtilities.panel = True`), the likelihood is multiplied within respondent.
- **Route B (any model).** VOT = (∂P_i/∂time_i)/(∂P_i/∂cost_i), by central differences. The step is `1e-4 ×` the attribute's range in the full dataset.
- **Selection.** `vot_route="auto"` uses A when the model supports it, otherwise B.
- **Units.** VOT is converted from the `ChoiceProblem` units to currency per hour.
- **Undefined VOT.** Where the marginal utility of cost, or the derivative with respect to cost, is zero (e.g. Swissmetro travel-pass holders), the VOT is dropped and counted in a flag.
- **Summary per alternative.** Each observation carries total weight 1. The summary reports the weighted mean, SD, the 5/25/50/75/95% quantiles, the share of negative values, and, given `upper_bound`, the share above a plausibility bound.
- **Quantiles.** The inverted weighted CDF, averaging at exact ties. This equals the median convention of the sample expanded by its weights.

### Scenarios (`Scenario`, `run_scenario`, `sensitivity`)
- **Shares.** Sample (or weighted) mean probabilities before and after the change.
- **Diversion.** Each alternative's gain as a fraction of the loss of the perturbed alternative.
- **Arc elasticity.** Midpoint arc elasticity, for single multiplicative changes.
- **No calibration.** No ASC calibration is applied: that is model-specific, and with a full set of ASCs an MNL already reproduces the sample shares.
- **Sensitivity curves.** `sensitivity()` gives the shares over a list of multipliers.

### Behavioral regularity (`behavioral_regularity`)
The definition and code conventions are those of Feng et al. (2024); see Plan §5.4.
- **Pairs.** Every (alternative, own time or cost attribute) pair. Own-cost pairs with the default step and tolerance are labeled `feng2024`; everything else is labeled `extension`.
- **Counts.** Each row records `n_excluded_unavailable` and `n_zero_change`.
- **Flags.** These cover excluded unavailable observations, time attributes, piecewise-constant models, and attributes the model never responds to.

## Metrics added to the validation module
`calculate_classification_metrics` (macro F1 and macro recall, by argmax) is part of `get_performance_metrics`. They therefore appear in CV folds, in the CV summaries (`macro_f1_mean`, ...), in the bootstrap apparent and OOB metrics, and in holdout results. Existing keys are unchanged.

## Verified on real data
A throwaway scipy MNL with the 0.1.x Swissmetro specification reproduces the 0.1.x baseline (`tests/fixtures/v0_1_baseline/`):
- In-sample LL −5331.25.
- Holdout: in-sample LL −4250.7, out-of-sample LL −1080.7, out-of-sample macro F1 0.46.
- Car-cost × 1.5 scenario: car share −24.5%, with diversion 18% to train and 82% to Swissmetro.
