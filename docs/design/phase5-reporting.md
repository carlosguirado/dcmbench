# Phase 5: specification cards, comparisons and figures

Status: **implemented 2026-10-20** (`dcmbench/reporting/`, `dcmbench/evaluation/compare.py`). Section numbers like "Plan §3.5" refer to `docs/PLAN.md`.

## Specification card (prototype, `card_version` 0.1.0)

`build_card(result)` turns an `EvaluationResult` into a plain JSON card, validated against `dcmbench/reporting/card_schema.json` (JSON Schema 2020-12). It is a strawman for community discussion of reporting conventions, not a standard. Its sections:

| Section | Contents |
|---|---|
| `model` | name, engine, spec hash (spec models), engine details (versions, optimizer, training report) |
| `data` | dataset, data version, sha256 of the file or table, citation, sample size, every preparation step |
| `protocol` | prediction protocol and its settings (holdout, CV or bootstrap, seeds, grouping); which data the behavioral analyses used |
| `software` | Python, dcmbench, numpy, pandas, scikit-learn versions |
| `fit` | in-sample LL, null LL, parameters, ρ², ρ̄² (or `null` with a reason in `not_applicable`), convergence |
| `prediction` | out-of-sample metrics (avg LL, ρ², ρ̄², macro F1, macro recall, accuracy, Brier), each as estimate, spread and how it was obtained |
| `value_of_time` | route (A-population, A-posterior, B), currency per hour, per-alternative mean, sd, 5/25/50/75/95% quantiles, share negative, share above a plausibility bound |
| `policy` | each scenario: perturbations, shares before and after, % change, diversion ratios, arc elasticities |
| `behavioral_regularity` | strong and weak regularity per (alternative, own attribute), labeled `feng2024` or `extension`, with counts |
| `flags` | every caveat raised during evaluation, by section |

**How `prediction` estimates are obtained.**
- **Holdout:** the value on the test data.
- **CV:** the mean over folds, with the sd as spread.
- **Bootstrap:** likelihood metrics are out-of-bag means with their sd; classification metrics are .632 estimates.

**Using cards.** `save_card`, `load_card` and `validate_card` read, write and check cards, and `render_markdown` gives a human-readable view. `compare_cards(cards)` tabulates any collection of cards side by side, including cards from different datasets or studies.

## Comparisons

`compare({name: model_or_fitter, ...}, dataset, protocol=CrossValidation(), **options)` evaluates every model under the same protocol object. The protocol's seed fixes the folds (respondent-grouped by default), so all models are trained and scored on identical splits.
- **`Comparison.table()`** gives one row per model.
- **`Comparison.save(directory)`** writes a card (JSON and Markdown) per model and the table (CSV).

The headline benchmark protocol is grouped 5-fold CV (decision log, 2026-10-20).

`evaluate()` gained two options:
- `keep_vot_samples=True` stores the individual VOT values that distribution plots need;
- `sensitivity=(attribute, alternatives, multipliers)` adds a share curve.

## Figures (`dcmbench/reporting/plots.py`)

- **`plot_vot(results, alternative)`** shows VOT distributions across models, generalizing the 0.1.x density comparison:
  - point values as spikes;
  - discrete distributions (segments, latent classes) as weighted spikes;
  - continuous ones as weighted densities, with the share outside the plotted range reported.
- **`plot_sensitivity(results, alternative)`** shows predicted share curves along an attribute multiplier, generalizing the 0.1.x car-cost analysis.
- **`plot_fit_vs_prediction(cards)`** plots in-sample against out-of-sample log-likelihood per observation, with colour for dataset and marker for model.

## The benchmark runner

`examples/benchmark.py` runs every model family on each dataset under grouped 5-fold CV, then writes the cards, comparison tables and figures. The model families are:
- MNL, NL, MXL and CUSTOM specs (PyTorch engine);
- TasteNet, ResLogit, latent class;
- RUMBoost (when its source is available);
- logistic regression, MLP, random forest and gradient boosting.

The Phase 7 scripts build on it.
