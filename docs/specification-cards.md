# Specification cards

A specification card summarises **what one model specification implies on one
dataset**. It is a machine-readable record of fit, out-of-sample prediction,
economic valuations, policy response and behavioural regularity, together with
the provenance needed to reproduce it.

> **Status: prototype (card schema 0.1.0).** The format is a proposal for
> community discussion. Fields will change. Feedback is welcome in the issue
> tracker.

## Making cards

```python
from dcmbench.evaluation import compare
from dcmbench.reporting import build_card, render_markdown, save_card, load_card, compare_cards

comparison = compare(models, dataset, protocol=CrossValidation(n_splits=5), scenarios=[car_cost])
comparison.save("results/")                          # <dataset>_<model>.card.json and .card.md, plus a CSV table

card = build_card(result)                            # from a single evaluate() result
save_card(card, "mnl.card.json")                     # validates against the schema first
print(render_markdown(card))
compare_cards([load_card(p) for p in paths])         # side-by-side DataFrame
```

## Sections

| Section | Contents |
|---|---|
| `card_version` | Schema version |
| `model` | Name, engine, spec hash (for spec models), engine details (e.g. optimizer, engine version, iterations) |
| `data` | Dataset, data version and SHA-256, citation, number of observations, every preparation step with row counts |
| `protocol` | Prediction protocol and its settings (folds, grouping), where the behavioural analyses were run, seeds |
| `software` | Python, DCMBench, numpy, pandas and scikit-learn versions |
| `fit` | Observations, log-likelihood, null log-likelihood, number of parameters, ρ², ρ̄², convergence, quantities not applicable, notes (e.g. how a panel likelihood is reported) |
| `prediction` | Out-of-sample metrics with their spread across folds |
| `value_of_time` | Route (A-population, A-posterior, B), currency per hour, optional upper bound, distribution summary per alternative |
| `policy` | Each scenario: share changes, diversion, arc elasticities |
| `behavioral_regularity` | Settings, then strong and weak regularity for each attribute and alternative pair |
| `flags` | Everything the reader should know: approximations, quantities that are not applicable and why, deviations from reference definitions |

The JSON Schema ships with the package (`dcmbench.reporting.card.schema()`),
and `validate_card()` checks a card against it.

## Design choices

- **Probability-based and model-agnostic.** Every number on a card is computed by DCMBench from the model's probabilities (and marginal utilities when available), never taken from the model's own report. Cards of different model classes are therefore comparable.
- **No silent gaps.** When a quantity cannot be computed, the card says "not applicable" and gives the reason. An empty field would be indistinguishable from an omission.
- **Reproducible by construction.** Data hash, preparation steps, seeds, folds and software versions are recorded, so a card can be regenerated, or a disagreement traced.
