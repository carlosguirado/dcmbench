# User guide

DCMBench has five parts. Each can be used on its own:

| Module | Role |
|---|---|
| `dcmbench.data` | Datasets and the `ChoiceProblem` that describes them |
| `dcmbench.specs` | Spec v2: a JSON model specification that several engines can estimate |
| `dcmbench.engines` | Fitters: Biogeme, PyTorch (specs or any module), scikit-learn, RUMBoost, precomputed probabilities |
| `dcmbench.evaluation` | `evaluate()` and `compare()`: fit, prediction, elasticities, VOT, scenarios, regularity |
| `dcmbench.reporting` | Specification cards, comparison tables, figures |

`dcmbench.validation` (cross-validation and bootstrap) is what
`evaluate()` uses for its prediction protocols. It can also be used directly
(see [`tutorials/`](../tutorials/)).

## 1. Data

```python
import dcmbench.data

dcmbench.data.list_datasets()              # ['lpmc', 'modecanada', 'swissmetro']
ds = dcmbench.data.load("swissmetro")      # download, verify SHA-256, cache
ds = ds.prepare("benchmark")               # a documented, named sample definition
ds.frame                                   # the table (pandas)
ds.problem                                 # ChoiceProblem: alternatives, choice codes, availability, attributes
ds.info.citation, ds.info.license          # where the data come from and on what terms
ds.provenance                              # every step from the downloaded file to this table
```

Loading never drops rows silently. Filters, subsamples and derived columns
are explicit and recorded:

```python
ds = ds.filter("PURPOSE == 1", label="commuters only")
ds = ds.subsample(2000, seed=1)
ds = ds.derive("business dummy", business=lambda f: (f.PURPOSE == 3).astype(int))
```

**The `ChoiceProblem`** is the contract between the data and every model and
analysis. It defines:
- the alternatives;
- how the choice column is coded;
- availability columns;
- attributes (e.g. `time`, `cost`), each mapping alternatives to columns, with a unit and a kind;
- the `group_column` identifying respondents.

Units matter: VOT is converted to currency per hour from the declared time
and cost units. The group column keeps each respondent's observations in one
fold.

## 2. Specifications (spec v2)

A spec is a JSON document (or a dict) that both `BiogemeEstimator` and
`TorchEstimator` can estimate:

```json
{
  "spec_version": 2,
  "name": "swissmetro_nl",
  "model": "NL",
  "alternatives": ["train", "sm", "car"],
  "parameters": {
    "ASC_TRAIN": {"start": 0}, "ASC_CAR": {"start": 0},
    "B_TIME": {"start": 0}, "B_COST": {"start": 0},
    "MU": {"start": 1.5, "lower": 1, "upper": 10}
  },
  "utilities": {
    "train": "ASC_TRAIN + B_TIME * TRAIN_TT / 100 + B_COST * TRAIN_CO * (GA == 0) / 100",
    "sm": "B_TIME * SM_TT / 100 + B_COST * SM_CO * (GA == 0) / 100",
    "car": "ASC_CAR + B_TIME * CAR_TT / 100 + B_COST * CAR_CO / 100"
  },
  "nests": [{"name": "existing", "parameter": "MU", "alternatives": ["train", "car"]}]
}
```

- **Model types:** `MNL`, `NL` (nest parameters ≥ 1, the Biogeme normalization) and `MXL`.
- **Formulas:** arithmetic, comparisons (giving 0/1), `log` and `exp`. Formulas are parsed into an expression tree and never `eval`'d. Names that are not parameters are data columns.
- **Mixed logit:** declare random parameters and draws:

  ```json
  "random_parameters": {"B_TIME_RND": {"distribution": "lognormal", "mean": "B_TIME", "spread": "B_TIME_S", "sign": -1}},
  "draws": {"number": 1000, "seed": 42, "type": "halton", "panel": true}
  ```

  - Distributions: `normal`, `lognormal` (with `sign`), `uniform`, `triangular`.
  - Draw types: `pseudo` or `halton`.
  - `panel: true` gives each respondent one draw across all their choices.
  - DCMBench generates the draws itself and hands the same draws to every engine, so engines agree exactly.

```python
from dcmbench.specs import load_spec
spec = load_spec("swissmetro_nl.json")     # or a dict
```

## 3. Engines

A **fitter** is any callable `train_data -> fitted model`. A **fitted model**
returns choice probabilities (see [Adding a model](adding-a-model.md)).

| Engine | Fitter | Notes |
|---|---|---|
| Biogeme ≥ 3.3 | `BiogemeEstimator(spec, problem)` | Writes no files; checks Biogeme's log-likelihood against DCMBench's own evaluation of the spec |
| PyTorch | `TorchEstimator(spec, problem)` | L-BFGS-B with autograd gradients and Hessian; robust (sandwich) standard errors |
| PyTorch module | `TorchModuleFitter(build, problem, columns, training=Training(...))` | Any `nn.Module` returning utilities, probabilities, or a `Mixture` (latent class) |
| scikit-learn | `SklearnFitter(estimator, problem, features)` | Probabilities of unavailable alternatives are masked and renormalized |
| RUMBoost | `RumboostFitter(...)` | Runs in a worker process (OpenMP isolation from PyTorch) |
| Anything | `PrecomputedModel(probabilities, problem, key)` | Probabilities computed elsewhere; fit and prediction only, since the model cannot respond to changed data |

## 4. Evaluation

```python
from dcmbench.evaluation import CrossValidation, Perturbation, Scenario, compare, evaluate

car_cost = Scenario("car cost x1.5", [Perturbation("cost", ("car",), multiply=1.5)])
result = evaluate(fitter, ds, name="MNL", protocol=CrossValidation(n_splits=5),
                  scenarios=[car_cost], sensitivity=("cost", ["car"], [0.5, 1.0, 1.5, 2.0]),
                  keep_vot_samples=True)
result.fit                       # log-likelihood, rho², n_params, convergence, notes
result["prediction"].summary     # out-of-sample metrics (mean and sd over folds)
result["vot"].summary            # VOT distribution per alternative, and the route used
result["regularity"].summary     # Feng et al. strong / weak regularity
```

**Protocols:**
- `Holdout(test_size=0.2, stratify=True | grouped=True)`;
- `CrossValidation(n_splits=5, grouped=True)`;
- `Bootstrap(n_draws=30)`.

Under cross-validation and bootstrap, the behavioural analyses use the
full-sample fit. Under a holdout they use the test part.

**Value of time:**
- **Route A** uses the model's marginal utilities: closed form for specs, autograd for PyTorch modules. It gives population distributions, or individual posterior means with `vot_posterior=True`; panel posteriors condition on all of a respondent's choices.
- **Route B** differentiates the probabilities, for models that expose no utilities (scikit-learn, RUMBoost). Piecewise-constant models use a larger step and are flagged.

**Regularity** follows Feng et al. (2024, Transportation Research Part C):
a forward difference with step range/200 and tolerance range/10⁶ on the
change in probability, for own cost. Time attributes are checked as an
extension, unavailable observations are excluded, and attributes that are
constant in the data are reported as not applicable. Every one of these
choices is listed in the result's flags.

**`compare(models, dataset, ...)`** runs `evaluate` for a dictionary of
fitters with identical folds, scenarios and settings. It returns a
`Comparison` with `.table()`, `.cards()` and `.save(directory)`.

## 5. Reporting

```python
from dcmbench.reporting import build_card, render_markdown, save_card, compare_cards
from dcmbench.reporting.plots import plot_fit_vs_prediction, plot_sensitivity, plot_vot

card = build_card(result)                  # a dict that follows the card JSON Schema
print(render_markdown(card))
plot_vot(comparison.results, "car")        # needs keep_vot_samples=True
plot_sensitivity(comparison.results, "car")
plot_fit_vs_prediction(list(comparison.cards().values()), groups={"MNL": "logit", "TasteNet": "hybrid"})
```

See [Specification cards](specification-cards.md).
