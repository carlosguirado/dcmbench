# DCMBench

**Evaluate discrete choice models of any class on a common footing.**

DCMBench takes a fitted model's **choice probabilities** and computes the rest
itself, the same way for every model:
- goodness of fit;
- out-of-sample prediction under respondent-grouped cross-validation;
- elasticities;
- value of time (VOT) distributions;
- policy scenarios and diversion;
- behavioural regularity checks.

A multinomial logit estimated in Biogeme, a mixed logit in PyTorch, a TasteNet,
a latent class model, RUMBoost and a random forest can therefore be compared
side by side, on the same data, folds and definitions.

```
pip install dcmbench                 # core: data, evaluation, reporting, scikit-learn models
pip install "dcmbench[biogeme]"      # + Biogeme engine
pip install "dcmbench[torch]"        # + PyTorch engines (spec models, any nn.Module)
pip install "dcmbench[all]"          # everything
```

Requires Python ≥ 3.12.

## Quick start

```python
import dcmbench.data
from dcmbench.engines.torch import TorchEstimator
from dcmbench.evaluation import CrossValidation, compare
from dcmbench.specs import load_spec

dataset = dcmbench.data.load("swissmetro").prepare("benchmark")

mnl = load_spec({
    "spec_version": 2, "name": "swissmetro_mnl", "model": "MNL",
    "alternatives": ["train", "sm", "car"],
    "parameters": {"ASC_TRAIN": {"start": 0}, "ASC_CAR": {"start": 0}, "B_TIME": {"start": 0}, "B_COST": {"start": 0}},
    "utilities": {
        "train": "ASC_TRAIN + B_TIME * TRAIN_TT / 100 + B_COST * TRAIN_CO * (GA == 0) / 100",
        "sm": "B_TIME * SM_TT / 100 + B_COST * SM_CO * (GA == 0) / 100",
        "car": "ASC_CAR + B_TIME * CAR_TT / 100 + B_COST * CAR_CO / 100",
    },
})

comparison = compare({"MNL": TorchEstimator(mnl, dataset.problem)}, dataset, protocol=CrossValidation(n_splits=5))
print(comparison.table()[["fit_ll", "oos_avg_ll", "oos_macro_f1", "vot_median[car]"]])
comparison.save("results/")   # a specification card per model (JSON + Markdown) and the comparison table
```

Add more models to the dictionary (any engine, any class) and they are
evaluated on identical folds.

## What you get for every model

| Dimension | What DCMBench computes |
|---|---|
| Fit | Log-likelihood, ρ², ρ̄² (per observation for panel models, so classes are comparable) |
| Prediction | Holdout, k-fold cross-validation grouped by respondent, or bootstrap: log-likelihood, accuracy, macro F1 / recall, Brier score |
| Elasticities | Point elasticities (own and cross) of time and cost |
| Value of time | **Route A** from the model's marginal utilities (population draws or individual posteriors); **Route B** from probability derivatives, for models without utilities. Every result says which route was used |
| Policy | Scenarios (e.g. car cost × 1.5): share changes, diversion, arc elasticities; sensitivity curves |
| Behavioural regularity | The strong and weak regularity checks of Feng et al. (2024), with any deviation from their definition flagged |
| Reporting | A **specification card** per model (JSON with a schema, plus Markdown), comparison tables, VOT / sensitivity / fit-vs-prediction figures |

Nothing falls back silently. When a quantity is not defined for a model, the
card records it as not applicable and says why. Examples: zero cost
sensitivity, a piecewise-constant tree model.

## Models and engines

| You have | Use |
|---|---|
| An MNL, nested logit or mixed logit written as a **spec** (JSON or dict) | `BiogemeEstimator` or `TorchEstimator`. The same spec runs on both, with identical draws |
| Any **PyTorch module** that returns utilities or probabilities (TasteNet, ResLogit, latent class, ...) | `TorchModuleFitter`. Marginal utilities come from autograd |
| A **scikit-learn** classifier | `SklearnFitter` |
| A **RUMBoost** model | `RumboostFitter` (install RUMBoost from source, see below) |
| Probabilities from **elsewhere** (Apollo, Stata, your own code) | `PrecomputedModel` |
| **Anything else** | Implement the four-method protocol below |

### Bring your own model

A fitted model only needs to return probabilities:

```python
import numpy as np
import pandas as pd


class EqualShares:
    """Every available alternative equally likely."""

    def __init__(self, problem):
        self.problem = problem

    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        available = self.problem.availability_matrix(data).astype(float)
        return available.div(available.sum(axis=1), axis=0)

    @property
    def n_params(self):
        return 0

    def null_loglikelihood(self, data):
        return float(-np.log(self.problem.availability_matrix(data).sum(axis=1)).sum())

    @property
    def converged(self):
        return True
```

Pass a fitter (a function from training data to a fitted model) to `evaluate`
or `compare`, e.g. `lambda train: EqualShares(dataset.problem)`. A model that
also implements `marginal_utilities` (see `dcmbench.protocols`) gets
Route A VOT and posterior distributions.

### Bring your own data

```python
import dcmbench.data
from dcmbench import Attribute, ChoiceProblem

problem = ChoiceProblem(
    name="my survey",
    alternatives=("bus", "car"),
    choice_column="choice",
    choice_codes={1: "bus", 2: "car"},
    attributes={
        "time": Attribute({"bus": "bus_time", "car": "car_time"}, unit="minute", kind="time"),
        "cost": Attribute({"bus": "bus_fare", "car": "car_cost"}, unit="USD", kind="cost"),
    },
    group_column="person_id",   # keeps each respondent in one fold
)
# my_dataset = dcmbench.data.from_frame(my_table, problem, name="my survey", citation="...")
```

Units are declared once, so VOT comes out in currency per hour whatever the
data's units.

## Built-in datasets

`dcmbench.data.load(name)` downloads a versioned copy, checks its SHA-256 and
caches it (`DCMBENCH_DATA_DIR`, default `~/.cache/dcmbench`). Every
preparation step is recorded on the dataset and on each card.

| Name | Data | Source and terms | Cite |
|---|---|---|---|
| `swissmetro` | Stated preference, intercity, 3 modes (6,768 obs. after `prepare("benchmark")`) | EPFL TRANSP-OR (Biogeme data sets), offered for research and education | Bierlaire, Axhausen & Abay (2001) |
| `modecanada` | Intercity, Toronto–Montreal corridor, 4 modes (4,324 obs.) | R package `mlogit` (GPL-2/3), data "kindly provided by S. Koppelman" | Bhat (1995) |
| `lpmc` | Revealed preference, London, 4 modes (81,086 trips; `prepare("sample_5000")` for the 0.1.x subsample) | EPFL TRANSP-OR (Biogeme data sets), offered for research and education | Hillel, Elshafie & Jin (2018) |

The datasets belong to their authors. DCMBench's MIT licence covers the code,
not the data. Please cite the original source when you use a dataset.

## Examples

[`examples/`](examples/) holds dataset-agnostic implementations of hybrid
models, written against the protocols above rather than shipped in the
library:
- TasteNet-MNL, ResLogit, latent class and RUMBoost, each with a check that reproduces its original paper's result;
- scikit-learn baselines;
- an MNL with the hybrids' covariates;
- `benchmark.py`, which runs all of them on the three datasets.

[`tutorials/`](tutorials/) covers internal validation (cross-validation and
bootstrap) with `dcmbench.validation`.

RUMBoost has no working PyPI release. Install it from source
(`pip install "dcmbench[rumboost]"` pins its LightGBM, then clone
[RUMBoost](https://github.com/big-ucl/rumboost) and put it on `PYTHONPATH`).
DCMBench runs it in a separate worker process, so its OpenMP runtime cannot
clash with PyTorch's.

## Documentation

- [User guide](docs/guide.md): datasets, specs, engines, evaluation, cards.
- [Adding a model](docs/adding-a-model.md) and [adding a dataset](docs/adding-a-dataset.md).
- [Specification cards](docs/specification-cards.md).
- [Changelog](CHANGELOG.md), including the migration notes from 0.1.x.

## Licence

MIT (code). See the table above for the datasets.
