# Examples

Each script evaluates one model family on the three benchmark datasets (Swissmetro, ModeCanada and LPMC) with `dcmbench.evaluation.evaluate()`, and prints one summary line per dataset. Run them from the repository root:

```
python examples/tastenet.py                 # --protocol cv for 5-fold respondent-grouped CV
python examples/reslogit.py
python examples/latent_class.py             # --classes 3, --unconstrained
python examples/ml_baselines.py             # logistic regression, MLP, random forest, gradient boosting
PYTHONPATH=third_party/rumboost python examples/rumboost_benchmark.py
```

The default protocol is an 80/20 holdout stratified on the choice. It splits by observation, so a respondent's choices can fall on both sides. For panel data (Swissmetro has 9 choices per respondent), use `--protocol cv`, which keeps respondents together; see "Respondent leakage" below.

## The full benchmark

```
PYTHONPATH=third_party/rumboost python examples/benchmark.py --out results
```

`benchmark.py` runs every model family on each dataset under 5-fold, respondent-grouped cross-validation. The families are:
- the specs in `specs/`: MNL, NL, MXL, CUSTOM;
- TasteNet, ResLogit, latent class;
- RUMBoost;
- logistic regression, MLP, random forest, gradient boosting.

It writes to `results/`:
- a specification card per model (JSON and Markdown);
- a comparison table per dataset (CSV);
- VOT distribution, car-cost sensitivity and fit-vs-prediction figures.

## On your own data

Every model here works on any dataset. Wrap your table, describe it once, and benchmark:

```python
import dcmbench.data
from dcmbench import Attribute, ChoiceProblem
from dcmbench.evaluation import CrossValidation, evaluate
from common import ModelInputs
import tastenet, reslogit

problem = ChoiceProblem(name="my survey", alternatives=("bus", "car", "bike"), choice_column="mode",
                        choice_codes={1: "bus", 2: "car", 3: "bike"}, availability={"car": "car_av"},
                        attributes={"time": Attribute({"bus": "bus_tt", "car": "car_tt", "bike": "bike_tt"}, unit="minute", kind="time"),
                                    "cost": Attribute({"bus": "bus_fare", "car": "car_cost"}, unit="EUR", kind="cost")},
                        group_column="person_id")
dataset = dcmbench.data.from_frame(my_table, problem, name="my survey")
inputs = ModelInputs.from_dataset(dataset, reference="car", characteristics=["age", "income"])

for name, fitter in {"TasteNet": tastenet.fitter(dataset, inputs), "ResLogit": reslogit.fitter(dataset, inputs)}.items():
    result = evaluate(fitter, dataset, name=name, protocol=CrossValidation(n_splits=5))
```

Specs (`BiogemeEstimator`, `TorchEstimator`), `RumboostFitter` and `SklearnFitter` take the same `problem`, so all of them are evaluated under identical splits and metrics. `tests/test_examples_own_data.py` runs this path on a user-supplied table.

## Benchmark specifications

[`specs/`](specs/) holds the spec v2 files for the classical benchmark models:

| Dataset | Models |
|---|---|
| Swissmetro | MNL, NL, MXL, CUSTOM |
| ModeCanada | MNL, NL, MXL |
| LPMC | MNL, NL, MXL |

They differ from the 0.1.x specifications (kept in `tests/fixtures/v0_1_baseline/specs/`) in four ways:
- nest parameters are bounded to [1, 10];
- the CUSTOM business effect uses `PURPOSE == 3`;
- CUSTOM income enters through bracket dummies;
- mixed logits use 1,000 Halton draws, with panel structure for Swissmetro and LPMC.

Estimate them with `BiogemeEstimator` or `TorchEstimator`.

## Hybrid models

| Model | How it plugs in | Code |
|---|---|---|
| TasteNet-MNL (Han et al., 2022) | PyTorch module via `TorchModuleFitter`; per-person tastes from a network | [`tastenet.py`](tastenet.py) |
| ResLogit (Wong and Farooq, 2021) | PyTorch module via `TorchModuleFitter`; residual layers over utilities | [`reslogit.py`](reslogit.py) |
| RUMBoost (Salvadé and Hillel, 2025) | `dcmbench.engines.rumboost.RumboostFitter` | [`rumboost_benchmark.py`](rumboost_benchmark.py) |
| Latent class | PyTorch module returning a `Mixture` | [`latent_class.py`](latent_class.py) |

The TasteNet and ResLogit modules are reimplementations, about 40 lines each; neither has an installable official package. RUMBoost is used from its own source: the PyPI release is outdated and does not install on Python 3.12, and the repository is not pip-installable. Set it up with:

```
git clone https://github.com/big-ucl/rumboost third_party/rumboost
git -C third_party/rumboost checkout 068070f
pip install lightgbm==4.5.0
```

### Fidelity checks

These scripts reproduce the original papers' own protocols and compare with their reported numbers:

| Script | Paper result | DCMBench |
|---|---|---|
| [`tastenet_fidelity.py`](tastenet_fidelity.py): Swissmetro, 70/15/15 split | test NLL 0.645 best / 0.652 mean (sd 0.008, 100 runs), accuracy 0.703 | 0.642 best / 0.650 mean (sd 0.008, 5 seeds), accuracy 0.70 |
| [`rumboost_fidelity.py`](rumboost_fidelity.py): LPMC, train on years 1–2, test on year 3 | test cross-entropy 0.6737 (RUMBoost-GBUV) | 0.6737 |

ResLogit's paper uses a different dataset (Mtl Trajet, without costs), so there is no direct check. Its architecture follows the authors' code: identity initialization, ASCs after the residual stack, RMSprop at learning rate 1e-3.

## What the examples show

- **Respondent leakage.** Under the default observation-level holdout, flexible models (TasteNet, gradient boosting, random forests, MLP) look far better than MNL on Swissmetro, because they recognize respondents seen in training. TasteNet's own validation split, which keeps respondents separate, shows it overfitting (NLL per observation 0.67 in training vs 0.91 in validation). The TasteNet paper's protocol also split by observation.
- **Behavioral regularity catches wrong-signed models.** Two examples:
  - An unconstrained latent class model on ModeCanada, or a multinomial logistic regression on Swissmetro, can predict that raising car cost *increases* car's share. Strong cost regularity for those alternatives is then 0.
  - RUMBoost with monotone constraints never violates the weak form. Its step-function utilities often fail the strong form because the probability doesn't change at all.
- **Not every model has a VOT distribution.**
  - Linear-in-attributes models (MNL, ResLogit, latent class) give β_time/β_cost, per class where there are classes.
  - TasteNet gives a per-person distribution.
  - Tree models give degenerate probability-derivative VOTs. These results are flagged.
- **Degenerate latent classes on ModeCanada.** Two classes split into a time-only and a cost-only class: time and cost are confounded in that dataset, since air is both fast and expensive.
