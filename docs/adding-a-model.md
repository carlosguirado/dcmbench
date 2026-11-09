# Adding a model

DCMBench never needs to know how a model works inside. It needs **choice
probabilities**, and optionally **marginal utilities** for parameter-based VOT.
Pick the route that matches what you have.

## A. The model fits a spec: write a spec

If it is an MNL, nested logit or mixed logit, write it as a spec v2 (see the
[user guide](guide.md#2-specifications-spec-v2)). Then use
`BiogemeEstimator(spec, problem)` or `TorchEstimator(spec, problem)`.
Standard errors, Route A VOT and posteriors come for free, and both engines
give the same answer.

## B. A PyTorch module

Write an `nn.Module` whose `forward(x, available)` returns utilities `(n, J)`.
Here `x` is a dict of 1-D tensors, one per column you list, and `available`
is an `(n, J)` boolean tensor. `TorchModuleFitter` handles batching,
standardization, early stopping, availability masking and marginal utilities
(by autograd, in raw data units).

```python
import torch
from torch import nn
from dcmbench.engines.torch_module import TorchModuleFitter, Training


class LinearInTimeAndCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.asc = nn.Parameter(torch.zeros(2))     # train, car (sm is the reference)
        self.beta = nn.Parameter(torch.zeros(2))    # time, cost

    def forward(self, x, available):
        time = torch.stack([x["TRAIN_TT"], x["SM_TT"], x["CAR_TT"]], dim=1) / 100   # one scale for all alternatives
        pays = 1 - x["GA"]                          # travel-pass (GA) holders pay no train / Swissmetro fare
        cost = torch.stack([x["TRAIN_CO"] * pays, x["SM_CO"] * pays, x["CAR_CO"]], dim=1) / 100
        asc = torch.stack([self.asc[0], torch.zeros(()), self.asc[1]])
        return asc + self.beta[0] * time + self.beta[1] * cost


fitter = TorchModuleFitter(LinearInTimeAndCost, problem,
                           columns=["TRAIN_TT", "SM_TT", "CAR_TT", "TRAIN_CO", "SM_CO", "CAR_CO", "GA"],
                           # A plain likelihood model: full batch, all data, to convergence (no early-stopping split).
                           training=Training(optimizer="lbfgs", learning_rate=1.0, batch_size=None,
                                             validation_fraction=0.0, max_epochs=100, patience=3))
```

This reproduces the spec MNL exactly (log-likelihood −5,331 on the Swissmetro
benchmark sample).

**Scaling and generic coefficients.** `standardize=` and `rescale=` scale each
listed column by its own statistics. That suits inputs to a network, such as
individual characteristics. It is wrong for attribute columns that share a
generic coefficient: `TRAIN_TT`, `SM_TT` and `CAR_TT` would each be divided by
a different number, so one coefficient would mean three different things. Scale
those by one shared constant inside the module, as above (`examples/common.py`
does it with `ModelInputs.time_scale` / `cost_scale`).

For a **latent class** model, return
`Mixture(class_utilities (n, K, J), class_log_weights (n, K))`. DCMBench then
treats the classes as a discrete taste distribution: VOT per class, with
posterior class membership.

Worked examples, all dataset-agnostic, are in [`examples/`](../examples/):
- [TasteNet-MNL](../examples/tastenet.py);
- [ResLogit](../examples/reslogit.py);
- [latent class](../examples/latent_class.py).

## C. A scikit-learn style classifier

```python
from dcmbench.engines.sklearn import SklearnFitter
fitter = SklearnFitter(my_estimator, problem, features=[...], piecewise_constant=False)
```

Set `piecewise_constant=True` for tree models. Elasticities, VOT and
regularity then use larger steps and are flagged as approximate.

## D. Probabilities computed elsewhere (Apollo, Stata, R, ...)

```python
from dcmbench.engines.precomputed import PrecomputedModel
model = PrecomputedModel(probabilities_table, problem, key="obs_id", n_params=7)
result = evaluate(model, dataset, name="Apollo MNL", analyses=())
```

Fit and prediction work. Behavioural analyses need a model that can respond
to changed data, so they are not available.

## E. Anything else: the protocol

A fitted model is any object with:

| Member | Returns |
|---|---|
| `predict_probabilities(data)` | DataFrame, one column per alternative (in `problem.alternatives` order), rows summing to 1, zeros for unavailable alternatives |
| `n_params` (property) | number of estimated parameters, or `None` when not meaningful |
| `null_loglikelihood(data)` | log-likelihood of equal shares over available alternatives |
| `converged` (property) | whether estimation converged |

Optional capabilities are detected at runtime:
- **`marginal_utilities(data, attributes) -> MarginalUtilities`** enables Route A VOT and posteriors. See `dcmbench.protocols.MarginalUtilities` for the array layout: observations × draws (or classes) × alternatives × attributes, with weights.
- **`coefficients() -> DataFrame`** puts parameter tables on the card.

A **fitter** is a callable `train_data -> fitted model`. Pass fitters, not
fitted models, to cross-validation and bootstrap, which re-estimate on each
fold.

DCMBench checks every probability table it receives: shape, column order,
finite values, rows summing to 1, and zeros on unavailable alternatives. It
also checks that predictions are deterministic. A model that violates any of
these fails loudly instead of producing quiet nonsense.
