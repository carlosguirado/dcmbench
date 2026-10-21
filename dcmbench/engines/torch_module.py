"""
Any PyTorch utility module as a choice model (``pip install dcmbench[torch]``).

The module maps data columns to utilities; DCMBench handles availability,
training, prediction and marginal utilities (by autograd, with respect to the
raw data columns). This is how hybrid models such as TasteNet or ResLogit
plug in.

Module contract::

    forward(x: dict[str, Tensor(n,)], available: BoolTensor(n, J)) -> Tensor(n, J)          # utilities
                                                                    or Mixture(utilities (n, K, J),
                                                                               log_weights (n, K))

A ``Mixture`` is a latent-class model: class-specific utilities and class
membership log-weights. Modules must work row by row (no batch statistics),
so predictions do not depend on which other rows are in the batch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Union

import numpy as np
import pandas as pd

from dcmbench.problem import ChoiceProblem
from dcmbench.protocols import MarginalUtilities

MASK = -1e100


class Mixture(NamedTuple):
    """Latent-class output: class-specific utilities ``(n, K, J)`` and membership log-weights ``(n, K)``."""

    utilities: object
    log_weights: object


@dataclass(frozen=True)
class Training:
    """
    Training settings. ``batch_size=None`` trains on the full batch.
    Early stopping monitors the log-likelihood on ``validation_fraction`` of
    the training data (respondent-grouped when the problem has a group
    column) and restores the best weights; ``patience`` is in epochs.
    """

    optimizer: str = "adam"  # adam | rmsprop | lbfgs
    learning_rate: float = 1e-3
    batch_size: Optional[int] = 64
    max_epochs: int = 200
    patience: int = 10
    weight_decay: float = 0.0
    validation_fraction: float = 0.15

    def __post_init__(self):
        if self.optimizer not in ("adam", "rmsprop", "lbfgs"):
            raise ValueError("optimizer must be 'adam', 'rmsprop' or 'lbfgs'")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")


def _log_probabilities(output, available, torch):
    """``(n, J)`` log-probabilities and, for mixtures, ``(n, K, J)`` per-class ones and ``(n, K)`` class weights."""
    if isinstance(output, Mixture):
        v = torch.where(available[:, None, :], output.utilities, torch.tensor(MASK, dtype=torch.float64))
        per_class = torch.log_softmax(v, dim=2)
        log_w = torch.log_softmax(output.log_weights, dim=1)
        return torch.logsumexp(per_class + log_w[:, :, None], dim=1), per_class, log_w
    v = torch.where(available, output, torch.tensor(MASK, dtype=torch.float64))
    return torch.log_softmax(v, dim=1), None, None


class TorchModuleModel:
    handles_availability = True

    def __init__(self, module, problem: ChoiceProblem, columns: Sequence[str], scaling: Dict[str, tuple],
                 n_params: Optional[int], report: Dict[str, object], name: str):
        self.module, self.problem, self.columns = module, problem, list(columns)
        self.scaling, self._n_params, self.report, self.name = scaling, n_params, report, name
        self.module.eval()

    # --- inputs -------------------------------------------------------------------------------------

    def _tensors(self, data: pd.DataFrame, requires_grad: bool = False):
        import torch

        raw = {c: torch.tensor(data[c].to_numpy(dtype=float), dtype=torch.float64, requires_grad=requires_grad)
               for c in self.columns}
        scaled = {c: (t - self.scaling[c][0]) / self.scaling[c][1] if c in self.scaling else t for c, t in raw.items()}
        available = torch.as_tensor(self.problem.availability_matrix(data).to_numpy())
        return raw, scaled, available

    # --- FittedModel ------------------------------------------------------------------------------

    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        import torch

        _, x, available = self._tensors(data)
        with torch.no_grad():
            log_p, _, _ = _log_probabilities(self.module(x, available), available, torch)
        p = torch.where(available, torch.exp(log_p), torch.zeros((), dtype=torch.float64)).numpy()
        return pd.DataFrame(p, columns=list(self.problem.alternatives), index=data.index)

    @property
    def n_params(self) -> Optional[int]:
        return self._n_params

    def null_loglikelihood(self, data: pd.DataFrame) -> float:
        return float(-np.log(self.problem.availability_matrix(data).sum(axis=1)).sum())

    @property
    def converged(self) -> bool:
        return bool(self.report["finite"])

    def describe(self):
        return {"engine": "pytorch module", "name": self.name, "module": repr(self.module), **self.report}

    # --- marginal utilities -----------------------------------------------------------------------

    def marginal_utilities(self, data: pd.DataFrame, attributes: Sequence[str]) -> MarginalUtilities:
        """∂U_j/∂x_j by autograd with respect to the raw columns (per latent class for mixtures)."""
        import torch

        raw, x, available = self._tensors(data, requires_grad=True)
        output = self.module(x, available)
        utilities = output.utilities if isinstance(output, Mixture) else output[:, None, :]  # (n, K, J)
        n, n_classes, n_alts = utilities.shape
        values = np.full((n, n_classes, n_alts, len(attributes)), np.nan)
        for k, attribute in enumerate(attributes):
            for j, alternative in enumerate(self.problem.alternatives):
                column = self.problem.attributes[attribute].columns.get(alternative)
                if column is None:
                    continue
                if column not in raw:
                    values[:, :, j, k] = 0.0  # the module never sees this column
                    continue
                for c in range(n_classes):
                    (grad,) = torch.autograd.grad(utilities[:, c, j].sum(), raw[column], retain_graph=True,
                                                  allow_unused=True)
                    values[:, c, j, k] = 0.0 if grad is None else grad.detach().numpy()

        log_p, per_class, log_w = _log_probabilities(output, available, torch)
        chosen = None
        if self.problem.choice_column in data.columns:
            chosen = self.problem.choices(data).map({a: i for i, a in enumerate(self.problem.alternatives)}).to_numpy()
        if per_class is None:
            weights = np.ones((n, 1))
            chosen_prob = None if chosen is None else np.exp(log_p.detach().numpy()[np.arange(n), chosen])[:, None]
        else:
            weights = np.exp(log_w.detach().numpy())
            chosen_prob = None if chosen is None else np.exp(per_class.detach().numpy()[np.arange(n), :, chosen])
        return MarginalUtilities(values, weights / weights.sum(axis=1, keepdims=True), self.problem.alternatives,
                                 list(attributes), chosen_prob=chosen_prob)


class TorchModuleFitter:
    """
    Fitter: ``TorchModuleFitter(build, problem, columns)(train) -> TorchModuleModel``.

    - ``build()`` returns a fresh, untrained ``torch.nn.Module`` (called once per fit, after seeding).
    - ``columns`` are the data columns passed to the module; ``standardize`` lists those to scale by
      training mean and standard deviation (inside the autograd graph, so marginal utilities stay in
      raw units).
    - ``n_params="count"`` reports the number of trainable parameters; pass ``None`` when no
      meaningful count exists.
    """

    def __init__(self, build: Callable[[], object], problem: ChoiceProblem, columns: Sequence[str], *,
                 standardize: Sequence[str] = (), training: Training = Training(), seed: int = 0,
                 n_params: Union[str, int, None] = "count", name: str = "torch module"):
        unknown = set(standardize) - set(columns)
        if unknown:
            raise ValueError(f"standardize names columns not passed to the module: {sorted(unknown)}")
        self.build, self.problem, self.columns = build, problem, list(columns)
        self.standardize, self.training, self.seed = list(standardize), training, seed
        self.n_params, self.name = n_params, name

    def _split(self, train: pd.DataFrame):
        if self.training.validation_fraction == 0:
            return train, None
        rng = np.random.default_rng(self.seed)
        group = self.problem.group_column
        if group is not None:
            ids = train[group].unique()
            held = set(rng.choice(ids, size=max(1, int(round(len(ids) * self.training.validation_fraction))), replace=False))
            mask = train[group].isin(held).to_numpy()
        else:
            mask = rng.random(len(train)) < self.training.validation_fraction
        return train[~mask], train[mask]

    def __call__(self, train: pd.DataFrame) -> TorchModuleModel:
        """Fit, holding out ``training.validation_fraction`` of ``train`` for early stopping."""
        fit_data, valid_data = self._split(train)
        return self.fit(fit_data, valid_data)

    def fit(self, fit_data: pd.DataFrame, valid_data: Optional[pd.DataFrame] = None) -> TorchModuleModel:
        """Fit on ``fit_data``, early-stopping on ``valid_data`` (or on ``fit_data`` itself if None)."""
        import torch

        torch.manual_seed(self.seed)
        scaling = {}
        for c in self.standardize:
            values = fit_data[c].to_numpy(dtype=float)
            scaling[c] = (float(values.mean()), float(values.std()) or 1.0)
        module = self.build().double()
        model = TorchModuleModel(module, self.problem, self.columns, scaling, None, {}, self.name)
        module.train()

        def tensors(frame):
            _, x, available = model._tensors(frame)
            chosen = torch.as_tensor(self.problem.choices(frame).map(
                {a: i for i, a in enumerate(self.problem.alternatives)}).to_numpy())
            return x, available, chosen

        x, available, chosen = tensors(fit_data)
        valid = tensors(valid_data) if valid_data is not None and len(valid_data) else None

        def nll(xb, avb, yb):
            log_p, _, _ = _log_probabilities(module(xb, avb), avb, torch)
            return -log_p[torch.arange(len(yb)), yb].sum()

        settings = self.training
        params = [p for p in module.parameters() if p.requires_grad]
        if settings.optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS(params, lr=settings.learning_rate, line_search_fn="strong_wolfe", max_iter=50)
        elif settings.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(params, lr=settings.learning_rate, weight_decay=settings.weight_decay)
        else:
            optimizer = torch.optim.Adam(params, lr=settings.learning_rate, weight_decay=settings.weight_decay)

        generator = torch.Generator().manual_seed(self.seed)
        n = len(chosen)
        best_loss, best_state, best_epoch, stale, epoch = np.inf, None, 0, 0, 0
        for epoch in range(1, settings.max_epochs + 1):
            module.train()
            if settings.optimizer == "lbfgs":
                def closure():
                    optimizer.zero_grad()
                    loss = nll(x, available, chosen) / n
                    loss.backward()
                    return loss
                optimizer.step(closure)
            else:
                order = torch.randperm(n, generator=generator) if settings.batch_size else torch.arange(n)
                size = settings.batch_size or n
                for start in range(0, n, size):
                    idx = order[start:start + size]
                    xb = {c: t[idx] for c, t in x.items()}
                    optimizer.zero_grad()
                    loss = nll(xb, available[idx], chosen[idx]) / len(idx)
                    loss.backward()
                    optimizer.step()
            module.eval()
            with torch.no_grad():
                monitored = float(nll(*valid) / len(valid[2])) if valid else float(nll(x, available, chosen) / n)
            if monitored < best_loss - 1e-9:
                best_loss, best_epoch, stale = monitored, epoch, 0
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            else:
                stale += 1
                if stale >= settings.patience:
                    break
        if best_state is not None:
            module.load_state_dict(best_state)
        module.eval()

        count = sum(int(p.numel()) for p in params)
        n_params = count if self.n_params == "count" else self.n_params
        report = {
            "torch": metadata.version("torch"), "seed": self.seed, "trainable_parameters": count,
            "epochs": epoch, "best_epoch": best_epoch,
            "best_monitored_nll_per_obs": best_loss,
            "monitored_on": "validation split" if valid else "training data",
            "stopped_early": epoch < settings.max_epochs, "finite": bool(np.isfinite(best_loss)),
            "training": asdict(settings),
        }
        return TorchModuleModel(module, self.problem, self.columns, scaling, n_params, report, self.name)
