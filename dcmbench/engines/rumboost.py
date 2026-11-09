"""
RUMBoost (Salvadé and Hillel, 2025): gradient-boosted random utility models.

Requires the RUMBoost source (MIT) and ``lightgbm==4.5.0``. The PyPI release
(1.0.2) predates the documented API and does not install on Python 3.12, and
the GitHub repository is not pip-installable, so check it out and put it on
the path::

    git clone https://github.com/big-ucl/rumboost third_party/rumboost
    git -C third_party/rumboost checkout 068070f
    export PYTHONPATH=third_party/rumboost

Training and prediction run in a separate worker process: LightGBM and PyTorch
each load an OpenMP runtime, and on macOS the two crash or deadlock in one
process.

Each alternative's utility is boosted from its own variables, with one
variable per tree (``interaction_constraints``) so utilities stay additive;
``monotone`` constraints (e.g. −1 for own time and cost) make utilities
non-increasing in those variables. Utilities are step functions, so the model
is piecewise constant. RUMBoost cannot use availability during training;
DCMBench masks unavailable alternatives at prediction time
(``handles_availability = False``), which is not equivalent to training with
them excluded.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import weakref
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from dcmbench.problem import ChoiceProblem

PINNED_COMMIT = "068070f"


def _check_importable() -> None:
    """Fail early, in this process, if RUMBoost cannot be imported (without importing it here)."""
    import importlib.util

    if importlib.util.find_spec("lightgbm") is None or importlib.util.find_spec("rumboost") is None:
        raise ImportError(
            "RUMBoost is not importable. Clone https://github.com/big-ucl/rumboost at commit "
            f"{PINNED_COMMIT}, add it to PYTHONPATH, and install lightgbm==4.5.0 (see dcmbench.engines.rumboost)."
        )


class _Worker:
    """A subprocess holding one RUMBoost model (see dcmbench.engines._rumboost_worker)."""

    def __init__(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        self.process = subprocess.Popen(
            [sys.executable, "-c", "from dcmbench.engines._rumboost_worker import main; main()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
        )
        self._finalizer = weakref.finalize(self, _Worker._stop, self.process)

    def call(self, command, payload=None):
        if self.process.poll() is not None:
            raise RuntimeError(f"the RUMBoost worker exited with code {self.process.returncode}")
        pickle.dump((command, payload), self.process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        self.process.stdin.flush()
        try:
            status, value = pickle.load(self.process.stdout)
        except EOFError:
            raise RuntimeError(f"the RUMBoost worker died (exit code {self.process.wait()})") from None
        if status != "ok":
            raise RuntimeError(f"RUMBoost failed in its worker process:\n{value}")
        return value

    @staticmethod
    def _stop(process):
        if process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=10)
            except Exception:
                process.kill()


def _columns(data: pd.DataFrame, columns: Sequence[str]):
    return {c: data[c].to_numpy(dtype=float) for c in columns}


class RumboostModel:
    """A trained RUMBoost model; it lives in its own worker process."""

    handles_availability = False
    piecewise_constant = True

    def __init__(self, worker: _Worker, problem: ChoiceProblem, columns: Sequence[str], report):
        self.worker, self.problem, self.columns, self.report = worker, problem, list(columns), report

    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.worker.call("predict", _columns(data, self.columns))
        return pd.DataFrame(probabilities, columns=list(self.problem.alternatives), index=data.index)

    @property
    def n_params(self) -> None:
        return None

    def null_loglikelihood(self, data: pd.DataFrame) -> float:
        return float(-np.log(self.problem.availability_matrix(data).sum(axis=1)).sum())

    @property
    def converged(self) -> bool:
        return True

    def describe(self):
        return {"engine": "rumboost", "commit": PINNED_COMMIT, "availability_in_training": False,
                "process": "separate worker (LightGBM and PyTorch cannot share a process on macOS)", **self.report}


class RumboostFitter:
    """
    Fitter: ``RumboostFitter(problem, variables, monotone)(train) -> RumboostModel``.

    - ``variables[alternative]`` lists the columns entering that alternative's utility.
    - ``monotone[column]`` is −1, 0 or +1 (default 0).
    - With ``validation_fraction > 0``, boosting stops early on a held-out part of the training
      data (respondent-grouped when the problem has a group column); with 0, exactly
      ``num_iterations`` rounds are run.
    """

    def __init__(self, problem: ChoiceProblem, variables: Mapping[str, Sequence[str]],
                 monotone: Optional[Mapping[str, int]] = None, *, learning_rate: float = 0.1, max_depth: int = 1,
                 num_iterations: int = 3000, early_stopping_rounds: int = 100, validation_fraction: float = 0.15,
                 seed: int = 0):
        unknown = set(variables) - set(problem.alternatives)
        if unknown or set(variables) != set(problem.alternatives):
            raise ValueError(f"give variables for exactly the alternatives {list(problem.alternatives)}")
        self.problem, self.variables = problem, {a: list(v) for a, v in variables.items()}
        self.monotone = dict(monotone or {})
        bad = {c: m for c, m in self.monotone.items() if m not in (-1, 0, 1)}
        if bad:
            raise ValueError(f"monotone constraints must be -1, 0 or 1: {bad}")
        self.learning_rate, self.max_depth = learning_rate, max_depth
        self.num_iterations, self.early_stopping_rounds = num_iterations, early_stopping_rounds
        self.validation_fraction, self.seed = validation_fraction, seed

    @property
    def columns(self):
        return sorted({c for cols in self.variables.values() for c in cols})

    def specification(self):
        structure = []
        for j, alternative in enumerate(self.problem.alternatives):
            columns = self.variables[alternative]
            structure.append({
                "utility": [j],
                "variables": columns,
                "shared": False,
                "boosting_params": {
                    "max_depth": self.max_depth, "learning_rate": self.learning_rate, "verbose": -1,
                    "monotone_constraints_method": "advanced",
                    "monotone_constraints": [self.monotone.get(c, 0) for c in columns],
                    "interaction_constraints": [[i] for i in range(len(columns))],
                },
            })
        general = {"num_classes": len(self.problem.alternatives), "num_iterations": self.num_iterations,
                   "verbosity": -1}
        if self.validation_fraction > 0:
            general["early_stopping_round"] = self.early_stopping_rounds
        return {"general_params": general, "rum_structure": structure}

    def _split(self, train: pd.DataFrame):
        if self.validation_fraction == 0:
            return train, None
        rng = np.random.default_rng(self.seed)
        group = self.problem.group_column
        if group is not None:
            ids = train[group].unique()
            held = set(rng.choice(ids, size=max(1, int(round(len(ids) * self.validation_fraction))), replace=False))
            mask = train[group].isin(held).to_numpy()
        else:
            mask = rng.random(len(train)) < self.validation_fraction
        return train[~mask], train[mask]

    def __call__(self, train: pd.DataFrame) -> RumboostModel:
        _check_importable()
        fit_data, valid_data = self._split(train)
        position = {a: i for i, a in enumerate(self.problem.alternatives)}
        labels = lambda frame: self.problem.choices(frame).map(position).to_numpy()  # noqa: E731
        worker = _Worker()
        best_iteration = worker.call("fit", {
            "seed": self.seed,
            "specification": self.specification(),
            "train": _columns(fit_data, self.columns), "train_label": labels(fit_data),
            "valid": None if valid_data is None else _columns(valid_data, self.columns),
            "valid_label": None if valid_data is None else labels(valid_data),
        })
        report = {"best_iteration": best_iteration, "learning_rate": self.learning_rate,
                  "max_depth": self.max_depth, "validation": "held-out split" if valid_data is not None else "none",
                  "monotone": self.monotone}
        return RumboostModel(worker, self.problem, self.columns, report)
