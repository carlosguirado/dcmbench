"""
Estimate a spec with PyTorch (``pip install dcmbench[torch]``).

The spec is compiled to torch through the same likelihood code DCMBench uses
for predictions. Gradients come from autograd, the optimizer is scipy's
L-BFGS-B (exact bounds), and robust standard errors use the sandwich
H⁻¹ B H⁻¹ with B the outer product of per-unit scores.
"""

from __future__ import annotations

from importlib import metadata

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from dcmbench.problem import ChoiceProblem
from dcmbench.specs import compute
from dcmbench.specs.backends import TorchBackend
from dcmbench.specs.model import SpecModel
from dcmbench.specs.spec import Spec


class TorchEstimator:
    """
    Fitter: ``TorchEstimator(spec, problem)(train) -> SpecModel``.

    ``gtol`` is the projected-gradient tolerance of L-BFGS-B on the total
    negative log-likelihood.
    """

    def __init__(self, spec: Spec, problem: ChoiceProblem, gtol: float = 1e-6, max_iterations: int = 2000):
        compute.check_compatible(spec, problem)
        self.spec, self.problem = spec, problem
        self.gtol, self.max_iterations = gtol, max_iterations

    def __call__(self, train: pd.DataFrame) -> SpecModel:
        import torch

        spec, problem = self.spec, self.problem
        if spec.draws is not None and spec.draws.panel:
            train = train.sort_values(problem.group_column, kind="mergesort")
        backend = TorchBackend()
        inputs = compute.Inputs(spec, problem, train, backend)
        free = spec.free_parameters()
        fixed = {n: p.start for n, p in spec.parameters.items() if p.fixed}

        def unit_ll(theta):
            values = {**fixed, **{name: theta[i] for i, name in enumerate(free)}}
            per_draw = compute.probabilities_per_draw(spec, problem, inputs.columns, inputs.n_obs, values,
                                                      inputs.draws, inputs.available, backend)
            return compute.unit_loglikelihoods(per_draw, inputs.chosen, inputs.units, inputs.n_units,
                                               inputs.panel, backend)

        def objective(x):
            theta = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            nll = -unit_ll(theta).sum()
            nll.backward()
            return float(nll.detach()), theta.grad.numpy().copy()

        start = np.array([spec.parameters[n].start for n in free], dtype=float)
        bounds = [(spec.parameters[n].lower, spec.parameters[n].upper) for n in free]
        result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": self.max_iterations, "gtol": self.gtol, "ftol": 1e-15})

        theta = torch.tensor(result.x, dtype=torch.float64)
        hessian = torch.autograd.functional.hessian(lambda t: unit_ll(t).sum(), theta).numpy()
        scores = torch.autograd.functional.jacobian(unit_ll, theta, strategy="forward-mode", vectorize=True).numpy()
        std_errors = _robust_std_errors(hessian, scores, free)

        values = {**fixed, **dict(zip(free, result.x.tolist()))}
        return SpecModel(
            spec, problem, values, std_errors,
            converged=bool(result.success),
            engine="pytorch",
            engine_details={"torch": metadata.version("torch"), "optimizer": "scipy L-BFGS-B",
                            "engine_loglikelihood": -float(result.fun), "iterations": int(result.nit),
                            "message": str(result.message)},
        )


def _robust_std_errors(hessian: np.ndarray, scores: np.ndarray, names):
    """Sandwich standard errors; NaN where the Hessian is singular."""
    try:
        information_inverse = np.linalg.inv(-hessian)
    except np.linalg.LinAlgError:
        return {name: np.nan for name in names}
    covariance = information_inverse @ (scores.T @ scores) @ information_inverse
    variances = np.diag(covariance)
    return {name: float(np.sqrt(v)) if v >= 0 else np.nan for name, v in zip(names, variances)}
