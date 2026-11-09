"""
MNL with covariates: the dataset's MNL spec plus alternative-specific
coefficients on the same individual characteristics the hybrid and machine
learning examples use.

It isolates the information set from the model class. Comparing
TasteNet or RUMBoost with a plain MNL mixes two effects: a more flexible
model, and more data about the traveller. This MNL sees the same
characteristics, linearly and with constant tastes, so a gap that remains is
due to the model class.

The model is an ordinary spec v2 (so it runs on Biogeme or PyTorch). Each
characteristic enters every non-reference utility with its own coefficient.
Non-binary characteristics are divided by their mean absolute value in the
formula, so coefficients are per average unit.

    python examples/mnl_covariates.py [--datasets swissmetro] [--protocol cv]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np

import common
from common import ModelInputs

SPECS = Path(__file__).parent / "specs"


def spec(dataset, inputs: ModelInputs, base=None) -> Dict[str, Any]:
    """Spec v2 dict: ``base`` (a spec dict or ``Spec``; default ``specs/<dataset>_mnl.json``) plus covariate terms."""
    base = base if base is not None else json.loads((SPECS / f"{dataset.name}_mnl.json").read_text())
    if hasattr(base, "to_dict"):
        base = base.to_dict()
    result = json.loads(json.dumps(base))
    result["name"] = f"{base['name']}_covariates"
    result["description"] = (f"{base['description']} Plus alternative-specific coefficients on: "
                             f"{', '.join(inputs.characteristics)} (non-binary ones divided by their mean absolute value).")
    frame = dataset.frame
    for column in inputs.characteristics:
        values = frame[column].to_numpy(dtype=float)
        binary = set(np.unique(values)) <= {0.0, 1.0}
        scale = 1.0 if binary else float(np.mean(np.abs(values)))
        term = column if binary else f"({column} / {scale:.6g})"
        for alternative in inputs.alternatives:
            if alternative == inputs.reference:
                continue
            name = f"B_{re.sub(r'[^0-9A-Za-z]', '_', column).upper()}_{alternative.upper()}"
            result["parameters"][name] = {"start": 0}
            result["utilities"][alternative] = f"{result['utilities'][alternative]} + {name} * {term}"
    return result


def fitter(dataset, inputs: ModelInputs, max_iterations: int = 10000, **options):
    """PyTorch fitter. Many weakly identified covariate terms (e.g. a rare alternative) can need
    more L-BFGS-B iterations than the default."""
    from dcmbench.engines.torch import TorchEstimator
    from dcmbench.specs import load_spec

    return TorchEstimator(load_spec(spec(dataset, inputs)), dataset.problem, max_iterations=max_iterations, **options)


def main():
    from dcmbench.evaluation import evaluate

    protocol, datasets = common.protocol_from_args(__doc__.splitlines()[1])
    for name in datasets:
        dataset = common.benchmark_dataset(name)
        result = evaluate(fitter(dataset, common.benchmark_inputs(dataset)), dataset, name="MNL + covariates",
                          protocol=protocol, scenarios=[common.car_cost_scenario(name)])
        common.report(result)


if __name__ == "__main__":
    main()
