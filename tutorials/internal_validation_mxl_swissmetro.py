"""
Swissmetro mixed logit (MXL) internal validation.

A spec v2 expresses genuine random coefficients (here a normally distributed
time coefficient with panel draws, one draw per respondent), and any engine
can estimate it. Demo settings are small; scale ``N_DRAWS``,
``N_INDIVIDUALS`` and ``N_SPLITS`` for real work.

For models a spec cannot express, ``dcmbench.validation.FunctionModelFitter``
wraps hand-written build and predict functions instead.
"""

from __future__ import annotations

import dcmbench.data
from dcmbench.engines.biogeme import BiogemeEstimator
from dcmbench.specs import load_spec
from dcmbench.validation import InternalValidator

N_DRAWS = 20  # small for a fast demo; use >= 500 draws for real inference
N_INDIVIDUALS = 100
N_SPLITS = 2

SPEC = {
    "spec_version": 2,
    "name": "swissmetro_mxl_normal",
    "model": "MXL",
    "alternatives": ["train", "sm", "car"],
    "parameters": {"ASC_TRAIN": {"start": 0}, "ASC_CAR": {"start": 0}, "B_TIME": {"start": 0},
                   "B_TIME_S": {"start": 1}, "B_COST": {"start": 0}},
    "random_parameters": {"B_TIME_RND": {"distribution": "normal", "mean": "B_TIME", "spread": "B_TIME_S"}},
    "draws": {"number": N_DRAWS, "seed": 0, "type": "halton", "panel": True},
    "utilities": {
        "train": "ASC_TRAIN + B_TIME_RND * TRAIN_TT / 100 + B_COST * TRAIN_CO * (GA == 0) / 100",
        "sm": "B_TIME_RND * SM_TT / 100 + B_COST * SM_CO * (GA == 0) / 100",
        "car": "ASC_CAR + B_TIME_RND * CAR_TT / 100 + B_COST * CAR_CO / 100",
    },
}


def main() -> None:
    dataset = dcmbench.data.load("swissmetro").prepare("benchmark")
    full_data, problem = dataset.frame, dataset.problem
    ids = full_data["ID"].drop_duplicates().sample(n=N_INDIVIDUALS, random_state=0)
    data = full_data[full_data["ID"].isin(ids)].reset_index(drop=True)

    validator = InternalValidator(
        BiogemeEstimator(load_spec(SPEC), problem),
        choice_column=problem.choice_column,
        choice_mapping=problem.choice_mapping,
        group_column="ID",
        random_state=0,
        verbose=True,
    )

    print(
        f"Running {N_SPLITS}-fold CV on MXL "
        f"({N_DRAWS} draws, {N_INDIVIDUALS} individuals, {len(data)} rows)..."
    )
    result = validator.cross_validate(data, n_splits=N_SPLITS)
    print(result.detail[["fold", "status", "converged", "avg_ll", "rho_sq", "rho_bar_sq"]])
    print()
    print(result.summary()[["avg_ll", "rho_sq", "rho_bar_sq", "pct_correct_predictions"]])


if __name__ == "__main__":
    main()
