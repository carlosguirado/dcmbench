"""
Swissmetro mixed logit (MXL) internal validation via FunctionModelFitter.

``fitter_from_spec`` cannot build genuine MXL (no ``bioDraws`` wiring). Use
hand-built build/predict functions instead. Demo settings are small; scale
``N_DRAWS``, ``N_INDIVIDUALS``, and ``N_SPLITS`` for real work.
"""

from __future__ import annotations

import biogeme.biogeme as bio
import biogeme.database as db
import pandas as pd
from biogeme import models
from biogeme.expressions import Beta, Variable, bioDraws, log, MonteCarlo

from dcmbench.datasets import fetch_data
from dcmbench.utils.model_extensions import prepare_swissmetro_data_standard
from dcmbench.validation import FunctionModelFitter, InternalValidator

N_DRAWS = 20  # small for a fast demo; use >= 500 draws for real inference
N_INDIVIDUALS = 100
N_SPLITS = 2

_AV_VARS = {1: Variable("TRAIN_AV"), 2: Variable("SM_AV"), 3: Variable("CAR_AV")}


def _quiet(model) -> None:
    for attr, value in (
        ("generate_html", False),
        ("generate_pickle", False),
        ("saveIterations", False),
    ):
        if hasattr(model, attr):
            setattr(model, attr, value)


def _mxl_utilities():
    """Same random-coefficient utilities for estimate and predict (fresh expressions)."""
    asc_car = Beta("ASC_CAR", 0, None, None, 1)
    asc_train = Beta("ASC_TRAIN", 0, None, None, 0)
    b_time = Beta("B_TIME", 0, None, None, 0)
    b_time_s = Beta("B_TIME_S", 1, None, None, 0)
    b_cost = Beta("B_COST", 0, None, None, 0)
    b_time_rnd = b_time + b_time_s * bioDraws("B_TIME_RND", "NORMAL_HALTON2")
    return {
        1: asc_train + b_time_rnd * Variable("TRAIN_TT") + b_cost * Variable("TRAIN_CO"),
        2: b_time_rnd * Variable("SM_TT") + b_cost * Variable("SM_CO"),
        3: asc_car + b_time_rnd * Variable("CAR_TT") + b_cost * Variable("CAR_CO"),
    }


def build_mxl(train_df: pd.DataFrame):
    database = db.Database("mxl_train", train_df.copy())
    utilities = _mxl_utilities()
    logprob = log(MonteCarlo(models.logit(utilities, _AV_VARS, Variable("CHOICE"))))
    model = bio.BIOGEME(database, logprob, number_of_draws=N_DRAWS)
    model.modelName = "swissmetro_mxl_normal"
    _quiet(model)
    return model.estimate()


def predict_mxl(results, data: pd.DataFrame) -> pd.DataFrame:
    database = db.Database("mxl_eval", data.copy())
    utilities = _mxl_utilities()
    simulate = {
        str(alt): MonteCarlo(models.logit(utilities, _AV_VARS, alt)) for alt in utilities
    }
    model = bio.BIOGEME(database, simulate, number_of_draws=N_DRAWS)
    _quiet(model)
    probs = model.simulate(results.get_beta_values())
    probs.columns = [int(c) for c in probs.columns]
    return probs


def null_loglikelihood_mxl(results, data: pd.DataFrame) -> float:
    database = db.Database("mxl_null", data.copy())
    dummy = {alt: Beta(f"dummy_{alt}", 0, None, None, 1) for alt in _AV_VARS}
    model = bio.BIOGEME(database, models.loglogit(dummy, _AV_VARS, Variable("CHOICE")))
    _quiet(model)
    return float(model.calculate_null_loglikelihood(_AV_VARS))


def main() -> None:
    full_data = prepare_swissmetro_data_standard(fetch_data("swissmetro_dataset"))
    ids = full_data["ID"].drop_duplicates().sample(n=N_INDIVIDUALS, random_state=0)
    data = full_data[full_data["ID"].isin(ids)].reset_index(drop=True)

    fitter = FunctionModelFitter(
        build_fn=build_mxl,
        predict_fn=predict_mxl,
        null_loglikelihood_fn=null_loglikelihood_mxl,
        choice_column="CHOICE",
    )
    validator = InternalValidator(
        fitter,
        choice_column="CHOICE",
        choice_mapping={1: 1, 2: 2, 3: 3},
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
