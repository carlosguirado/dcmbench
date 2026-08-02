"""
Mixed logit (MXL) internal validation, using a hand-built Biogeme model.

Why this tutorial exists (and doesn't use `fitter_from_spec`)
---------------------------------------------------------------
dcmbench's JSON-spec builder (`dcmbench.model_specifications.build_model_from_spec`,
used by `fitter_from_spec`/`BiogemeSpecFitter`) only substitutes plain, non-random
`Beta` objects into utility formulas. It cannot express a `bioDraws`-based random
taste parameter, so it cannot build a genuine mixed logit model -- specs with
`model_type: "MXL"` either raise a clear error (if `random_parameters` is set) or
silently fall back to plain MNL (if not). See the README's Internal Validation
section for details.

`InternalValidator` itself does not care how a `FittedModel` is produced --- it
only calls `fitter(train_df)` and expects back an object with
`predict_probabilities`, `n_params`, `converged`, and `null_loglikelihood`.
`FunctionModelFitter` is a thin adapter that packages plain build/predict
functions into that protocol, so you can validate ANY hand-written Biogeme
model, including MXL with explicit `bioDraws`. This mirrors the
`build_*`/`predict_*` function-pair pattern used for MXL models in the
dcm-internal-validation experiments repo.

This example estimates a Swissmetro MXL with a normally-distributed travel-time
coefficient (`B_TIME_RND = B_TIME + B_TIME_S * bioDraws(...)`) and runs it
through k-fold cross-validation, with `group_column="ID"` so each traveler's
repeated choices stay together in one fold. Settings below are intentionally
small (few draws, few individuals, few folds) so this finishes quickly as a
demo; scale `N_DRAWS`, `N_INDIVIDUALS`, and `N_SPLITS` up for a real study.
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
N_INDIVIDUALS = 100  # keep ALL of each sampled individual's trips (see below)
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
    """Rebuild the same random-coefficient utility specification from scratch.

    Biogeme expressions are tied to a specific Database at BIOGEME-object
    construction time, so both estimation and OOS prediction/null-LL need
    their own fresh set of Beta/bioDraws/Variable objects bound to the
    right database -- reusing objects built for the training database
    against a different database is not supported.
    """
    asc_car = Beta("ASC_CAR", 0, None, None, 1)  # fixed for identification
    asc_train = Beta("ASC_TRAIN", 0, None, None, 0)
    b_time = Beta("B_TIME", 0, None, None, 0)
    b_time_s = Beta("B_TIME_S", 1, None, None, 0)
    b_cost = Beta("B_COST", 0, None, None, 0)
    b_time_rnd = b_time + b_time_s * bioDraws("B_TIME_RND", "NORMAL_HALTON2")

    utilities = {
        1: asc_train + b_time_rnd * Variable("TRAIN_TT") + b_cost * Variable("TRAIN_CO"),
        2: b_time_rnd * Variable("SM_TT") + b_cost * Variable("SM_CO"),
        3: asc_car + b_time_rnd * Variable("CAR_TT") + b_cost * Variable("CAR_CO"),
    }
    return utilities


def build_mxl(train_df: pd.DataFrame):
    """ModelFitter build_fn: estimate a Swissmetro MXL on the training fold."""
    database = db.Database("mxl_train", train_df.copy())
    utilities = _mxl_utilities()
    choice = Variable("CHOICE")

    logprob = log(MonteCarlo(models.logit(utilities, _AV_VARS, choice)))
    model = bio.BIOGEME(database, logprob, number_of_draws=N_DRAWS)
    model.modelName = "swissmetro_mxl_normal"
    _quiet(model)
    return model.estimate()


def predict_mxl(results, data: pd.DataFrame) -> pd.DataFrame:
    """ModelFitter predict_fn: re-simulate probabilities on new data."""
    database = db.Database("mxl_eval", data.copy())
    utilities = _mxl_utilities()

    # String keys for the simulate dict (safest/most standard Biogeme usage);
    # renamed to integer alternative IDs afterwards to match `choice_mapping`.
    simulate = {str(alt): MonteCarlo(models.logit(utilities, _AV_VARS, alt)) for alt in utilities}
    model = bio.BIOGEME(database, simulate, number_of_draws=N_DRAWS)
    _quiet(model)

    probs = model.simulate(results.get_beta_values())
    probs.columns = [int(c) for c in probs.columns]
    return probs


def null_loglikelihood_mxl(results, data: pd.DataFrame) -> float:
    """
    ModelFitter null_loglikelihood_fn: availability-aware null LL on `data`.

    calculate_null_loglikelihood only needs *some* valid BIOGEME object bound
    to the right database plus the availability dict -- the utility formula's
    content (random or not) doesn't affect the null-model calculation, so a
    trivial deterministic placeholder utility is enough here.
    """
    database = db.Database("mxl_null", data.copy())
    choice = Variable("CHOICE")
    dummy_utilities = {alt: Beta(f"dummy_{alt}", 0, None, None, 1) for alt in _AV_VARS}
    logprob = models.loglogit(dummy_utilities, _AV_VARS, choice)
    model = bio.BIOGEME(database, logprob)
    _quiet(model)
    return float(model.calculate_null_loglikelihood(_AV_VARS))


def main() -> None:
    raw = fetch_data("swissmetro_dataset")
    full_data = prepare_swissmetro_data_standard(raw)

    # Sample by INDIVIDUAL (ID), keeping all of each sampled person's trips,
    # rather than sampling rows directly -- Swissmetro is a repeated-choice
    # panel (each ID has multiple CHOICE observations), so subsampling rows
    # would fragment individuals across the demo subsample for no reason.
    rng_ids = full_data["ID"].drop_duplicates().sample(n=N_INDIVIDUALS, random_state=0)
    data = full_data[full_data["ID"].isin(rng_ids)].reset_index(drop=True)

    fitter = FunctionModelFitter(
        build_fn=build_mxl,
        predict_fn=predict_mxl,
        null_loglikelihood_fn=null_loglikelihood_mxl,
        choice_column="CHOICE",
    )
    validator = InternalValidator(
        fitter,
        choice_column="CHOICE",
        # CHOICE code -> predict_probabilities() column name. Swissmetro's
        # CHOICE codes (1/2/3 = Train/Swissmetro/Car) and predict_mxl's
        # renamed probability columns both use the same alternative-ID
        # convention here, so this is an identity map -- not because
        # choice_mapping is inherently redundant, but because the two label
        # systems happen to coincide for this model/adapter combination.
        choice_mapping={1: 1, 2: 2, 3: 3},
        # Keep all trips for a person in the same fold -- without this, a
        # person's other trips could appear in the training fold while some
        # of their trips are held out for testing, optimistically biasing
        # OOS scores for repeated-choice panel data like Swissmetro.
        group_column="ID",
        random_state=0,
        verbose=True,
    )

    print(f"Running {N_SPLITS}-fold cross-validation on a mixed logit model "
          f"({N_DRAWS} draws, {N_INDIVIDUALS} individuals, {len(data)} rows)...")
    result = validator.cross_validate(data, n_splits=N_SPLITS)
    print(result.detail[["fold", "status", "converged", "avg_ll", "rho_sq", "rho_bar_sq"]])
    print()
    print(result.summary()[["avg_ll", "rho_sq", "rho_bar_sq", "pct_correct_predictions"]])


if __name__ == "__main__":
    main()
