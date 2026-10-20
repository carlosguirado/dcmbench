"""
Internal validation tutorial (Swissmetro MNL).

Demonstrates k-fold cross-validation and prediction-error bootstrap using
``dcmbench.validation``. Uses small ``n_splits`` / ``n_bootstrap`` so the
example finishes quickly; increase them for research-quality runs.

Each fold or bootstrap draw re-estimates the model (full MLE).
"""

import dcmbench.data
from dcmbench.engines.biogeme import BiogemeEstimator
from dcmbench.specs import load_spec
from dcmbench.validation import InternalValidator

SPEC = {
    "spec_version": 2,
    "name": "swissmetro_mnl",
    "model": "MNL",
    "alternatives": ["train", "sm", "car"],
    "parameters": {"ASC_TRAIN": {"start": 0}, "ASC_CAR": {"start": 0}, "B_TIME": {"start": 0}, "B_COST": {"start": 0}},
    "utilities": {
        "train": "ASC_TRAIN + B_TIME * TRAIN_TT / 100 + B_COST * TRAIN_CO * (GA == 0) / 100",
        "sm": "B_TIME * SM_TT / 100 + B_COST * SM_CO * (GA == 0) / 100",
        "car": "ASC_CAR + B_TIME * CAR_TT / 100 + B_COST * CAR_CO / 100",
    },
}


def main():
    # Commuter and business trips; times and costs are scaled inside the spec
    dataset = dcmbench.data.load("swissmetro").prepare("benchmark")
    data, problem = dataset.frame, dataset.problem

    # Optional: subsample individuals for a faster demo
    demo_ids = data["ID"].drop_duplicates().sample(n=80, random_state=0)
    data = data[data["ID"].isin(demo_ids)].copy()
    print(f"Using {len(data)} observations from {data['ID'].nunique()} individuals")

    fitter = BiogemeEstimator(load_spec(SPEC), problem)

    validator = InternalValidator(
        fitter,
        choice_column=problem.choice_column,
        choice_mapping=problem.choice_mapping,  # {CHOICE code -> probability column}
        group_column="ID",
        random_state=42,
        verbose=True,
    )

    # --- K-fold CV (tiny settings for the tutorial) ---
    print("\n=== Cross-validation ===")
    cv = validator.cross_validate(data, n_splits=3)
    print(cv.folds[["fold", "status", "avg_ll", "rho_sq", "pct_correct_predictions", "macro_f1"]])
    print("\nSummary:")
    print(cv.summary()[["avg_ll", "rho_sq", "pct_correct_predictions", "n_success"]])

    # --- Prediction-error bootstrap ---
    print("\n=== Bootstrap (OOB + 0.632) ===")
    boot = validator.bootstrap(data, n_bootstrap=5)
    print(boot.iterations[["bootstrap_iter", "status", "avg_ll", "rho_sq"]].head())
    summary = boot.summary()
    cols = [
        "avg_ll",
        "rho_sq",
        "fitting_factor_oob",
        "fitting_factor_632",
        "pct_correct_predictions_632",
        "n_success",
    ]
    print("\nSummary:")
    print(summary[cols])


if __name__ == "__main__":
    main()
