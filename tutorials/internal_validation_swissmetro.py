"""
Internal validation tutorial (Swissmetro MNL).

Demonstrates k-fold cross-validation and prediction-error bootstrap using
``dcmbench.validation``. Uses small ``n_splits`` / ``n_bootstrap`` so the
example finishes quickly; increase them for research-quality runs.

Each fold or bootstrap draw re-estimates the model (full MLE).
"""

from dcmbench.datasets import fetch_data
from dcmbench.model_specifications import fetch_model_spec
from dcmbench.utils.model_extensions import prepare_swissmetro_data_standard
from dcmbench.validation import InternalValidator, fitter_from_spec


def main():
    # Load and prepare Swissmetro (filter PURPOSE, scale costs/times)
    raw = fetch_data("swissmetro_dataset")
    data = prepare_swissmetro_data_standard(raw)

    # Optional: subsample individuals for a faster demo
    demo_ids = data["ID"].drop_duplicates().sample(n=80, random_state=0)
    data = data[data["ID"].isin(demo_ids)].copy()
    print(f"Using {len(data)} observations from {data['ID'].nunique()} individuals")

    spec = fetch_model_spec("mode_choice/mnl_swissmetro.json")
    fitter = fitter_from_spec(spec, choice_column="CHOICE")

    # {CHOICE code -> probability column name}; identity when both use 1/2/3
    choice_mapping = {1: 1, 2: 2, 3: 3}

    validator = InternalValidator(
        fitter,
        choice_column="CHOICE",
        choice_mapping=choice_mapping,
        group_column="ID",
        random_state=42,
        verbose=True,
    )

    # --- K-fold CV (tiny settings for the tutorial) ---
    print("\n=== Cross-validation ===")
    cv = validator.cross_validate(data, n_splits=3)
    print(cv.folds[["fold", "status", "avg_ll", "rho_sq", "pct_correct_predictions"]])
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
