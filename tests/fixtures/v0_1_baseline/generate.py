"""Generate the 0.1.x regression baseline from the 0.1.x benchmark scripts.

Runs ``02_model_benchmarking.py`` (one of the 0.1.x benchmark scripts, not in
this repo) unchanged, recording every Biogeme estimation it performs, then
re-estimates the same models on an 80/20 holdout split stratified on the
choice (``random_state=42``) and scores both partitions.

Usage::

    python generate.py <path/to/02_model_benchmarking.py> <output_dir>

Environment (0.1.x era): ``dcmbench==0.1.2``, ``biogeme==3.2.14``,
``tomlkit<0.13.3``, scikit-learn. Run from an empty working directory, since
Biogeme writes ``biogeme.toml`` and report files there.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_script(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark_0_1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe_estimation(model, results) -> dict:
    """Everything a later regression test may want to compare against."""
    params = results.get_estimated_parameters()
    draws = getattr(model.database, "theDraws", None)
    return {
        "final_loglikelihood": float(results.data.logLike),
        "n_parameters": int(results.data.nparam),
        "sample_size": int(results.data.sampleSize),
        "configured_number_of_draws": int(model.number_of_draws),
        "draws_actually_used": None if draws is None else int(draws.shape[1]),
        "parameters": {
            name: {
                "value": float(row["Value"]),
                "robust_std_err": float(row["Rob. Std err"]),
            }
            for name, row in params.iterrows()
        },
    }


def record_full_sample(script) -> dict:
    """Run the script's own main() and capture each estimation."""
    import biogeme.biogeme as bio

    recorded = {}
    original_estimate = bio.BIOGEME.estimate

    def recording_estimate(self, *args, **kwargs):
        results = original_estimate(self, *args, **kwargs)
        recorded[self.modelName] = describe_estimation(self, results)
        return results

    bio.BIOGEME.estimate = recording_estimate
    try:
        script.main()
    finally:
        bio.BIOGEME.estimate = original_estimate
    return recorded


class UtilityCapture:
    """Record the utilities and nests each builder passes to biogeme.models."""

    def __init__(self, models_module):
        self.models = models_module
        self.last = None
        self._originals = {}

    def __enter__(self):
        for name in ("loglogit", "lognested", "logit"):
            original = getattr(self.models, name)
            self._originals[name] = original
            setattr(self.models, name, self._wrap(name, original))
        return self

    def __exit__(self, *exc):
        for name, original in self._originals.items():
            setattr(self.models, name, original)

    def _wrap(self, name, original):
        def wrapper(*args, **kwargs):
            if name == "lognested":
                V, av, nests = args[0], args[1], args[2]
            else:
                V, av, nests = args[0], args[1], None
            self.last = {"V": V, "av": av, "nests": nests}
            return original(*args, **kwargs)

        return wrapper


def predict_probabilities(builder, data, betas, kind, capture):
    """Choice probabilities on ``data`` at the given parameter values."""
    import biogeme.biogeme as bio
    import biogeme.database as db
    from biogeme import models
    from biogeme.expressions import MonteCarlo

    builder(data)  # only to capture V/av/nests bound to these columns
    V, av, nests = capture.last["V"], capture.last["av"], capture.last["nests"]
    logit = capture._originals["logit"]  # unwrapped, so the capture is left alone
    formulas = {}
    for alt in V:
        if kind == "NL":
            formulas[f"P_{alt}"] = models.nested(V, av, nests, alt)
        elif kind == "MXL":
            formulas[f"P_{alt}"] = MonteCarlo(logit(V, av, alt))
        else:
            formulas[f"P_{alt}"] = logit(V, av, alt)
    sim = bio.BIOGEME(
        db.Database("predict", data.reset_index(drop=True)),
        formulas,
        number_of_draws=200,
        generate_html=False,
        generate_pickle=False,
    )
    sim.modelName = "predict"
    probs = sim.simulate(betas)
    probs.columns = [int(c.split("_", 1)[1]) for c in probs.columns]
    return probs


def score(probs: pd.DataFrame, choice: pd.Series) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    choice = choice.reset_index(drop=True)
    chosen = probs.to_numpy()[np.arange(len(choice)), probs.columns.get_indexer(choice)]
    predicted = probs.idxmax(axis=1)
    labels = sorted(probs.columns)
    return {
        "loglikelihood": float(np.log(np.clip(chosen, 1e-300, None)).sum()),
        "accuracy": float(accuracy_score(choice, predicted)),
        "macro_f1": float(f1_score(choice, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(choice, predicted, labels=labels, average="macro", zero_division=0)),
    }


def holdout_80_20(script) -> list[dict]:
    """Re-estimate on an 80/20 stratified split and score both partitions."""
    from sklearn.model_selection import train_test_split
    from dcmbench.datasets import fetch_data

    configs = [
        ("Swissmetro", "swissmetro_dataset", script.prepare_swissmetro_data, {
            "MNL": script.build_swissmetro_mnl,
            "NL": script.build_swissmetro_nl,
            "MXL": script.build_swissmetro_mxl,
            "CUSTOM": script.build_swissmetro_custom_heterogeneity,
        }),
        ("ModeCanada", "modecanada_dataset", script.prepare_modecanada_data, {
            "MNL": script.build_modecanada_mnl,
            "NL": script.build_modecanada_nl,
            "MXL": script.build_modecanada_mxl_lognormal,
        }),
        ("LPMC", "ltds_dataset", script.prepare_ltds_data, {
            "MNL": script.build_ltds_mnl,
            "NL": script.build_ltds_nl,
            "MXL": script.build_ltds_mxl_lognormal,
        }),
    ]
    rows = []
    with UtilityCapture(script.models) as capture:
        for dataset, loader_name, prepare, builders in configs:
            data = prepare(fetch_data(loader_name))
            train, test = train_test_split(
                data, test_size=0.2, stratify=data["CHOICE"], random_state=42
            )
            train = train.reset_index(drop=True)
            test = test.reset_index(drop=True)
            for model_type, builder in builders.items():
                np.random.seed(42)
                model = builder(train)
                results = model.estimate()
                betas = results.get_beta_values()
                kind = "MXL" if model_type == "MXL" else ("NL" if model_type == "NL" else "MNL")
                is_scores = score(predict_probabilities(builder, train, betas, kind, capture), train["CHOICE"])
                oos_scores = score(predict_probabilities(builder, test, betas, kind, capture), test["CHOICE"])
                rows.append({
                    "dataset": dataset,
                    "model": model_type,
                    "n_train": len(train),
                    "n_test": len(test),
                    "is_ll_estimation": float(results.data.logLike),
                    **{f"is_{k}": v for k, v in is_scores.items()},
                    **{f"oos_{k}": v for k, v in oos_scores.items()},
                    "parameters": {k: float(v) for k, v in betas.items()},
                })
                print(f"{dataset} {model_type}: IS LL {rows[-1]['is_ll_estimation']:.1f}, "
                      f"OOS LL {oos_scores['loglikelihood']:.1f}, OOS F1 {oos_scores['macro_f1']:.3f}")
    return rows


def main():
    script_path, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    script = load_script(script_path)

    random.seed(42)
    np.random.seed(42)
    full_sample = record_full_sample(script)
    (out_dir / "full_sample_estimates.json").write_text(json.dumps(full_sample, indent=2))

    holdout = holdout_80_20(script)
    (out_dir / "holdout_80_20.json").write_text(json.dumps(holdout, indent=2))
    pd.DataFrame([{k: v for k, v in r.items() if k != "parameters"} for r in holdout]).to_csv(
        out_dir / "holdout_80_20.csv", index=False
    )


if __name__ == "__main__":
    main()
