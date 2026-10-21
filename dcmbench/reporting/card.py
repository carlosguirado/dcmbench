"""
Specification cards (prototype, ``card_version`` 0.1.0).

A card summarises what one model specification implies on one dataset:
statistical fit, out-of-sample prediction, value-of-time distribution,
policy response and behavioral regularity, together with provenance and
every caveat flagged during evaluation. Cards are plain JSON, validated
against ``card_schema.json``, so cards from different studies can be
collected and compared. The format is a strawman for community discussion,
not a standard.
"""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dcmbench.results import EvaluationResult

CARD_VERSION = "0.1.0"

# card metric -> (holdout column, CV summary prefix, bootstrap column, multiplier)
_PREDICTION_METRICS = {
    "avg_ll": ("avg_ll", "avg_ll", "avg_ll", 1.0),
    "rho_sq": ("rho_sq", "rho_sq", "rho_sq", 1.0),
    "rho_bar_sq": ("rho_bar_sq", "rho_bar_sq", "rho_bar_sq", 1.0),
    "macro_f1": ("macro_f1", "macro_f1", "macro_f1_632", 1.0),
    "macro_recall": ("macro_recall", "macro_recall", "macro_recall_632", 1.0),
    "accuracy": ("pct_correct_predictions", "pct_correct_predictions", "pct_correct_predictions_632", 0.01),
    "brier": ("brier", "brier", "brier_632", 1.0),
}


def _number(value) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else value


def _clean(value):
    """Plain JSON values: numpy scalars to Python, NaN to None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float):
        return _number(value)
    return value


def _prediction(result: EvaluationResult) -> Optional[Dict[str, Any]]:
    if "prediction" not in result.analyses:
        return None
    analysis = result["prediction"]
    row = analysis.summary.iloc[0].to_dict()
    protocol = analysis.settings["protocol"]
    metrics = {}
    for name, (holdout, cv, boot, scale) in _PREDICTION_METRICS.items():
        if protocol == "holdout":
            value, spread, how = row.get(holdout), None, "holdout test data"
        elif protocol == "cross_validation":
            value, spread, how = row.get(f"{cv}_mean"), row.get(f"{cv}_std"), "mean (sd) over folds"
        else:
            is_likelihood = name in ("avg_ll", "rho_sq", "rho_bar_sq")
            value = row.get(boot)
            spread = row.get(f"{boot}_std") if is_likelihood else None
            how = "out-of-bag mean (sd) over draws" if is_likelihood else ".632 bootstrap"
        if value is None:
            continue
        metrics[name] = {
            "estimate": _number(value * scale if value is not None else None),
            "spread": _number(spread * scale) if spread is not None else None,
            "how": how,
        }
    if protocol == "holdout" and "loglikelihood" in row:
        metrics["loglikelihood"] = {"estimate": _number(row["loglikelihood"]), "spread": None, "how": "holdout test data"}
    return {"metrics": metrics}


def _value_of_time(result: EvaluationResult) -> Optional[Dict[str, Any]]:
    if "vot" not in result.analyses:
        return None
    analysis = result["vot"]
    rows = []
    for row in analysis.summary.to_dict(orient="records"):
        rows.append({k: _clean(v) for k, v in row.items()})
    return {
        "route": analysis.settings["route"],
        "currency_per_hour": analysis.settings["currency_per_hour"],
        "upper_bound": _clean(analysis.settings.get("upper_bound")),
        "by_alternative": rows,
    }


def _policy(result: EvaluationResult) -> List[Dict[str, Any]]:
    entries = []
    for key, analysis in result.analyses.items():
        if analysis.kind != "scenario":
            continue
        entries.append({
            "scenario": analysis.settings["scenario"],
            "perturbations": list(analysis.settings["perturbations"]),
            "diversion_from": analysis.settings.get("diversion_from"),
            "shares": [_clean(r) for r in analysis.summary.to_dict(orient="records")],
        })
    return entries


def _regularity(result: EvaluationResult) -> Optional[Dict[str, Any]]:
    if "regularity" not in result.analyses:
        return None
    analysis = result["regularity"]
    columns = ["alternative", "attribute", "definition", "strong", "weak", "n_obs", "n_excluded_unavailable",
               "n_zero_change"]
    return {
        "settings": _clean(analysis.settings),
        "pairs": [_clean({c: r[c] for c in columns}) for r in analysis.summary.to_dict(orient="records")],
    }


def build_card(result: EvaluationResult) -> Dict[str, Any]:
    """The specification card of an ``EvaluationResult``."""
    model = dict(result.provenance.model)
    dataset = result.provenance.dataset
    behavioral = next((a.settings.get("evaluated_on") for a in result.analyses.values() if a.kind != "prediction"),
                      None)
    prediction_settings = result["prediction"].settings if "prediction" in result.analyses else None
    card = {
        "card_version": CARD_VERSION,
        "model": {
            "name": result.model_name,
            "engine": str(model.get("engine", model.get("class", "unknown"))),
            "spec_sha256": model.get("spec_sha256"),
            "details": _clean({k: v for k, v in model.items() if k not in ("name", "engine", "spec_sha256")}),
        },
        "data": {
            "dataset": dataset.get("name", result.problem_name),
            "data_version": str(dataset.get("data_version", "")),
            "sha256": str(dataset.get("sha256", "")),
            "citation": str(dataset.get("citation", "")),
            "n_obs": int(result.provenance.steps[-1].rows_after) if result.provenance.steps
            and result.provenance.steps[-1].rows_after is not None else int(result.fit.n_obs),
            "steps": [_clean({"action": s.action, "description": s.description, "rows_after": s.rows_after})
                      for s in result.provenance.steps],
        },
        "protocol": {
            "prediction": None if prediction_settings is None else _clean(
                {"name": prediction_settings["protocol"],
                 **{k: v for k, v in prediction_settings.items() if k not in ("protocol", "evaluated_on")}}),
            "behavioral_analyses_on": behavioral or "none",
            "seeds": _clean(result.provenance.seeds),
        },
        "software": {k: str(v) for k, v in result.provenance.software.items()},
        "fit": _clean({
            "n_obs": result.fit.n_obs,
            "loglikelihood": result.fit.loglikelihood,
            "null_loglikelihood": result.fit.null_loglikelihood,
            "n_params": result.fit.n_params,
            "rho_sq": result.fit.rho_sq,
            "rho_bar_sq": result.fit.rho_bar_sq,
            "converged": result.fit.converged,
            "not_applicable": result.fit.not_applicable,
            "notes": list(result.fit.notes),
        }),
        "prediction": _prediction(result),
        "value_of_time": _value_of_time(result),
        "policy": _policy(result),
        "behavioral_regularity": _regularity(result),
        "flags": [{"section": key, "message": flag} for key, analysis in result.analyses.items()
                  for flag in analysis.flags],
    }
    validate_card(card)
    return card


def schema() -> Dict[str, Any]:
    return json.loads(resources.files("dcmbench.reporting").joinpath("card_schema.json").read_text())


def validate_card(card: Dict[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` unless ``card`` matches the card schema."""
    import jsonschema

    jsonschema.validate(card, schema())


def save_card(card: Dict[str, Any], path: Union[str, Path]) -> Path:
    validate_card(card)
    path = Path(path)
    path.write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
    return path


def load_card(path: Union[str, Path]) -> Dict[str, Any]:
    card = json.loads(Path(path).read_text())
    validate_card(card)
    return card


# --- rendering -------------------------------------------------------------------------------------

def _fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(card: Dict[str, Any]) -> str:
    """A human-readable view of a card."""
    lines = [f"# Specification card: {card['model']['name']} on {card['data']['dataset']}", ""]
    data, protocol = card["data"], card["protocol"]
    lines += [
        f"- **Engine**: {card['model']['engine']}" + (f" (spec `{card['model']['spec_sha256'][:12]}`)" if card["model"].get("spec_sha256") else ""),
        f"- **Data**: {data['dataset']}, {data['n_obs']} observations, version `{data['data_version'][:12]}`, sha256 `{data['sha256'][:12]}`",
        f"- **Prediction protocol**: {protocol['prediction']['name'] if protocol['prediction'] else 'none'}; "
        f"behavioral analyses on {protocol['behavioral_analyses_on']}",
        f"- **Card version**: {card['card_version']} (prototype)",
        "",
        "## Statistical fit (in-sample)",
        "",
        "| LL | null LL | parameters | ρ² | ρ̄² | converged |",
        "|---|---|---|---|---|---|",
    ]
    fit = card["fit"]
    lines.append(f"| {_fmt(fit['loglikelihood'], 1)} | {_fmt(fit['null_loglikelihood'], 1)} | {_fmt(fit['n_params'])} | "
                 f"{_fmt(fit['rho_sq'])} | {_fmt(fit['rho_bar_sq'])} | {fit['converged']} |")
    lines += [f"\n*Note*: {note}" for note in fit.get("notes", [])]
    if card["prediction"]:
        lines += ["", "## Prediction (out of sample)", "", "| metric | estimate | spread | how |", "|---|---|---|---|"]
        for name, m in card["prediction"]["metrics"].items():
            lines.append(f"| {name} | {_fmt(m['estimate'])} | {_fmt(m['spread'])} | {m['how']} |")
    if card["value_of_time"]:
        vot = card["value_of_time"]
        lines += ["", f"## Value of time ({vot['currency_per_hour']}/hour, route {vot['route']})", "",
                  "| alternative | n | mean | 5% | 50% | 95% | share < 0 |", "|---|---|---|---|---|---|---|"]
        for r in vot["by_alternative"]:
            lines.append(f"| {r['alternative']} | {r['n_obs']} | {_fmt(r['mean'], 1)} | {_fmt(r['q05'], 1)} | "
                         f"{_fmt(r['q50'], 1)} | {_fmt(r['q95'], 1)} | {_fmt(r['share_negative'])} |")
    for scenario in card["policy"]:
        lines += ["", f"## Policy: {scenario['scenario']}", "", "| alternative | base share | scenario share | change (%) | diversion |",
                  "|---|---|---|---|---|"]
        for r in scenario["shares"]:
            lines.append(f"| {r['alternative']} | {_fmt(r['share_base'])} | {_fmt(r['share_scenario'])} | "
                         f"{_fmt(r['pct_change'], 1)} | {_fmt(r.get('diversion'))} |")
    if card["behavioral_regularity"]:
        lines += ["", "## Behavioral regularity", "", "| alternative | attribute | definition | strong | weak | n |",
                  "|---|---|---|---|---|---|"]
        for r in card["behavioral_regularity"]["pairs"]:
            lines.append(f"| {r['alternative']} | {r['attribute']} | {r['definition']} | {_fmt(r['strong'])} | "
                         f"{_fmt(r['weak'])} | {r['n_obs']} |")
    if card["flags"]:
        lines += ["", "## Flags", ""] + [f"- *{f['section']}*: {f['message']}" for f in card["flags"]]
    return "\n".join(lines) + "\n"
