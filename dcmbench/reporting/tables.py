"""Side-by-side tables of specification cards."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Union

import pandas as pd

Cards = Union[Mapping[str, Dict[str, Any]], Iterable[Dict[str, Any]]]


def _cards(cards: Cards) -> Dict[str, Dict[str, Any]]:
    if isinstance(cards, Mapping):
        return dict(cards)
    return {f"{c['model']['name']} | {c['data']['dataset']}": c for c in cards}


def compare_cards(cards: Cards) -> pd.DataFrame:
    """
    One row per card: fit, out-of-sample prediction, VOT median and negative
    share per alternative, own-cost strong regularity per alternative, and
    the share change of each alternative in the first policy scenario.
    Cards may come from different datasets and studies.
    """
    rows = []
    for label, card in _cards(cards).items():
        fit = card["fit"]
        row = {
            "model": card["model"]["name"],
            "dataset": card["data"]["dataset"],
            "engine": card["model"]["engine"],
            "n_params": fit["n_params"],
            "fit_ll": fit["loglikelihood"],
            "fit_ll_per_obs": fit["loglikelihood"] / fit["n_obs"] if fit["n_obs"] else None,
            "rho_bar_sq": fit["rho_bar_sq"],
            "protocol": card["protocol"]["prediction"]["name"] if card["protocol"]["prediction"] else None,
        }
        if card["prediction"]:
            for name, metric in card["prediction"]["metrics"].items():
                row[f"oos_{name}"] = metric["estimate"]
                if metric["spread"] is not None:
                    row[f"oos_{name}_sd"] = metric["spread"]
        if card["value_of_time"]:
            row["vot_route"] = card["value_of_time"]["route"]
            for r in card["value_of_time"]["by_alternative"]:
                row[f"vot_median[{r['alternative']}]"] = r["q50"]
                row[f"vot_share_negative[{r['alternative']}]"] = r["share_negative"]
        if card["behavioral_regularity"]:
            for r in card["behavioral_regularity"]["pairs"]:
                if r["attribute"] == "cost":
                    row[f"regularity_strong_cost[{r['alternative']}]"] = r["strong"]
        if card["policy"]:
            scenario = card["policy"][0]
            row["scenario"] = scenario["scenario"]
            for r in scenario["shares"]:
                row[f"pct_change[{r['alternative']}]"] = r["pct_change"]
        row["n_flags"] = len(card["flags"])
        rows.append((label, row))
    return pd.DataFrame([r for _, r in rows], index=[label for label, _ in rows])
