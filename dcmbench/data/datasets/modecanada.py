"""
ModeCanada intercity mode choice (Bhat, 1995), as distributed with the R package mlogit.

4,324 travelers choosing between train, car, bus and air. The file is in long
format (one row per available alternative); the canonical table is wide, with
``<alt>_<attribute>`` columns. An alternative is available when its row exists;
attributes of unavailable alternatives are 0. Times in minutes, costs in CAD,
income in thousand CAD.
"""

from __future__ import annotations

import pandas as pd

from dcmbench.data.dataset import DatasetEntry
from dcmbench.data.datasets import DATA_REPO_COMMIT, DATA_REPO_URL
from dcmbench.data.download import DataFile
from dcmbench.problem import Attribute, ChoiceProblem
from dcmbench.results import Step

ALTERNATIVES = ("train", "car", "bus", "air")
ALTERNATIVE_ATTRIBUTES = ("cost", "ivt", "ovt", "freq")
CASE_ATTRIBUTES = ("dist", "income", "urban", "noalt")


def _columns(attribute):
    return {alt: f"{alt}_{attribute}" for alt in ALTERNATIVES}


PROBLEM = ChoiceProblem(
    name="modecanada",
    alternatives=ALTERNATIVES,
    choice_column="choice",
    choice_codes={alt: alt for alt in ALTERNATIVES},
    availability={alt: f"{alt}_av" for alt in ALTERNATIVES},
    attributes={
        "time": Attribute(_columns("time"), unit="minute", kind="time"),
        "ivt": Attribute(_columns("ivt"), unit="minute", kind="time"),
        "ovt": Attribute(_columns("ovt"), unit="minute", kind="time"),
        "cost": Attribute(_columns("cost"), unit="CAD", kind="cost"),
        "freq": Attribute(_columns("freq"), unit="count"),
    },
)


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """One row per case from the long format (``case``, ``alt``, ``choice`` 0/1)."""
    chosen = long.loc[long["choice"] == 1].set_index("case")["alt"]
    if not chosen.index.is_unique or len(chosen) != long["case"].nunique():
        raise ValueError("every case must have exactly one chosen alternative")

    case_level = long.groupby("case")[list(CASE_ATTRIBUTES)].first()
    attributes = long.pivot(index="case", columns="alt", values=list(ALTERNATIVE_ATTRIBUTES))
    available = attributes["cost"].notna()

    wide = case_level.copy()
    wide.insert(0, "choice", chosen)
    for alt in ALTERNATIVES:
        for attribute in ALTERNATIVE_ATTRIBUTES:
            wide[f"{alt}_{attribute}"] = attributes[(attribute, alt)].fillna(0)
        wide[f"{alt}_time"] = wide[f"{alt}_ivt"] + wide[f"{alt}_ovt"]
        wide[f"{alt}_av"] = available[alt].astype(int)
    return wide.reset_index()


def build(raw):
    wide = to_wide(raw)
    steps = [
        Step("reshape", "long to wide: one row per case; an alternative is available when its row exists",
             {"id": "case", "alternatives": list(ALTERNATIVES)}, len(raw), len(wide)),
        Step("derive", "time = ivt + ovt for each alternative", {}, len(wide), len(wide)),
        Step("fill", "attributes of unavailable alternatives set to 0", {}, len(wide), len(wide)),
    ]
    return wide, steps


ENTRY = DatasetEntry(
    name="modecanada",
    file=DataFile(
        url=f"{DATA_REPO_URL}/modecanada/modecanada.csv.gz",
        sha256="8097d95e8a86d4b5f03c114b34383efef020046f1350749e52027e44589015e8",
    ),
    problem=PROBLEM,
    build=build,
    description="ModeCanada: intercity mode choice between train, car, bus and air in the Toronto–Montreal corridor.",
    citation=(
        "Bhat, C. R. (1995). A heteroscedastic extreme value model of intercity travel mode choice. "
        "Transportation Research Part B, 29(6), 471–483."
    ),
    source_url="https://cran.r-project.org/package=mlogit",
    license=None,  # to be confirmed before release
    data_version=DATA_REPO_COMMIT,
)
