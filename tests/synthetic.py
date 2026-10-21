"""Synthetic choice data and specs shared by the engine tests."""

import numpy as np
import pandas as pd

from dcmbench.data.dataset import Dataset, DatasetInfo
from dcmbench.problem import Attribute, ChoiceProblem
from dcmbench.specs import load_spec

ALTS = ("train", "sm", "car")
PROBLEM = ChoiceProblem(
    name="synthetic", alternatives=ALTS, choice_column="CHOICE", choice_codes={1: "train", 2: "sm", 3: "car"},
    availability={"car": "CAR_AV"}, group_column="ID",
    attributes={"time": Attribute({a: f"{a.upper()}_TT" for a in ALTS}, unit="minute", kind="time"),
                "cost": Attribute({a: f"{a.upper()}_CO" for a in ALTS}, unit="CHF", kind="cost")},
)
TRUE = {"ASC_TRAIN": -0.4, "ASC_CAR": 0.3, "B_TIME": -0.02, "B_COST": -0.05, "B_TIME_S": 0.012}


def synthetic(n_people=600, choices_each=5, mixed=False, seed=0):
    """Choices simulated from an MNL, or a panel MXL with β_time ~ N(B_TIME, B_TIME_S) per respondent."""
    rng = np.random.default_rng(seed)
    n = n_people * choices_each
    data = pd.DataFrame({"ID": np.repeat(np.arange(n_people), choices_each)})
    for alt in ALTS:
        data[f"{alt.upper()}_TT"] = rng.uniform(20, 150, n)
        data[f"{alt.upper()}_CO"] = rng.uniform(5, 60, n)
    data["CAR_AV"] = (rng.random(n) > 0.15).astype(int)
    b_time = TRUE["B_TIME"] + (TRUE["B_TIME_S"] * rng.standard_normal(n_people)[data.ID] if mixed else 0)
    v = np.column_stack([
        TRUE["ASC_TRAIN"] + b_time * data.TRAIN_TT + TRUE["B_COST"] * data.TRAIN_CO,
        b_time * data.SM_TT + TRUE["B_COST"] * data.SM_CO,
        TRUE["ASC_CAR"] + b_time * data.CAR_TT + TRUE["B_COST"] * data.CAR_CO,
    ]) + rng.gumbel(size=(n, 3))
    v[data.CAR_AV.to_numpy() == 0, 2] = -np.inf
    data["CHOICE"] = v.argmax(axis=1) + 1
    return data


def spec(model="MNL", draws=None):
    b = "B_TIME_RND" if model == "MXL" else "B_TIME"
    spec = {
        "spec_version": 2, "name": f"synthetic_{model.lower()}", "model": model, "alternatives": list(ALTS),
        "parameters": {"ASC_TRAIN": {"start": 0}, "ASC_CAR": {"start": 0}, "B_TIME": {"start": 0}, "B_COST": {"start": 0}},
        "utilities": {"train": f"ASC_TRAIN + {b} * TRAIN_TT + B_COST * TRAIN_CO", "sm": f"{b} * SM_TT + B_COST * SM_CO",
                      "car": f"ASC_CAR + {b} * CAR_TT + B_COST * CAR_CO"},
    }
    if model == "NL":
        spec["parameters"]["MU"] = {"start": 1.5, "lower": 1, "upper": 10}
        spec["nests"] = [{"name": "existing", "parameter": "MU", "alternatives": ["train", "car"]}]
    if model == "MXL":
        spec["parameters"]["B_TIME_S"] = {"start": 0.01}
        spec["random_parameters"] = {"B_TIME_RND": {"distribution": "normal", "mean": "B_TIME", "spread": "B_TIME_S"}}
        spec["draws"] = draws or {"number": 100, "seed": 1, "type": "halton", "panel": True}
    return load_spec(spec)


def dataset(frame):
    return Dataset("synthetic", frame, PROBLEM, DatasetInfo("synthetic", "none", "none", None, "test", "0" * 64, "none"))
