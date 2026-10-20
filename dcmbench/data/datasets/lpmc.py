"""
London Passenger Mode Choice (LPMC; Hillel, Elshafie and Jin, 2018).

81,086 trips from the London Travel Demand Survey with reconstructed choice
sets: walk, cycle, public transport and drive. All four alternatives are
treated as available. Durations in **hours**, costs in GBP. Also known by
its file name, ``ltds``.
"""

from __future__ import annotations

from dcmbench.data.dataset import Dataset, DatasetEntry
from dcmbench.data.datasets import DATA_REPO_COMMIT, DATA_REPO_URL
from dcmbench.data.download import DataFile
from dcmbench.problem import Attribute, ChoiceProblem
from dcmbench.results import Step

ALTERNATIVES = ("walk", "cycle", "pt", "drive")

PROBLEM = ChoiceProblem(
    name="lpmc",
    alternatives=ALTERNATIVES,
    choice_column="travel_mode",
    choice_codes={alt: alt for alt in ALTERNATIVES},
    attributes={
        "time": Attribute(
            {"walk": "dur_walking", "cycle": "dur_cycling", "pt": "dur_pt_total", "drive": "dur_driving"},
            unit="hour",
            kind="time",
        ),
        "cost": Attribute({"pt": "cost_transit", "drive": "cost_driving_total"}, unit="GBP", kind="cost"),
    },
    group_column="person_id",
)


def build(raw):
    frame = raw.copy()
    frame["person_id"] = frame["household_id"] * 100 + frame["person_n"]
    step = Step("derive", "person_id = household_id * 100 + person_n (respondent identifier)", {}, len(raw), len(raw))
    return frame, [step]


def benchmark_sample(dataset: Dataset) -> Dataset:
    """Random subsample of 5,000 trips (seed 42), the DCMBench benchmark sample."""
    return dataset.subsample(5000, seed=42)


ENTRY = DatasetEntry(
    name="lpmc",
    aliases=("ltds",),
    file=DataFile(
        url=f"{DATA_REPO_URL}/ltds/ltds.csv.gz",
        sha256="bef2ac4ec7cab2b57e50e2503bd41761dfef1ea1b5b8d9f25beeb06f3726db0a",
    ),
    problem=PROBLEM,
    build=build,
    description="London Passenger Mode Choice: revealed-preference urban trips with reconstructed choice sets.",
    citation=(
        "Hillel, T., Elshafie, M. Z. E. B. and Jin, Y. (2018). Recreating passenger mode choice-sets for "
        "transport simulation: A case study of London, UK. Proceedings of the Institution of Civil "
        "Engineers – Smart Infrastructure and Construction, 171(1), 29–42."
    ),
    source_url="https://doi.org/10.1680/jsmic.17.00018",
    license=None,  # to be confirmed before release
    data_version=DATA_REPO_COMMIT,
    preparations={"benchmark": benchmark_sample},
)
