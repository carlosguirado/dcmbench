"""
Swissmetro stated-preference survey (Bierlaire, Axhausen and Abay, 2001).

10,728 choices by 1,192 respondents (9 each) between train, Swissmetro and
car. Times in minutes, costs in CHF. Codebook:
https://transp-or.epfl.ch/courses/dca2017/dataDescriptions/SwissmetroDescription.pdf
(note INCOME 4 = unknown; PURPOSE 1 = commuter, 3 = business).
"""

from __future__ import annotations

from dcmbench.data.dataset import Dataset, DatasetEntry
from dcmbench.data.datasets import DATA_REPO_COMMIT, DATA_REPO_URL
from dcmbench.data.download import DataFile
from dcmbench.problem import Attribute, ChoiceProblem
from dcmbench.results import Step

PROBLEM = ChoiceProblem(
    name="swissmetro",
    alternatives=("train", "sm", "car"),
    choice_column="CHOICE",
    choice_codes={1: "train", 2: "sm", 3: "car"},
    availability={"train": "TRAIN_AV", "sm": "SM_AV", "car": "CAR_AV"},
    attributes={
        "time": Attribute({"train": "TRAIN_TT", "sm": "SM_TT", "car": "CAR_TT"}, unit="minute", kind="time"),
        "cost": Attribute({"train": "TRAIN_CO", "sm": "SM_CO", "car": "CAR_CO"}, unit="CHF", kind="cost"),
        "headway": Attribute({"train": "TRAIN_HE", "sm": "SM_HE"}, unit="minute", kind="time"),
    },
    group_column="ID",
)


def build(raw):
    known = raw["CHOICE"] != 0
    step = Step("filter", "drop observations with no recorded choice (CHOICE == 0)", {"query": "CHOICE != 0"},
                len(raw), int(known.sum()))
    return raw[known], [step]


def benchmark_sample(dataset: Dataset) -> Dataset:
    """Commuter and business trips, as in the standard Biogeme Swissmetro examples (N = 6,768)."""
    return dataset.filter("PURPOSE in (1, 3)", label="commuter and business trips (PURPOSE 1 or 3)")


ENTRY = DatasetEntry(
    name="swissmetro",
    file=DataFile(
        url=f"{DATA_REPO_URL}/swissmetro/swissmetro.csv.gz",
        sha256="bd54e4e431a0cbee89a66a0fad9eb2b2783f7e70ac11f0a2e83d2e42fe24191c",
    ),
    problem=PROBLEM,
    build=build,
    description="Swissmetro stated-preference survey: intercity mode choice between train, Swissmetro and car.",
    citation=(
        "Bierlaire, M., Axhausen, K. and Abay, G. (2001). The acceptance of modal innovation: "
        "The case of Swissmetro. Swiss Transport Research Conference."
    ),
    source_url="https://transp-or.epfl.ch/courses/dca2017/dataDescriptions/SwissmetroDescription.pdf",
    license=None,  # to be confirmed before release
    data_version=DATA_REPO_COMMIT,
    preparations={"benchmark": benchmark_sample},
)
