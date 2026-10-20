"""Unit tests for dcmbench.data, using a local fake dataset (no network)."""

import gzip

import pandas as pd
import pytest

import dcmbench.data as data
from dcmbench.data.dataset import DatasetEntry
from dcmbench.data.datasets.modecanada import to_wide
from dcmbench.data.download import DataFile, DataIntegrityError, sha256_of
from dcmbench.problem import Attribute, ChoiceProblem
from dcmbench.results import Step

PROBLEM = ChoiceProblem(
    name="fake",
    alternatives=("a", "b"),
    choice_column="CHOICE",
    choice_codes={1: "a", 2: "b"},
    attributes={"time": Attribute({"a": "A_TT", "b": "B_TT"}, unit="minute", kind="time")},
)


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    """Register a dataset served from a local file:// URL, with an isolated cache."""
    monkeypatch.setenv("DCMBENCH_DATA_DIR", str(tmp_path / "cache"))
    source = tmp_path / "fake.csv.gz"
    frame = pd.DataFrame({"CHOICE": [1, 2, 0, 1, 2, 1], "A_TT": [10, 20, 30, 40, 50, 60], "B_TT": [5, 5, 5, 5, 5, 5]})
    with gzip.open(source, "wt") as handle:
        frame.to_csv(handle, index=False)

    def build(raw):
        keep = raw["CHOICE"] != 0
        return raw[keep], [Step("filter", "drop CHOICE == 0", {}, len(raw), int(keep.sum()))]

    def entry(sha256):
        return DatasetEntry(
            name="fake",
            aliases=("fake_alias",),
            file=DataFile(url=source.as_uri(), sha256=sha256),
            problem=PROBLEM,
            build=build,
            description="fake",
            citation="none",
            source_url="none",
            license=None,
            data_version="test",
            preparations={"short": lambda ds: ds.filter("A_TT < 45", label="short trips")},
        )

    data.register(entry(sha256_of(source)))
    return source, entry


def test_load_records_every_step(fake_dataset):
    ds = data.load("fake")
    assert len(ds) == 5
    assert [s.action for s in ds.provenance] == ["load", "filter"]
    assert ds.provenance[0].rows_after == 6 and ds.provenance[1].rows_after == 5
    assert data.load("fake_alias").name == "fake"


def test_load_uses_verified_cache(fake_dataset):
    source, _ = fake_dataset
    data.load("fake")
    source.unlink()  # a second load must not need the source
    assert len(data.load("fake")) == 5


def test_checksum_mismatch_raises(fake_dataset):
    _, entry = fake_dataset
    data.register(entry("0" * 64))
    with pytest.raises(DataIntegrityError, match="expected"):
        data.load("fake")


def test_corrupt_cache_is_downloaded_again(fake_dataset):
    _, entry = fake_dataset
    ds = data.load("fake")
    cached = data.cache_dir() / entry(ds.info.sha256).file.cache_name
    cached.write_bytes(b"corrupted")
    assert len(data.load("fake")) == 5
    assert sha256_of(cached) == ds.info.sha256


def test_filter_subsample_prepare_return_new_datasets(fake_dataset):
    ds = data.load("fake")
    short = ds.prepare("short")
    sample = ds.subsample(3, seed=1)
    assert len(ds) == 5 and len(short) == 3 and len(sample) == 3
    assert short.provenance[-1].description == "short trips"
    assert sample.provenance[-1].params == {"n": 3, "seed": 1}
    assert list(sample.frame.index) == [0, 1, 2]
    with pytest.raises(KeyError, match="no preparation"):
        ds.prepare("missing")
    with pytest.raises(ValueError, match="cannot sample"):
        ds.subsample(10, seed=1)


def test_unknown_dataset_lists_available():
    with pytest.raises(KeyError, match="swissmetro"):
        data.load("no_such_dataset")


def test_modecanada_long_to_wide():
    long = pd.DataFrame(
        {
            "case": [1, 1, 1, 2, 2],
            "alt": ["train", "car", "air", "car", "bus"],
            "choice": [0, 1, 0, 0, 1],
            "dist": [100, 100, 100, 200, 200],
            "cost": [20.0, 10.0, 150.0, 15.0, 8.0],
            "ivt": [60, 50, 40, 90, 120],
            "ovt": [30, 0, 60, 0, 20],
            "freq": [4, 0, 5, 0, 3],
            "income": [40, 40, 40, 60, 60],
            "urban": [1, 1, 1, 0, 0],
            "noalt": [3, 3, 3, 2, 2],
        }
    )
    wide = to_wide(long).set_index("case")
    assert list(wide["choice"]) == ["car", "bus"]
    assert wide.loc[1, ["train_av", "car_av", "bus_av", "air_av"]].tolist() == [1, 1, 0, 1]
    assert wide.loc[2, ["train_av", "car_av", "bus_av", "air_av"]].tolist() == [0, 1, 1, 0]
    assert wide.loc[1, "air_time"] == 100 and wide.loc[2, "bus_time"] == 140
    assert wide.loc[2, "train_cost"] == 0
    assert wide.loc[2, "income"] == 60


def test_modecanada_requires_one_choice_per_case():
    long = pd.DataFrame(
        {"case": [1, 1], "alt": ["train", "car"], "choice": [1, 1], "dist": 1, "cost": 1.0,
         "ivt": 1, "ovt": 1, "freq": 1, "income": 1, "urban": 1, "noalt": 2}
    )
    with pytest.raises(ValueError, match="exactly one chosen"):
        to_wide(long)
