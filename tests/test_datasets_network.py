"""
Built-in datasets against the 0.1.x benchmark samples (downloads ~3 MB once).

Opt in with ``DCMBENCH_NETWORK_TESTS=1``; files are cached in
``$DCMBENCH_DATA_DIR`` (default ``~/.cache/dcmbench``).
"""

import os

import pandas as pd
import pytest

import dcmbench.data as data

pytestmark = pytest.mark.skipif(
    os.environ.get("DCMBENCH_NETWORK_TESTS") != "1", reason="set DCMBENCH_NETWORK_TESTS=1 to download datasets"
)

# Benchmark samples from the 0.1.x scripts: sample size and mode shares in % (2 dp).
REFERENCE = {
    "swissmetro": (6768, {"train": 13.42, "sm": 60.43, "car": 26.15}),
    "modecanada": (4324, {"train": 14.41, "car": 51.18, "bus": 0.37, "air": 34.04}),
    "lpmc": (5000, {"walk": 17.10, "cycle": 2.84, "pt": 35.06, "drive": 45.00}),
}


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_benchmark_samples_match_reference(name):
    ds = data.load(name)
    if name != "modecanada":
        ds = ds.prepare("benchmark")
    ds.validate()
    n, shares = REFERENCE[name]
    assert len(ds) == n
    observed = ds.problem.choices(ds.frame).value_counts(normalize=True) * 100
    for alternative, share in shares.items():
        assert observed[alternative] == pytest.approx(share, abs=0.005)


def test_lpmc_benchmark_sample_selects_0_1_rows():
    """dcmbench 0.1.x sampled the full file with DataFrame.sample(n=5000, random_state=42)."""
    ds = data.load("lpmc")
    raw = pd.read_csv(data.download.fetch(data.registry.get_entry("lpmc").file))
    expected = raw.sample(n=5000, random_state=42)["trip_id"].tolist()
    assert ds.prepare("benchmark").frame["trip_id"].tolist() == expected


def test_panel_identifiers():
    swissmetro = data.load("swissmetro").prepare("benchmark")
    assert swissmetro.frame.groupby("ID").size().max() == 9
    lpmc = data.load("lpmc")
    assert lpmc.frame["person_id"].nunique() == lpmc.frame.groupby(["household_id", "person_n"]).ngroups
