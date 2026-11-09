"""Registry of known datasets."""

from __future__ import annotations

import importlib
from typing import Dict, List

import pandas as pd

from dcmbench.data.dataset import Dataset, DatasetEntry
from dcmbench.data.download import fetch
from dcmbench.results import Step

# Built-in datasets live in one module each, exposing ``ENTRY``.
_BUILTIN_MODULES = {
    "swissmetro": "dcmbench.data.datasets.swissmetro",
    "modecanada": "dcmbench.data.datasets.modecanada",
    "lpmc": "dcmbench.data.datasets.lpmc",
}

_entries: Dict[str, DatasetEntry] = {}


def register(entry: DatasetEntry) -> None:
    """Make a dataset available to ``load`` under its name and aliases."""
    for key in (entry.name, *entry.aliases):
        _entries[key] = entry


def get_entry(name: str) -> DatasetEntry:
    if name not in _entries:
        for module_name in _BUILTIN_MODULES.values():
            register(importlib.import_module(module_name).ENTRY)
    if name not in _entries:
        raise KeyError(f"Unknown dataset {name!r}; available: {list_datasets()}")
    return _entries[name]


def list_datasets() -> List[str]:
    for module_name in _BUILTIN_MODULES.values():
        register(importlib.import_module(module_name).ENTRY)
    return sorted({entry.name for entry in _entries.values()})


def load(name: str) -> Dataset:
    """
    Load a dataset's canonical table, downloading and verifying it if needed.

    No rows are dropped silently: every step from the downloaded file to the
    canonical table is recorded in ``Dataset.provenance``.
    """
    entry = get_entry(name)
    path = fetch(entry.file)
    raw = pd.read_csv(path)
    steps = [Step("load", f"read {entry.file.filename}", {"url": entry.file.url, "sha256": entry.file.sha256}, None, len(raw))]
    frame, build_steps = entry.build(raw)
    steps.extend(build_steps)
    dataset = Dataset(
        name=entry.name,
        frame=frame.reset_index(drop=True),
        problem=entry.problem,
        info=entry.info(),
        provenance=tuple(steps),
        preparations=entry.preparations,
    )
    return dataset.validate()
