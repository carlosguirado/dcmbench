"""
Versioned, verified datasets with their choice-problem descriptions.

    ds = dcmbench.data.load("swissmetro").prepare("benchmark")
    ds.frame, ds.problem, ds.info, ds.provenance
"""

from dcmbench.data.dataset import Dataset, DatasetEntry, DatasetInfo
from dcmbench.data.download import DataFile, DataIntegrityError, cache_dir
from dcmbench.data.registry import list_datasets, load, register

__all__ = [
    "Dataset",
    "DatasetEntry",
    "DatasetInfo",
    "DataFile",
    "DataIntegrityError",
    "cache_dir",
    "list_datasets",
    "load",
    "register",
]
