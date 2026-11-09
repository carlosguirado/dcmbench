"""
Versioned, verified datasets with their choice-problem descriptions.

    ds = dcmbench.data.load("swissmetro").prepare("benchmark")
    ds.frame, ds.problem, ds.info, ds.provenance

    mine = dcmbench.data.from_frame(my_table, my_problem, name="my survey")   # your own data
"""

from dcmbench.data.dataset import Dataset, DatasetEntry, DatasetInfo, from_frame
from dcmbench.data.download import DataFile, DataIntegrityError, cache_dir
from dcmbench.data.registry import list_datasets, load, register

__all__ = [
    "Dataset",
    "DatasetEntry",
    "DatasetInfo",
    "DataFile",
    "DataIntegrityError",
    "cache_dir",
    "from_frame",
    "list_datasets",
    "load",
    "register",
]
