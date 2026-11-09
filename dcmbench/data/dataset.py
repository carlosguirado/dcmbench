"""Datasets: a canonical table, its choice problem, and how it was prepared."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from dcmbench.data.download import DataFile
from dcmbench.problem import ChoiceProblem
from dcmbench.results import Step


@dataclass(frozen=True)
class DatasetInfo:
    """Bibliographic and version information for a dataset."""

    description: str
    citation: str
    source_url: str
    license: Optional[str]
    data_version: str
    sha256: str
    url: str


@dataclass(frozen=True)
class DatasetEntry:
    """
    How to obtain one dataset.

    ``build`` turns the downloaded file's DataFrame into the canonical wide
    table and returns the steps it applied, so they appear in provenance.
    ``preparations`` are named, documented sample definitions (e.g. the benchmark sample).
    """

    name: str
    file: DataFile
    problem: ChoiceProblem
    build: Callable[[pd.DataFrame], Tuple[pd.DataFrame, Sequence[Step]]]
    description: str
    citation: str
    source_url: str
    license: Optional[str]
    data_version: str
    aliases: Tuple[str, ...] = ()
    preparations: Mapping[str, Callable[["Dataset"], "Dataset"]] = field(default_factory=dict)

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            description=self.description,
            citation=self.citation,
            source_url=self.source_url,
            license=self.license,
            data_version=self.data_version,
            sha256=self.file.sha256,
            url=self.file.url,
        )


@dataclass(frozen=True, eq=False)
class Dataset:
    """
    A canonical table with its choice problem and the steps that produced it.

    Methods never modify the dataset in place; each returns a new one with
    the step appended to ``provenance``.
    """

    name: str
    frame: pd.DataFrame
    problem: ChoiceProblem
    info: DatasetInfo
    provenance: Tuple[Step, ...] = ()
    preparations: Mapping[str, Callable[["Dataset"], "Dataset"]] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.frame)

    def _with(self, frame: pd.DataFrame, step: Step) -> "Dataset":
        return replace(self, frame=frame.reset_index(drop=True), provenance=self.provenance + (step,))

    def filter(self, condition: Union[str, Callable[[pd.DataFrame], pd.Series]], *, label: str) -> "Dataset":
        """
        Keep rows matching ``condition``: a ``DataFrame.query`` string or a
        function returning a boolean mask. ``label`` describes the filter in provenance.
        """
        if isinstance(condition, str):
            mask = self.frame.eval(condition)
            params: Dict[str, Any] = {"query": condition}
        else:
            mask = condition(self.frame)
            params = {"function": getattr(condition, "__name__", repr(condition))}
        mask = pd.Series(mask, index=self.frame.index).astype(bool)
        step = Step("filter", label, params, len(self.frame), int(mask.sum()))
        return self._with(self.frame[mask], step)

    def subsample(self, n: int, *, seed: int) -> "Dataset":
        """Simple random sample of ``n`` rows without replacement (``DataFrame.sample``)."""
        if n > len(self.frame):
            raise ValueError(f"cannot sample {n} rows from {len(self.frame)}")
        step = Step("subsample", f"random sample of {n} rows", {"n": n, "seed": seed}, len(self.frame), n)
        return self._with(self.frame.sample(n=n, random_state=seed), step)

    def derive(self, label: str, **columns: Callable[[pd.DataFrame], Any]) -> "Dataset":
        """
        Add columns computed from the table, e.g. dummies:
        ``ds.derive("purpose dummies", HBW=lambda f: (f.purpose == "HBW").astype(int))``.
        Existing columns cannot be overwritten.
        """
        clash = set(columns) & set(self.frame.columns)
        if clash:
            raise ValueError(f"columns already exist: {sorted(clash)}")
        frame = self.frame.assign(**{name: function(self.frame) for name, function in columns.items()})
        step = Step("derive", label, {"columns": sorted(columns)}, len(self.frame), len(frame))
        return self._with(frame, step)

    def prepare(self, name: str) -> "Dataset":
        """Apply a named preparation, e.g. ``"benchmark"``."""
        if name not in self.preparations:
            raise KeyError(f"{self.name} has no preparation {name!r}; available: {sorted(self.preparations)}")
        return self.preparations[name](self)

    def validate(self) -> "Dataset":
        """Check the table against the choice problem; returns self for chaining."""
        self.problem.validate(self.frame)
        return self


def from_frame(
    frame: pd.DataFrame,
    problem: ChoiceProblem,
    name: str,
    *,
    description: str = "",
    citation: str = "",
    source_url: str = "",
    license: Optional[str] = None,
) -> Dataset:
    """
    Wrap your own table as a ``Dataset`` so it can be evaluated like the built-in ones.

    The table is validated against ``problem``. Its content hash (pandas row
    hashes, sha256) is recorded so results can be traced to the exact data.
    """
    import hashlib

    rows = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    digest = hashlib.sha256(rows.tobytes() + ",".join(map(str, frame.columns)).encode()).hexdigest()
    info = DatasetInfo(description=description, citation=citation, source_url=source_url, license=license,
                       data_version="user-supplied", sha256=digest, url="")
    step = Step("load", "user-supplied table", {"sha256": digest, "columns": list(map(str, frame.columns))},
                None, len(frame))
    dataset = Dataset(name=name, frame=frame.reset_index(drop=True), problem=problem, info=info, provenance=(step,))
    return dataset.validate()
