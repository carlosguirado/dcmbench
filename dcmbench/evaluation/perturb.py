"""Changing alternative attributes in the data, the basis of every behavioral analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from dcmbench.problem import ChoiceProblem

Amount = Union[float, np.ndarray]


def attribute_column(problem: ChoiceProblem, attribute: str, alternative: str) -> str:
    if attribute not in problem.attributes:
        raise KeyError(f"{problem.name} has no attribute {attribute!r}; available: {sorted(problem.attributes)}")
    columns = problem.attributes[attribute].columns
    if alternative not in columns:
        raise KeyError(f"alternative {alternative!r} has no {attribute!r} attribute; it is defined for {sorted(columns)}")
    return columns[alternative]


def shift(data: pd.DataFrame, problem: ChoiceProblem, attribute: str, alternative: str, amount: Amount) -> pd.DataFrame:
    """Copy of ``data`` with ``amount`` added to one alternative's attribute (scalar or per row)."""
    column = attribute_column(problem, attribute, alternative)
    return data.assign(**{column: data[column].to_numpy(dtype=float) + amount})


@dataclass(frozen=True)
class Perturbation:
    """
    A change to one attribute of some alternatives: ``x * multiply`` or ``x + add``.

    ``alternatives=None`` means every alternative that has the attribute.
    """

    attribute: str
    alternatives: Optional[Tuple[str, ...]] = None
    multiply: Optional[float] = None
    add: Optional[float] = None

    def __post_init__(self):
        if (self.multiply is None) == (self.add is None):
            raise ValueError("give exactly one of multiply= or add=")
        if self.alternatives is not None:
            object.__setattr__(self, "alternatives", tuple(self.alternatives))

    def targets(self, problem: ChoiceProblem) -> Sequence[str]:
        if self.alternatives is None:
            return tuple(problem.attributes[self.attribute].columns)
        return self.alternatives

    def apply(self, data: pd.DataFrame, problem: ChoiceProblem) -> pd.DataFrame:
        changed = {}
        for alternative in self.targets(problem):
            column = attribute_column(problem, self.attribute, alternative)
            values = data[column].to_numpy(dtype=float)
            changed[column] = values * self.multiply if self.multiply is not None else values + self.add
        return data.assign(**changed)

    def describe(self) -> str:
        change = f"× {self.multiply}" if self.multiply is not None else f"+ {self.add}"
        targets = "all alternatives" if self.alternatives is None else ", ".join(self.alternatives)
        return f"{self.attribute} {change} ({targets})"
