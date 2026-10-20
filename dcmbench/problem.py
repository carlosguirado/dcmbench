"""Description of a choice problem over a wide-format table (one row per observation)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

# Hours per unit of time.
TIME_UNITS: Dict[str, float] = {"second": 1 / 3600, "minute": 1 / 60, "hour": 1.0}
CURRENCIES = ("CHF", "CAD", "GBP", "USD", "EUR")

_COST_UNIT = re.compile(r"^(?:(?P<scale>\d+(?:\.\d+)?)\s+)?(?P<currency>[A-Z]{3})$")

AttributeKind = Literal["time", "cost", "other"]


def parse_cost_unit(unit: str) -> Tuple[str, float]:
    """Split a cost unit such as ``"CHF"`` or ``"0.01 CHF"`` into (currency, scale)."""
    match = _COST_UNIT.match(unit)
    if match is None or match["currency"] not in CURRENCIES:
        raise ValueError(
            f"Unknown cost unit {unit!r}; expected a currency in {CURRENCIES}, "
            "optionally preceded by a scale (e.g. '0.01 CHF')"
        )
    return match["currency"], float(match["scale"] or 1.0)


def hours_per_unit(unit: str) -> float:
    """Hours in one ``unit`` of time."""
    if unit not in TIME_UNITS:
        raise ValueError(f"Unknown time unit {unit!r}; expected one of {sorted(TIME_UNITS)}")
    return TIME_UNITS[unit]


def to_currency_per_hour(value, time_unit: str, cost_unit: str):
    """
    Convert a value of time from ``cost_unit`` per ``time_unit`` to currency per hour.

    Returns ``(converted, currency)``. Works elementwise on arrays.
    """
    currency, scale = parse_cost_unit(cost_unit)
    return np.asarray(value) * scale / hours_per_unit(time_unit), currency


@dataclass(frozen=True)
class Attribute:
    """One attribute of the alternatives, e.g. travel time, with a column per alternative."""

    columns: Mapping[str, str]
    unit: str
    kind: AttributeKind = "other"

    def __post_init__(self):
        if not self.columns:
            raise ValueError("Attribute.columns must name at least one alternative")
        if self.kind == "time":
            hours_per_unit(self.unit)
        elif self.kind == "cost":
            parse_cost_unit(self.unit)
        elif self.kind != "other":
            raise ValueError(f"Attribute.kind must be 'time', 'cost' or 'other', not {self.kind!r}")


@dataclass(frozen=True)
class ChoiceProblem:
    """
    Alternatives, choice, availability and attributes of one choice problem.

    The description never modifies the data: raw choice codes stay in
    ``choice_column`` and are mapped to alternative names by ``choice_codes``.
    Probability tables are ordered by ``alternatives``.
    """

    name: str
    alternatives: Tuple[str, ...]
    choice_column: str
    choice_codes: Mapping[Any, str]
    availability: Mapping[str, Optional[str]] = field(default_factory=dict)
    attributes: Mapping[str, Attribute] = field(default_factory=dict)
    group_column: Optional[str] = None
    weight_column: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "alternatives", tuple(self.alternatives))
        alternatives = set(self.alternatives)
        if not self.alternatives or len(alternatives) != len(self.alternatives):
            raise ValueError("alternatives must be a non-empty sequence of unique names")

        coded = set(self.choice_codes.values())
        if not coded <= alternatives:
            raise ValueError(f"choice_codes map to unknown alternatives: {sorted(coded - alternatives)}")
        if coded != alternatives:
            raise ValueError(f"choice_codes has no code for alternatives: {sorted(alternatives - coded)}")

        unknown = set(self.availability) - alternatives
        if unknown:
            raise ValueError(f"availability names unknown alternatives: {sorted(unknown)}")
        for name, attribute in self.attributes.items():
            unknown = set(attribute.columns) - alternatives
            if unknown:
                raise ValueError(f"attribute {name!r} names unknown alternatives: {sorted(unknown)}")

    @property
    def choice_mapping(self) -> Dict[Any, str]:
        """``{choice code -> probability column}``, as taken by ``InternalValidator``."""
        return dict(self.choice_codes)

    def availability_column(self, alternative: str) -> Optional[str]:
        return self.availability.get(alternative)

    def attribute_kind(self, kind: AttributeKind) -> Dict[str, Attribute]:
        return {name: a for name, a in self.attributes.items() if a.kind == kind}

    def required_columns(self) -> Tuple[str, ...]:
        columns = [self.choice_column]
        columns += [c for c in self.availability.values() if c is not None]
        for attribute in self.attributes.values():
            columns += list(attribute.columns.values())
        columns += [c for c in (self.group_column, self.weight_column) if c is not None]
        return tuple(dict.fromkeys(columns))

    def choices(self, data: pd.DataFrame) -> pd.Series:
        """Chosen alternative name per observation."""
        chosen = data[self.choice_column].map(self.choice_codes)
        if chosen.isna().any():
            unknown = data.loc[chosen.isna(), self.choice_column].value_counts()
            raise ValueError(
                f"{self.name}: choice codes not in choice_codes: {unknown.to_dict()}"
            )
        return chosen.rename("choice")

    def availability_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Boolean ``n_obs × alternatives`` table; alternatives without a column are always available."""
        matrix = {}
        for alternative in self.alternatives:
            column = self.availability_column(alternative)
            matrix[alternative] = (
                np.ones(len(data), dtype=bool) if column is None else data[column].to_numpy() == 1
            )
        return pd.DataFrame(matrix, index=data.index)

    def validate(self, data: pd.DataFrame) -> None:
        """Raise ``ValueError`` if ``data`` does not fit this description."""
        missing = [c for c in self.required_columns() if c not in data.columns]
        if missing:
            raise ValueError(f"{self.name}: columns missing from data: {missing}")

        chosen = self.choices(data)

        for alternative in self.alternatives:
            column = self.availability_column(alternative)
            if column is not None and not data[column].isin([0, 1]).all():
                raise ValueError(f"{self.name}: availability column {column!r} must contain only 0/1")

        available = self.availability_matrix(data)
        chosen_available = available.to_numpy()[
            np.arange(len(data)), available.columns.get_indexer(chosen)
        ]
        if not chosen_available.all():
            raise ValueError(
                f"{self.name}: {int((~chosen_available).sum())} observations chose an unavailable alternative"
            )

        for name, attribute in self.attributes.items():
            for column in attribute.columns.values():
                values = data[column]
                if not pd.api.types.is_numeric_dtype(values) or not np.isfinite(values).all():
                    raise ValueError(f"{self.name}: attribute {name!r} column {column!r} must be finite numbers")

        for column in (self.group_column, self.weight_column):
            if column is not None and data[column].isna().any():
                raise ValueError(f"{self.name}: column {column!r} has missing values")
