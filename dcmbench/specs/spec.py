"""
Spec v2: a JSON model specification that any supported engine can estimate.

    {
      "spec_version": 2,
      "name": "swissmetro_mnl",
      "model": "MNL" | "NL" | "MXL",
      "alternatives": ["train", "sm", "car"],
      "parameters": {"ASC_TRAIN": {"start": 0}, "MU": {"start": 1.5, "lower": 1, "upper": 10},
                     "ASC_SM": {"start": 0, "fixed": true}},
      "utilities": {"train": "ASC_TRAIN + B_TIME * TRAIN_TT / 100", ...},
      "nests": [{"name": "existing", "parameter": "MU", "alternatives": ["train", "car"]}],
      "random_parameters": {"B_TIME_RND": {"distribution": "normal", "mean": "B_TIME", "spread": "B_TIME_S"}},
      "draws": {"number": 200, "seed": 42, "type": "pseudo", "panel": false}
    }

Choice codes and availability come from the dataset's ``ChoiceProblem``.
Formulas use the expression language in ``dcmbench.specs.expression``;
names that are neither parameters nor random parameters are data columns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple, Union

from dcmbench.specs.expression import Node, parse, symbols

MODELS = ("MNL", "NL", "MXL")
DISTRIBUTIONS = ("normal", "lognormal", "uniform", "triangular")
DRAW_TYPES = ("pseudo", "halton")


class SpecError(ValueError):
    """The specification is invalid."""


@dataclass(frozen=True)
class Parameter:
    name: str
    start: float = 0.0
    lower: Optional[float] = None
    upper: Optional[float] = None
    fixed: bool = False

    def __post_init__(self):
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise SpecError(f"parameter {self.name}: lower bound exceeds upper bound")
        if (self.lower is not None and self.start < self.lower) or (self.upper is not None and self.start > self.upper):
            raise SpecError(f"parameter {self.name}: start {self.start} is outside its bounds")


@dataclass(frozen=True)
class RandomParameter:
    """
    A random coefficient built from a standard draw ξ:

    - normal:     mean + spread·ξ, ξ ~ N(0, 1)
    - lognormal:  sign·exp(mean + spread·ξ), ξ ~ N(0, 1)
    - uniform:    mean + spread·(2u − 1), u ~ U(0, 1)
    - triangular: mean + spread·t, t ~ symmetric triangular on [−1, 1]
    """

    name: str
    distribution: str
    mean: str
    spread: str
    sign: float = 1.0

    def __post_init__(self):
        if self.distribution not in DISTRIBUTIONS:
            raise SpecError(f"random parameter {self.name}: distribution must be one of {DISTRIBUTIONS}")
        if self.sign not in (1.0, -1.0):
            raise SpecError(f"random parameter {self.name}: sign must be 1 or -1")
        if self.sign != 1.0 and self.distribution != "lognormal":
            raise SpecError(f"random parameter {self.name}: sign applies to lognormal parameters only")

    @property
    def base(self) -> str:
        """The standard draw it is built from."""
        return "normal" if self.distribution in ("normal", "lognormal") else "uniform"


@dataclass(frozen=True)
class Nest:
    name: str
    parameter: str
    alternatives: Tuple[str, ...]


@dataclass(frozen=True)
class Draws:
    number: int = 1000
    seed: int = 0
    type: str = "pseudo"
    panel: bool = False

    def __post_init__(self):
        if self.number < 1:
            raise SpecError("draws.number must be positive")
        if self.type not in DRAW_TYPES:
            raise SpecError(f"draws.type must be one of {DRAW_TYPES}")


@dataclass(frozen=True)
class Spec:
    name: str
    model: str
    alternatives: Tuple[str, ...]
    parameters: Mapping[str, Parameter]
    utilities: Mapping[str, str]
    nests: Tuple[Nest, ...] = ()
    random_parameters: Mapping[str, RandomParameter] = field(default_factory=dict)
    draws: Optional[Draws] = None
    description: str = ""
    parsed: Mapping[str, Node] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        if self.model not in MODELS:
            raise SpecError(f"model must be one of {MODELS}, not {self.model!r}")
        object.__setattr__(self, "alternatives", tuple(self.alternatives))
        if set(self.utilities) != set(self.alternatives):
            raise SpecError("utilities must define exactly one formula per alternative")
        object.__setattr__(self, "parsed", {alt: parse(self.utilities[alt]) for alt in self.alternatives})

        names = set(self.parameters)
        clash = names & set(self.random_parameters)
        if clash:
            raise SpecError(f"names used for both parameters and random parameters: {sorted(clash)}")
        for rp in self.random_parameters.values():
            missing = {rp.mean, rp.spread} - names
            if missing:
                raise SpecError(f"random parameter {rp.name} refers to undefined parameters {sorted(missing)}")

        if self.model == "NL":
            if not self.nests:
                raise SpecError("an NL spec needs nests")
            seen = set()
            for nest in self.nests:
                if nest.parameter not in names:
                    raise SpecError(f"nest {nest.name}: parameter {nest.parameter} is not defined")
                unknown = set(nest.alternatives) - set(self.alternatives)
                if unknown or seen & set(nest.alternatives):
                    raise SpecError(f"nest {nest.name}: unknown or repeated alternatives")
                seen |= set(nest.alternatives)
        elif self.nests:
            raise SpecError(f"nests are only allowed in NL specs (this is {self.model})")

        if self.model == "MXL":
            if not self.random_parameters or self.draws is None:
                raise SpecError("an MXL spec needs random_parameters and draws")
        elif self.random_parameters:
            raise SpecError(f"random parameters are only allowed in MXL specs (this is {self.model})")

        used = self.used_symbols()
        unused = names - used - {rp.mean for rp in self.random_parameters.values()} \
            - {rp.spread for rp in self.random_parameters.values()} - {n.parameter for n in self.nests}
        if unused:
            raise SpecError(f"parameters defined but never used: {sorted(unused)}")

    # --- names ------------------------------------------------------------------------------------

    def used_symbols(self) -> FrozenSet[str]:
        return frozenset().union(*(symbols(node) for node in self.parsed.values()))

    def data_columns(self) -> FrozenSet[str]:
        """Data columns the utilities refer to."""
        return self.used_symbols() - set(self.parameters) - set(self.random_parameters)

    def free_parameters(self) -> Tuple[str, ...]:
        return tuple(name for name, p in self.parameters.items() if not p.fixed)

    @property
    def n_free_parameters(self) -> int:
        return len(self.free_parameters())

    # --- (de)serialization ------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "spec_version": 2,
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "alternatives": list(self.alternatives),
            "parameters": {
                n: {k: v for k, v in (("start", p.start), ("lower", p.lower), ("upper", p.upper), ("fixed", p.fixed))
                    if not (k in ("lower", "upper") and v is None) and not (k == "fixed" and not v)}
                for n, p in self.parameters.items()
            },
            "utilities": dict(self.utilities),
        }
        if self.nests:
            data["nests"] = [{"name": n.name, "parameter": n.parameter, "alternatives": list(n.alternatives)}
                             for n in self.nests]
        if self.random_parameters:
            data["random_parameters"] = {
                n: {"distribution": rp.distribution, "mean": rp.mean, "spread": rp.spread,
                    **({"sign": rp.sign} if rp.sign != 1.0 else {})}
                for n, rp in self.random_parameters.items()
            }
        if self.draws is not None:
            data["draws"] = {"number": self.draws.number, "seed": self.draws.seed,
                             "type": self.draws.type, "panel": self.draws.panel}
        return data

    def sha256(self) -> str:
        """Hash of the canonical JSON form, for provenance."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Spec":
        if data.get("spec_version") != 2:
            raise SpecError("not a spec v2 document (spec_version must be 2); see dcmbench.specs.v1 for old specs")
        allowed = {"spec_version", "name", "description", "model", "alternatives", "parameters", "utilities",
                   "nests", "random_parameters", "draws"}
        unknown = set(data) - allowed
        if unknown:
            raise SpecError(f"unknown spec fields: {sorted(unknown)}")
        try:
            return cls(
                name=data["name"],
                description=data.get("description", ""),
                model=data["model"],
                alternatives=tuple(data["alternatives"]),
                parameters={n: Parameter(n, **p) for n, p in data["parameters"].items()},
                utilities=dict(data["utilities"]),
                nests=tuple(Nest(n["name"], n["parameter"], tuple(n["alternatives"])) for n in data.get("nests", [])),
                random_parameters={n: RandomParameter(n, **rp) for n, rp in data.get("random_parameters", {}).items()},
                draws=Draws(**data["draws"]) if "draws" in data else None,
            )
        except KeyError as exc:
            raise SpecError(f"spec is missing required field {exc}") from None
        except TypeError as exc:
            raise SpecError(f"invalid spec field: {exc}") from None


def load_spec(source: Union[str, Path, Mapping[str, Any]]) -> Spec:
    """A ``Spec`` from a dict, a JSON string, or a path to a JSON file."""
    if isinstance(source, Mapping):
        return Spec.from_dict(source)
    text = str(source)
    if text.lstrip().startswith("{"):
        return Spec.from_dict(json.loads(text))
    return Spec.from_dict(json.loads(Path(source).read_text()))
