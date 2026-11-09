"""
Read 0.1.x ("v1") JSON specs.

v1 specs number alternatives (``"1"``, ``"2"``, ...) and carry their own choice
column and availability. ``convert`` turns one into a spec v2 plus the
``ChoiceProblem`` it implies; alternative names are the v1 keys.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping, Tuple

from dcmbench.problem import ChoiceProblem
from dcmbench.specs.spec import Nest, Parameter, Spec, SpecError


def model_type(spec_v1: Mapping[str, Any]) -> str:
    return str(spec_v1.get("metadata", {}).get("model_type", "MNL")).upper()


def check_supported(spec_v1: Mapping[str, Any]) -> None:
    """v1 cannot express random parameters; fail early with a pointer to spec v2."""
    if model_type(spec_v1) == "MXL":
        if spec_v1.get("random_parameters"):
            raise NotImplementedError(
                "v1 specs cannot express mixed logit random parameters (the v1 builder never created "
                "bioDraws). Write a spec v2 with random_parameters and draws, or use FunctionModelFitter."
            )
        warnings.warn(
            "model_type='MXL' but no random_parameters; this will estimate plain MNL. "
            "Write a spec v2 for genuine mixed logit.",
            UserWarning,
            stacklevel=3,
        )


def _code(key: str):
    return int(key) if key.isdigit() else key


def convert(spec_v1: Mapping[str, Any], choice_column: str = "CHOICE") -> Tuple[Spec, ChoiceProblem]:
    alternatives = tuple(str(k) for k in spec_v1["utilities"])
    kind = model_type(spec_v1)
    kind = "MNL" if kind == "MXL" else kind
    parameters: Dict[str, Parameter] = {
        name: Parameter(
            name,
            start=float(p.get("initial_value", 0)),
            lower=p.get("lower_bound"),
            upper=p.get("upper_bound"),
            fixed=bool(p.get("fixed", False)),
        )
        for name, p in spec_v1["parameters"].items()
    }
    nests = tuple(
        Nest(name, n["nest_param"], tuple(str(a) for a in n["alternatives"]))
        for name, n in spec_v1.get("nests", {}).items()
    )
    if kind == "NL" and not nests:
        raise SpecError("v1 NL spec without nests")
    spec = Spec(
        name=spec_v1.get("metadata", {}).get("name", "v1_spec"),
        model=kind,
        alternatives=alternatives,
        parameters=parameters,
        utilities={k: spec_v1["utilities"][k]["formula"] for k in alternatives},
        nests=nests if kind == "NL" else (),
        description="converted from a v1 spec",
    )
    availability = {}
    for key, expression in spec_v1.get("availability", {}).items():
        expression = str(expression).strip()
        if expression in ("1", "1.0"):
            availability[str(key)] = None
        elif expression.isidentifier():
            availability[str(key)] = expression
        else:
            raise SpecError(f"v1 availability for alternative {key} must be 1 or a column name, not {expression!r}")
    problem = ChoiceProblem(
        name=spec.name,
        alternatives=alternatives,
        choice_column=choice_column,
        choice_codes={_code(k): k for k in alternatives},
        availability=availability,
    )
    return spec, problem
