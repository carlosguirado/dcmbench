"""
The engine-neutral expression language of spec v2.

A formula such as ``"ASC + B_COST * log(1 + COST) * (GA == 0)"`` is parsed
with Python's ``ast`` into a small allow-listed tree:

- numbers and names (parameters, random parameters or data columns);
- ``+ - * / **`` and unary minus;
- ``log(x)`` and ``exp(x)``;
- comparisons ``== != < <= > >=``, used as 0/1 indicators.

Anything else is rejected. The tree is evaluated with numpy or torch,
differentiated symbolically, and compiled to Biogeme expressions.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Union


class ExpressionError(ValueError):
    """A formula uses syntax outside the spec v2 expression language."""


@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Sym:
    name: str


@dataclass(frozen=True)
class BinOp:
    op: str  # + - * / **
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class Neg:
    operand: "Node"


@dataclass(frozen=True)
class Call:
    function: str  # log, exp
    argument: "Node"


@dataclass(frozen=True)
class Compare:
    op: str  # == != < <= > >=
    left: "Node"
    right: "Node"


Node = Union[Num, Sym, BinOp, Neg, Call, Compare]

_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
_COMPARES = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
FUNCTIONS = ("log", "exp")


def parse(formula: str) -> Node:
    """Parse a formula string; raises ``ExpressionError`` on anything not allowed."""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"cannot parse {formula!r}: {exc.msg}") from None
    return _convert(tree.body, formula)


def _convert(node: ast.AST, formula: str) -> Node:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Num(float(node.value))
    if isinstance(node, ast.Name):
        if node.id in FUNCTIONS:
            raise ExpressionError(f"{node.id} must be called, as in {node.id}(x), in {formula!r}")
        return Sym(node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return BinOp(_BINOPS[type(node.op)], _convert(node.left, formula), _convert(node.right, formula))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return Neg(_convert(node.operand, formula))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _convert(node.operand, formula)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise ExpressionError(f"only {', '.join(FUNCTIONS)} may be called, in {formula!r}")
        if len(node.args) != 1 or node.keywords:
            raise ExpressionError(f"{node.func.id}() takes exactly one argument, in {formula!r}")
        return Call(node.func.id, _convert(node.args[0], formula))
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or type(node.ops[0]) not in _COMPARES:
            raise ExpressionError(f"use one comparison at a time, e.g. (a < b) * (b < c), in {formula!r}")
        return Compare(_COMPARES[type(node.ops[0])], _convert(node.left, formula), _convert(node.comparators[0], formula))
    raise ExpressionError(f"{type(node).__name__} is not allowed in spec formulas: {formula!r}")


def symbols(node: Node) -> FrozenSet[str]:
    """Every name used in ``node``."""
    if isinstance(node, Sym):
        return frozenset([node.name])
    if isinstance(node, Num):
        return frozenset()
    if isinstance(node, (BinOp, Compare)):
        return symbols(node.left) | symbols(node.right)
    if isinstance(node, Neg):
        return symbols(node.operand)
    return symbols(node.argument)


def to_string(node: Node) -> str:
    if isinstance(node, Num):
        return repr(node.value)
    if isinstance(node, Sym):
        return node.name
    if isinstance(node, (BinOp, Compare)):
        return f"({to_string(node.left)} {node.op} {to_string(node.right)})"
    if isinstance(node, Neg):
        return f"(-{to_string(node.operand)})"
    return f"{node.function}({to_string(node.argument)})"


# --- evaluation -----------------------------------------------------------------------------------

_ARITHMETIC: Dict[str, Callable] = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "**": operator.pow,
}
_COMPARISON: Dict[str, Callable] = {
    "==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
}


def evaluate(node: Node, env: Dict[str, object], backend) -> object:
    """
    Evaluate ``node`` with values from ``env`` (numbers, numpy arrays or torch
    tensors). ``backend`` supplies ``log``, ``exp`` and ``indicator`` (bool → float).
    """
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Sym):
        if node.name not in env:
            raise KeyError(f"no value for {node.name!r}")
        return env[node.name]
    if isinstance(node, BinOp):
        return _ARITHMETIC[node.op](evaluate(node.left, env, backend), evaluate(node.right, env, backend))
    if isinstance(node, Neg):
        return -evaluate(node.operand, env, backend)
    if isinstance(node, Call):
        return getattr(backend, node.function)(evaluate(node.argument, env, backend))
    return backend.indicator(_COMPARISON[node.op](evaluate(node.left, env, backend), evaluate(node.right, env, backend)))


# --- symbolic differentiation ---------------------------------------------------------------------

ZERO, ONE = Num(0.0), Num(1.0)


def _add(a: Node, b: Node) -> Node:
    if a == ZERO:
        return b
    if b == ZERO:
        return a
    return BinOp("+", a, b)


def _sub(a: Node, b: Node) -> Node:
    if b == ZERO:
        return a
    if a == ZERO:
        return Neg(b)
    return BinOp("-", a, b)


def _mul(a: Node, b: Node) -> Node:
    if a == ZERO or b == ZERO:
        return ZERO
    if a == ONE:
        return b
    if b == ONE:
        return a
    return BinOp("*", a, b)


def _div(a: Node, b: Node) -> Node:
    if a == ZERO:
        return ZERO
    if b == ONE:
        return a
    return BinOp("/", a, b)


def derivative(node: Node, name: str) -> Node:
    """∂node/∂name. Comparisons are treated as locally constant (derivative 0)."""
    if name not in symbols(node):
        return ZERO
    if isinstance(node, Sym):
        return ONE
    if isinstance(node, Neg):
        inner = derivative(node.operand, name)
        return ZERO if inner == ZERO else Neg(inner)
    if isinstance(node, Compare):
        return ZERO
    if isinstance(node, Call):
        inner = derivative(node.argument, name)
        outer = node if node.function == "exp" else BinOp("/", ONE, node.argument)
        return _mul(outer, inner)
    a, b = node.left, node.right
    da, db = derivative(a, name), derivative(b, name)
    if node.op == "+":
        return _add(da, db)
    if node.op == "-":
        return _sub(da, db)
    if node.op == "*":
        return _add(_mul(da, b), _mul(a, db))
    if node.op == "/":
        return _div(_sub(_mul(da, b), _mul(a, db)), BinOp("**", b, Num(2.0)))
    # a ** b
    if db == ZERO:
        return _mul(_mul(b, BinOp("**", a, _sub(b, ONE))), da)
    return _mul(node, _add(_mul(db, Call("log", a)), _div(_mul(b, da), a)))
