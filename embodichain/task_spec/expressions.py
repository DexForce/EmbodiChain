# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Closed predicate AST with explicit thresholds and SI normalization."""

from __future__ import annotations

from decimal import Decimal
import re
from typing import Any

from ._json import fields, items, json_copy, text
from .registry import registry_snapshot

__all__ = ["validate_expression"]

_UNITS = {
    "m": ("length", 0, "m"),
    "cm": ("length", -2, "m"),
    "mm": ("length", -3, "m"),
    "s": ("time", 0, "s"),
    "ms": ("time", -3, "s"),
    "rad": ("angle", 0, "rad"),
}
_SPECS = registry_snapshot()["predicates"]
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


def _quantity(
    value: object, dimension: str, path: str, *, signed: bool = False
) -> dict[str, Any]:
    quantity = fields(value, {"value", "unit"}, set(), path)
    number = quantity["value"]
    unit = text(quantity["unit"], f"{path}.unit")
    if type(number) not in (int, float, str):
        raise ValueError(f"{path}.value must be a decimal quantity")
    token = str(number)
    if len(token) > 128 or _DECIMAL.fullmatch(token) is None:
        raise ValueError(f"{path}.value must be a bounded finite decimal")
    exponent_token = token.lower().partition("e")[2]
    if exponent_token and not -256 <= int(exponent_token) <= 256:
        raise ValueError(f"{path}.value exceeds decimal exponent limit")
    decimal = Decimal(token)
    if not signed and decimal < 0:
        raise ValueError(f"{path}.value must be nonnegative")
    if not -256 <= decimal.as_tuple().exponent <= 256:
        raise ValueError(f"{path}.value exceeds decimal exponent limit")
    if unit not in _UNITS or _UNITS[unit][0] != dimension:
        raise ValueError(f"{path} unsupported {dimension} unit {unit!r}")
    _, shift, si = _UNITS[unit]
    sign, digits, exponent = decimal.as_tuple()
    # Constructing a tuple is exact and independent of the ambient context.
    normalized = Decimal((sign, digits, exponent + shift))
    if normalized.is_zero():
        token = "0"
    else:
        token = format(normalized, "f")
        if "." in token:
            token = token.rstrip("0").rstrip(".")
    if len(token) > 128:
        raise ValueError(f"{path}.value exceeds canonical decimal length limit")
    return {"value": token, "unit": si}


def _expression(
    value: object, roles: dict[str, Any] | None, depth: int = 0
) -> dict[str, Any]:
    if depth > 16:
        raise ValueError("expression exceeds TaskSpec depth limit")
    if not isinstance(value, dict):
        raise ValueError("expression must be an object")
    if "op" in value:
        fields(value, {"op", "args"}, set(), "expression")
        op = text(value["op"], "expression.op")
        if op not in {"and", "or", "not"}:
            raise ValueError(f"unsupported logical operator {op!r}")
        args = items(value["args"], "expression.args", nonempty=True)
        if op == "not" and len(args) != 1:
            raise ValueError("not requires exactly one argument")
        return {"op": op, "args": [_expression(arg, roles, depth + 1) for arg in args]}
    name = text(value.get("predicate"), "expression.predicate")
    spec = _SPECS.get(name)
    if spec is None:
        raise ValueError(f"unsupported predicate {name!r}")
    required = {"predicate"} | spec["roles"].keys() | spec["quantities"].keys()
    if name == "relative_position":
        required |= {"relation"}
    fields(value, required, set(), f"predicate {name}")
    result: dict[str, Any] = {"predicate": name}
    for key, kinds in spec["roles"].items():
        role = text(value[key], f"{name}.{key}")
        if roles is not None and (
            role not in roles or roles[role]["kind"] not in kinds
        ):
            raise ValueError(
                f"{name}.{key} has an unknown or incompatible role {role!r}"
            )
        result[key] = role
    if len(set(result[key] for key in spec["roles"])) != len(spec["roles"]):
        raise ValueError(f"{name} requires distinct role references")
    for key, dimension in spec["quantities"].items():
        result[key] = _quantity(
            value[key], dimension, f"{name}.{key}", signed=key == "position"
        )
        if key == "duration" and Decimal(result[key]["value"]) <= 0:
            raise ValueError("held_stable.duration must be positive")
    if name == "relative_position":
        relation = text(value["relation"], "relative_position.relation")
        if relation not in {"left_of", "right_of", "in_front_of", "behind"}:
            raise ValueError(f"unsupported relative_position relation {relation!r}")
        result["relation"] = relation
    return result


def validate_expression(value: object, path: str = "expression") -> dict[str, Any]:
    """Validate and normalize one bounded expression, without evaluating it.

    Args:
        value: JSON predicate or and/or/not expression with explicit quantities.
        path: Diagnostic context included in validation errors.

    Returns:
        Detached expression with lengths in meters, time in seconds and angles
        in radians. Role membership is checked by the template decoder.

    Raises:
        ValueError: Unknown fields, predicates, units, malformed or oversized data.
    """
    try:
        return _expression(json_copy(value), None)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
