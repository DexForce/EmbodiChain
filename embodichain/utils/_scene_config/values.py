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
"""Private canonical value validation for feasibility artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from numbers import Real
from enum import Enum
import numpy as np
import sys
from pathlib import Path
from types import MappingProxyType

__all__ = []


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty stripped string")
    return value


def _nonnegative_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _rigid_pose(value: object) -> tuple[tuple[float, ...], ...]:
    try:
        rows = tuple(tuple(row) for row in value)  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError("arena_pose must be a 4x4 rigid transform") from exc
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("arena_pose must be a 4x4 rigid transform")
    if any(
        isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(item)
        for row in rows
        for item in row
    ):
        raise ValueError("arena_pose must contain finite real numbers")
    pose = tuple(
        tuple(float(item) if item != 0 else 0.0 for item in row) for row in rows
    )
    tolerance = 1e-6
    if any(
        abs(pose[3][index] - target) > tolerance
        for index, target in enumerate((0, 0, 0, 1))
    ):
        raise ValueError("arena_pose must have homogeneous last row [0, 0, 0, 1]")
    rotation = tuple(row[:3] for row in pose[:3])
    if any(
        abs(sum(rotation[k][i] * rotation[k][j] for k in range(3)) - (i == j))
        > tolerance
        for i in range(3)
        for j in range(3)
    ):
        raise ValueError("arena_pose rotation must be orthonormal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > tolerance:
        raise ValueError("arena_pose rotation must be proper (determinant +1)")
    return (*pose[:3], (0.0, 0.0, 0.0, 1.0))


def plain(value: object) -> object:
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    # Tensor values can only exist when torch is already loaded by the caller.
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return plain(value.detach().cpu().tolist())
    if isinstance(value, Path) or (
        torch is not None and isinstance(value, torch.device)
    ):
        return str(value)
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, Enum):
        return plain(value.value)
    if isinstance(getattr(value, "name", None), str) and isinstance(
        getattr(value, "value", None), Real
    ):
        return {
            "enum_type": type(value).__module__ + "." + type(value).__qualname__,
            "name": value.name,
            "value": plain(value.value),
        }
    if isinstance(value, (set, frozenset)):
        return sorted(
            (plain(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True)
        )
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def owned_json(value: Mapping[str, object]) -> Mapping[str, object]:
    copied = json.loads(json.dumps(plain(value), allow_nan=False))

    def freeze(item: object) -> object:
        if isinstance(item, dict):
            return MappingProxyType({k: freeze(v) for k, v in item.items()})
        if isinstance(item, list):
            return tuple(freeze(v) for v in item)
        return item

    return freeze(copied)


def finite(values: object, name: str) -> tuple[float, ...]:
    result = tuple(values)
    if any(
        isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
        for v in result
    ):
        raise ValueError(f"{name} requires finite real values")
    return tuple(float(v) for v in result)


def count(value: int, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def names(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} requires a sequence")
    result = tuple(values)
    for value in result:
        _identifier(value, name)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def poses(values: Mapping[str, object]) -> Mapping[str, tuple]:
    return MappingProxyType(
        {
            _identifier(k, "entity UID"): _rigid_pose(v)
            for k, v in sorted(values.items())
        }
    )


def atomic_json(path: Path, value: object) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(plain(value), stream, allow_nan=False, sort_keys=True, indent=2)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
