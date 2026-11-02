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
import torch
from pathlib import Path
from types import MappingProxyType

from .._validation import _identifier, _rigid_pose

__all__ = []


def plain(value: object) -> object:
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, torch.Tensor):
        return plain(value.detach().cpu().tolist())
    if isinstance(value, (Path, torch.device)):
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
