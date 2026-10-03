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

"""Deterministic JSON canonicalization used by the TaskSpec protocol.

TaskSpec identities are content identities.  This module deliberately accepts
only JSON values (plus dataclasses and enums that resolve to JSON values) and
never evaluates a string or calls a user supplied object.  Keeping this code
free of simulator imports lets the generated specification be shared by the
asset, scene, Task Program, and expansion pipelines.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Final

__all__ = [
    "CANONICALIZATION_VERSION",
    "canonical_hash",
    "canonical_json",
    "canonicalize",
    "json_snapshot",
]

CANONICALIZATION_VERSION: Final = "task_spec_c14n/v1"


def canonicalize(
    value: Any, *, drop_keys: AbstractSet[str] | frozenset[str] = frozenset()
) -> Any:
    """Return a detached, deterministically ordered JSON-compatible value.

    Args:
        value: A JSON value, mapping, dataclass, or enum value.
        drop_keys: Mapping keys omitted recursively before hashing.  This is
            used to exclude derived identities such as ``semantic_hash``.

    Returns:
        A new value containing only JSON primitives, lists, and string-keyed
        dictionaries.

    Raises:
        TypeError: If ``value`` contains a callable, set, bytes, path, or
            another value outside the protocol.
        ValueError: If a floating-point value is NaN or infinite.
    """
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Enum):
        value = value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("TaskSpec JSON values must contain finite floats.")
        return value
    if callable(value):
        raise TypeError("TaskSpec values must not contain callables.")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("TaskSpec values must not contain byte strings.")
    if isinstance(value, AbstractSet):
        raise TypeError("TaskSpec values must not contain sets.")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("TaskSpec mapping keys must be strings.")
            if key in drop_keys:
                continue
            result[key] = canonicalize(item, drop_keys=drop_keys)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, Sequence):
        if isinstance(value, (str, bytes, bytearray, memoryview)):
            raise TypeError("TaskSpec scalar values must be JSON strings.")
        return [canonicalize(item, drop_keys=drop_keys) for item in value]
    raise TypeError(f"Unsupported TaskSpec value type: {type(value).__name__}.")


def canonical_json(value: Any) -> str:
    """Serialize one protocol value using the TaskSpec canonical JSON form."""
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(
    value: Any, *, drop_keys: AbstractSet[str] | frozenset[str] = frozenset()
) -> str:
    """Return the lowercase SHA-256 digest of a canonical protocol value."""
    payload = json.dumps(
        canonicalize(value, drop_keys=drop_keys),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_snapshot(value: Any) -> Any:
    """Return a validated detached snapshot suitable for public contracts."""
    return canonicalize(value)
