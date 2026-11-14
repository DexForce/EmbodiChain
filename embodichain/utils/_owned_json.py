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


"""Owned immutable finite JSON nodes and defensive mutable projections."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from types import MappingProxyType

__all__: list[str] = []

_ATOMS = frozenset((str, bool, int, type(None)))


class _FrozenTuple(tuple):
    __slots__ = ()


class _FrozenMap(Mapping):
    """Own validated immutable children; expose no mutable backing dictionary."""

    __slots__ = ("_data", "_strict", "_encoded")

    def __init__(self, values, strict):
        object.__setattr__(self, "_data", MappingProxyType(values))
        object.__setattr__(self, "_strict", strict)
        object.__setattr__(self, "_encoded", None)

    def __setattr__(self, name, value):
        raise TypeError("Frozen metadata cannot be modified")

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)

    def mutable_copy(self):
        if self._encoded is None:
            encoded = json.dumps(
                self._data,
                default=dict,
                ensure_ascii=False,
                allow_nan=False,
            )
            object.__setattr__(self, "_encoded", encoded)
        # Every call returns new mutable containers. No caller can modify the
        # validated immutable source through the returned tree.
        return json.loads(self._encoded)


def _key_valid(key):
    return type(key) is str and bool(key) and key == key.strip()


def _strict_keys(value):
    if type(value) is _FrozenMap:
        return value._strict
    if type(value) is _FrozenTuple:
        return all(_strict_keys(item) for item in value)
    return True


def freeze_metadata(value: object) -> object:
    kind = type(value)
    if kind in _ATOMS or kind in (_FrozenMap, _FrozenTuple):
        return value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("metadata must contain only finite JSON values")
        return value
    if isinstance(value, (list, tuple)):
        return _FrozenTuple(freeze_metadata(item) for item in value)
    if isinstance(value, Mapping):
        result = {}
        strict = True
        for key, item in value.items():
            # The expansion contract accepts whitespace around nonempty keys;
            # the Gym JSON-copy contract is stricter. Retain both behaviours.
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metadata key must be a nonempty string")
            owned = freeze_metadata(item)
            result[key] = owned
            strict &= _key_valid(key) and _strict_keys(owned)
        return _FrozenMap(result, strict)
    raise ValueError("metadata must contain only finite JSON values")
