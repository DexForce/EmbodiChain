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

"""Strict JSON-compatible value ownership helpers for Gym protocols."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

__all__: list[str] = []


def json_safe_copy(value: Any, *, field_name: str) -> Any:
    """Return an owned JSON value without implicit type coercion."""
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key or key != key.strip():
                raise ValueError(
                    f"{field_name} mapping keys must be non-empty strings "
                    "without outer whitespace."
                )
            result[key] = json_safe_copy(
                item,
                field_name=f"{field_name}.{key}",
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            json_safe_copy(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{field_name} contains non-JSON value {type(value).__name__}.")
