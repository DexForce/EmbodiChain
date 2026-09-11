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

"""Bounded JSON validation shared by pure protocol decoders."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


def json_copy(value: object) -> Any:
    """Copy JSON data, rejecting cycles, non-finite numbers and oversized trees."""
    budget = [16384]

    def visit(item: object, depth: int) -> Any:
        budget[0] -= 1
        if depth > 64 or budget[0] < 0:
            raise ValueError("JSON structure exceeds TaskSpec depth/node limit")
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float and math.isfinite(item):
            return item
        if type(item) is list:
            return [visit(child, depth + 1) for child in item]
        if type(item) is dict and all(type(key) is str for key in item):
            return {key: visit(child, depth + 1) for key, child in item.items()}
        raise ValueError("TaskSpec requires finite, string-keyed JSON data")

    return visit(value, 0)


def fields(
    value: object, required: set[str], optional: set[str], path: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise ValueError(
            f"{path} invalid fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{path} must be a nonempty, trimmed string")
    return value


def digest(value: object, path: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


def items(value: object, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{path} must be {'a nonempty' if nonempty else 'a'} list")
    return value


def dumps(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def hash_json(value: object) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()
