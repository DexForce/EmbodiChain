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

"""Scalar comparisons used by protocol-only evidence adapters."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

__all__: list[str] = []


def _within_target(value: Any, target: Any, tolerance: float) -> Any:
    return abs(value - target) <= tolerance


def _closed_temporal_status(mode: str, statuses: set[str], *, complete: bool) -> str:
    if mode == "eventually" and "satisfied" in statuses:
        return "satisfied"
    if mode == "always" and "failed" in statuses:
        return "failed"
    if not complete or not statuses:
        return "unavailable"
    if mode == "always":
        priority = ("unavailable", "unknown", "not_run")
        fallback = "satisfied"
    else:
        priority = ("unknown", "not_run", "unavailable")
        fallback = "failed"
    return next((status for status in priority if status in statuses), fallback)


def _compare(observed: Any, expected: Any, operator: str) -> bool:
    if operator == "eq":
        return observed == expected
    if operator == "ne":
        return observed != expected
    if operator == "gt":
        return observed > expected
    if operator == "gte":
        return observed >= expected
    if operator == "lt":
        return observed < expected
    if operator == "lte":
        return observed <= expected
    if operator == "in":
        if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
            raise TypeError("operator='in' expects a sequence as expected value")
        return observed in expected
    if operator == "contains":
        return expected in observed
    if operator == "approx":
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            raise TypeError("operator='approx' requires a numeric observation")
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            raise TypeError("operator='approx' requires a numeric expected value")
        return math.isclose(
            float(observed), float(expected), rel_tol=1e-6, abs_tol=1e-6
        )
    raise ValueError(f"unsupported predicate operator {operator!r}")
