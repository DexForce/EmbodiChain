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

"""Eligibility checks; a completed run is not automatically comparable."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["comparison_reasons"]


def _field(row: Mapping[str, Any], path: str) -> Any:
    value = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def comparison_reasons(
    rows: Sequence[Mapping[str, Any]], *, invariant_paths: Sequence[str]
) -> tuple[str, ...]:
    """Return missing/mismatched conditions, leaving raw results available.

    Domains declare invariants and supply independent quality qualification.
    This function never promotes task completion or image non-emptiness into
    quality qualification, and never invents a speedup from incompatible data.
    """
    complete = [r for r in rows if r.get("status") == "completed"]
    reasons = []
    if len({r.get("backend") for r in complete}) < 2:
        reasons.append("need_at_least_two_completed_backends")
    if not complete or any(r.get("quality_status") != "qualified" for r in complete):
        reasons.append("quality_status")
    for path in invariant_paths:
        values = [_field(row, path) for row in complete]
        if (
            not values
            or any(v is None for v in values)
            or any(v != values[0] for v in values)
        ):
            reasons.append(path)
    return tuple(reasons)
