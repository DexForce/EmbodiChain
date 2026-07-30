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

"""Shared semantic success policies for generation and runtime."""

from __future__ import annotations

from typing import Any

__all__ = ["upright_in_place_success_spec"]


def upright_in_place_success_spec(
    object_uid: str,
    *,
    local_axis: str = "z",
    xy_tolerance: float,
    max_tilt: float,
) -> dict[str, Any]:
    """Build the one canonical upright-in-place success predicate."""
    return {
        "op": "all",
        "terms": [
            {
                "type": "object_xy_near_initial",
                "object": str(object_uid),
                "tolerance": float(xy_tolerance),
            },
            {
                "type": "object_upright",
                "object": str(object_uid),
                "local_axis": str(local_axis),
                "max_tilt": float(max_tilt),
            },
        ],
    }
