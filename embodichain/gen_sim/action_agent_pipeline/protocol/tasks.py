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

"""Task-routing and spatial-relation values serialized by the pipeline."""

from __future__ import annotations

from typing import Final

__all__ = [
    "MANIPULATION_INTENTS",
    "RELATIVE_RELATIONS",
    "SIDE_RELATIONS",
    "TASK_ROUTE_ARRANGEMENT_LINE",
    "TASK_ROUTE_OBJECT_MANIPULATION",
    "TASK_ROUTE_STACKING",
    "TASK_ROUTE_UNSUPPORTED",
    "TASK_ROUTES",
]

TASK_ROUTE_STACKING: Final = "stacking"
TASK_ROUTE_ARRANGEMENT_LINE: Final = "arrangement_line"
TASK_ROUTE_OBJECT_MANIPULATION: Final = "object_manipulation"
TASK_ROUTE_UNSUPPORTED: Final = "unsupported"
TASK_ROUTES: Final = frozenset(
    {
        TASK_ROUTE_STACKING,
        TASK_ROUTE_ARRANGEMENT_LINE,
        TASK_ROUTE_OBJECT_MANIPULATION,
        TASK_ROUTE_UNSUPPORTED,
    }
)

RELATIVE_RELATIONS: Final = frozenset(
    {
        "inside",
        "on",
        "left_of",
        "right_of",
        "front_of",
        "behind",
        "front_left_of",
        "back_left_of",
        "front_right_of",
        "back_right_of",
    }
)
SIDE_RELATIONS: Final = RELATIVE_RELATIONS - {"inside", "on"}
MANIPULATION_INTENTS: Final = frozenset(
    {"place_relative", "hold_hover", "coordinated_pickment"}
)
