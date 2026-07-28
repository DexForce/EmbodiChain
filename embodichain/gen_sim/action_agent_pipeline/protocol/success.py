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

"""Success predicate names shared by config generation and evaluation."""

from __future__ import annotations

from typing import Final

__all__ = ["SUCCESS_TERM_ALIASES", "SUCCESS_TERM_TYPES", "SuccessTerm"]


class SuccessTerm:
    """Canonical success predicate names serialized into generated configs."""

    OBJECT_POSITION_NEAR: Final = "object_position_near"
    OBJECT_XY_NEAR: Final = "object_xy_near"
    OBJECT_IN_CONTAINER: Final = "object_in_container"
    OBJECT_ON_OBJECT: Final = "object_on_object"
    OBJECT_NOT_FALLEN: Final = "object_not_fallen"
    OBJECT_AXIS_OFFSET_NEAR: Final = "object_axis_offset_near"
    OBJECT_AXIS_NEAR: Final = "object_axis_near"
    OBJECTS_COLLINEAR: Final = "objects_collinear"
    OBJECTS_ORDERED: Final = "objects_ordered"
    OBJECT_LIFTED: Final = "object_lifted"
    OBJECT_HELD_BY_GRIPPER: Final = "object_held_by_gripper"
    OBJECT_HELD_BY_BOTH_GRIPPERS: Final = "object_held_by_both_grippers"
    BOTH_GRIPPERS_OPEN: Final = "both_grippers_open"
    GRIPPERS_CLEAR_OF_OBJECT: Final = "grippers_clear_of_object"
    BOTH_ARMS_AT_INITIAL_QPOS: Final = "both_arms_at_initial_qpos"


SUCCESS_TERM_TYPES: Final = frozenset(
    value
    for name, value in vars(SuccessTerm).items()
    if name.isupper() and isinstance(value, str)
)

# Aliases are read-only compatibility input. Newly generated configs always
# emit the canonical value on the right-hand side.
SUCCESS_TERM_ALIASES: Final = {
    "object_near_position": SuccessTerm.OBJECT_POSITION_NEAR,
    "object_near_xy": SuccessTerm.OBJECT_XY_NEAR,
    "object_on": SuccessTerm.OBJECT_ON_OBJECT,
    "on_object": SuccessTerm.OBJECT_ON_OBJECT,
    "not_fallen": SuccessTerm.OBJECT_NOT_FALLEN,
    "object_relative_axis_near": SuccessTerm.OBJECT_AXIS_OFFSET_NEAR,
    "object_coordinate_near": SuccessTerm.OBJECT_AXIS_NEAR,
    "object_height_above_initial": SuccessTerm.OBJECT_LIFTED,
    "object_gripper_near": SuccessTerm.OBJECT_HELD_BY_GRIPPER,
}
