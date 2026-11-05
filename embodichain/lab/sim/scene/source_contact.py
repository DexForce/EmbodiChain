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

"""Explicit source contact-offset margins without native simulation construction."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
import math
from typing import Any

__all__ = ["source_grasp_contact_clearances"]


def source_grasp_contact_clearances(
    rigid_objects: Sequence[Mapping[str, Any]],
    robot_declaration: Mapping[str, Any],
) -> dict[str, float]:
    """Reserve the source robot and object contact envelopes for mesh fitting.

    Args:
        rigid_objects: Ordered physical source objects and explicit attributes.
        robot_declaration: Physical robot fields including optional link overrides.

    Returns:
        Required jaw separation per object, using the actual Default collision
        descriptor default. No simulation, Robot, collision query or sensor is made.
    """
    from dexsim.spawn import DexsimCollisionDesc

    default_offset = DexsimCollisionDesc().contact_offset

    def offset(attrs: Mapping[str, Any] | None, fallback: float) -> float:
        configured = ((attrs or {}).get("collision_props") or {}).get("contact_offset")
        value = fallback if configured is None else configured
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Grasp-fit requires a finite non-negative contact_offset.")
        return float(value)

    robot_offset = offset(robot_declaration.get("attrs"), default_offset)
    robot_offset = max(
        [robot_offset]
        + [
            offset(group.get("attrs"), robot_offset)
            for group in (robot_declaration.get("link_attrs") or {}).values()
        ]
    )
    return {
        str(obj["uid"]): robot_offset + offset(obj.get("attrs"), default_offset)
        for obj in rigid_objects
    }
