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

"""Frozen-reference demonstration prototypes and tabletop pose transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import FloatArray, NormalizedReal2SimUnit, _array
from .normalize import _rigid

__all__ = [
    "SkillMotionPrototype",
    "build_motion_prototype",
    "retarget_motion_prototype",
]

_PHASES = {
    "pick_up": ("approach", "close", "lift"),
    "move_held_object": ("transport",),
    "place": ("approach", "release", "retract"),
}
_REFERENCES = {
    "pick_up": ("object", "object", "gravity"),
    "move_held_object": ("target",),
    "place": ("target", "target", "target"),
}


@dataclass(frozen=True)
class SkillMotionPrototype:
    """A normalized skill unit with frozen object and target references.

    The stored world TCP trajectory is evidence, not an executable plan. A
    robot-specific host still has to perform continuous IK, connection planning,
    complete-plan construction and physical validation.
    """

    unit: NormalizedReal2SimUnit
    object_to_tcp: FloatArray | None

    def __post_init__(self) -> None:
        if not isinstance(self.unit, NormalizedReal2SimUnit):
            raise TypeError("unit must be a NormalizedReal2SimUnit")
        if (self.unit.world_tcp is None) != (self.object_to_tcp is None):
            raise ValueError(
                "TCP poses and the object-relative grasp must be provided together"
            )
        if self.object_to_tcp is not None:
            relation = _array(self.object_to_tcp, "object_to_tcp", 2)
            _rigid(relation, "object_to_tcp")
            object.__setattr__(self, "object_to_tcp", relation)


def build_motion_prototype(unit: NormalizedReal2SimUnit) -> SkillMotionPrototype:
    """Create a prototype from one explicitly phased normalized skill unit.

    Pick derives the grasp relation at the first lift sample; transport and
    place derive it at their initial held-object sample. The caller must supply
    the object's world pose at that exact event; it is never inferred from the
    TCP trajectory or an object category.

    Args:
        unit: Normalized source unit with explicit object and TCP poses.

    Returns:
        An immutable prototype whose interaction reference is frozen.

    Raises:
        ValueError: Required phases, anchors or TCP poses are absent.
    """
    if not isinstance(unit, NormalizedReal2SimUnit):
        raise TypeError("unit must be a NormalizedReal2SimUnit")
    phases = tuple(phase.name for phase in unit.phases)
    if phases != _PHASES[unit.source.skill_id]:
        raise ValueError(
            f"{unit.source.skill_id} requires phases {_PHASES[unit.source.skill_id]}"
        )
    references = tuple(phase.reference for phase in unit.phases)
    if references != _REFERENCES[unit.source.skill_id]:
        raise ValueError(
            f"{unit.source.skill_id} requires frozen references {_REFERENCES[unit.source.skill_id]}"
        )
    if unit.source.skill_id == "pick_up" and unit.phases[1].kind not in {
        "contact",
        "hold",
    }:
        raise ValueError("Pick close must be a contact or hold phase")
    if unit.source.skill_id == "pick_up" and unit.phases[2].kind not in {
        "contact",
        "hold",
    }:
        raise ValueError("Pick lift must remain a contact or hold phase")
    if unit.source.skill_id == "place" and unit.phases[1].kind not in {
        "contact",
        "hold",
    }:
        raise ValueError("Place release must be a contact or hold phase")
    if unit.source.object_pose is None:
        raise ValueError("prototype needs an explicit object pose")
    if (
        unit.source.skill_id in {"move_held_object", "place"}
        and unit.source.target_pose is None
    ):
        raise ValueError("transport and place require a target reference")
    if unit.world_tcp is None:
        return SkillMotionPrototype(unit=unit, object_to_tcp=None)
    index = unit.phases[2].start if unit.source.skill_id == "pick_up" else 0
    relation = (
        np.linalg.inv(_rigid(unit.source.object_pose, "object_pose"))
        @ unit.world_tcp[index]
    )
    return SkillMotionPrototype(unit=unit, object_to_tcp=relation)


def _tabletop_delta(
    current: np.ndarray, reference: np.ndarray, name: str
) -> np.ndarray:
    transform = _rigid(current, name) @ np.linalg.inv(
        _rigid(reference, f"source {name}")
    )
    if not np.allclose(transform[:3, 2], [0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError(
            f"{name} changes roll or pitch; tabletop transfer supports yaw only"
        )
    if abs(transform[2, 3]) > 1e-6:
        raise ValueError(f"{name} changes height; tabletop transfer supports XY only")
    return transform


def retarget_motion_prototype(
    prototype: SkillMotionPrototype,
    *,
    object_pose: FloatArray | None = None,
    target_pose: FloatArray | None = None,
    held_object_to_tcp: FloatArray | None = None,
) -> FloatArray:
    """Transfer prototype TCP poses to tabletop object and target references.

    Pick uses the *frozen* object pose for approach and close. Lift follows a
    gravity-aligned rigid transform, preserving world-up displacement instead
    of taking the relative pose to the moving object. Transport and place use
    the target reference; a changed held relation is applied in object space.

    Args:
        prototype: Prepared prototype containing normalized world TCP poses.
        object_pose: Current world object pose for Pick.
        target_pose: Current world target pose for transport or Place.
        held_object_to_tcp: Current verified held relation for transport/place.

    Returns:
        Owned world-frame TCP poses with shape ``(N, 4, 4)``.

    Raises:
        ValueError: Required references are absent or outside tabletop scope.
    """
    if not isinstance(prototype, SkillMotionPrototype):
        raise TypeError("prototype must be a SkillMotionPrototype")
    unit = prototype.unit
    poses = unit.world_tcp
    if poses is None or prototype.object_to_tcp is None:
        raise ValueError("Cartesian transfer requires normalized TCP poses")
    if unit.source.skill_id == "pick_up":
        if object_pose is None:
            raise ValueError("Pick requires a current object pose")
        assert unit.source.object_pose is not None
        delta = _tabletop_delta(object_pose, unit.source.object_pose, "object_pose")
        # A tabletop yaw preserves gravity, so applying the same rigid change
        # to the lift preserves world-up motion as well as the grasp relation.
        result = delta @ poses
    else:
        if target_pose is None or held_object_to_tcp is None:
            raise ValueError("transport and place require target and held relation")
        assert unit.source.target_pose is not None
        delta = _tabletop_delta(target_pose, unit.source.target_pose, "target_pose")
        held = _rigid(held_object_to_tcp, "held_object_to_tcp")
        # Recover source object motion from the frozen grasp and reapply the
        # currently verified relation. This is a geometric transfer only.
        result = delta @ poses @ np.linalg.inv(prototype.object_to_tcp) @ held
    result = np.array(result, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result
