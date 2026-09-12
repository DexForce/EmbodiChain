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

"""Configured simulation integration for semantic hand-over poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from embodichain.lab.task_program.semantics import (
    SemanticPose,
)
from embodichain.lab.task_program.compiler.lowering import (
    HandOverPoseProvider,
    HandOverPoseTargets,
    SemanticObjectTarget,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions import PlanningContext
    from embodichain.lab.task_program.semantics import BoundSemanticCall, HandOver


def _validated_pose(
    position: tuple[float, float, float],
    quaternion_xyzw: tuple[float, float, float, float],
    *,
    field_name: str,
) -> SemanticPose:
    """Build and validate one unbatched semantic pose declaration."""
    if type(position) is not tuple or len(position) != 3:
        raise TypeError(f"{field_name}_position must be an exact 3-tuple.")
    if type(quaternion_xyzw) is not tuple or len(quaternion_xyzw) != 4:
        raise TypeError(f"{field_name}_quaternion_xyzw must be an exact 4-tuple.")
    try:
        return SemanticPose(
            position=position,
            quaternion_xyzw=quaternion_xyzw,
        )
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"Invalid {field_name} hand-over pose: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ConfiguredHandOverPoseProvider(HandOverPoseProvider):
    """Resolve a hand-over delivery target from immutable configuration.

    The provider carries object-space poses rather than arm trajectories. The
    shared semantic compiler and atomic ``HandOver`` implementation remain
    responsible for grasp selection, IK, motion generation, transfer, release,
    and delivery. Keeping the numeric declaration as tuple fields also makes
    the provider suitable for task-registration catalog fingerprinting.

    Args:
        final_position: World-frame object delivery position.
        final_quaternion_xyzw: World-frame object delivery orientation.
    """

    provider_id: ClassVar[str] = "simulation.configured_handover_pose"

    final_position: tuple[float, float, float]
    final_quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        final = _validated_pose(
            self.final_position,
            self.final_quaternion_xyzw,
            field_name="final",
        )
        object.__setattr__(
            self,
            "final_position",
            tuple(float(value) for value in final.position.tolist()),
        )
        object.__setattr__(
            self,
            "final_quaternion_xyzw",
            tuple(float(value) for value in final.quaternion_xyzw.tolist()),
        )

    def resolve(
        self,
        call: HandOver,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
    ) -> HandOverPoseTargets:
        """Return an independently owned object-space delivery target.

        Args:
            call: Canonical hand-over semantic call.
            context: Latest immutable planning observation.
            bound: Engine/profile-bound hand-over call.

        Returns:
            Configured final object-space target.
        """
        del call, context, bound
        return HandOverPoseTargets(
            final=SemanticObjectTarget(
                pose=SemanticPose(
                    position=self.final_position,
                    quaternion_xyzw=self.final_quaternion_xyzw,
                )
            ),
        )


__all__ = ["ConfiguredHandOverPoseProvider"]
