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

"""Observed native receiver geometry through the standard simulation factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import uuid4

import torch

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope
from embodichain.lab.sim.scene.native_hand_scene import capture_native_hand_scene

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.atomic_actions.invocation import ResolvedActionRequest
    from embodichain.lab.sim.atomic_actions.state import PlanningContext
    from embodichain.lab.sim.scene.native_hand_scene import NativeHandSceneQuery
    from .bindings import SimulationSceneBinding

__all__ = ["SimulationHandOverSceneProvider"]


class SimulationHandOverSceneProvider:
    """Bind explicit receiver resources to one observed Default scene snapshot.

    Scene owns geometry capture and collision queries; Atomic owns candidate and
    command qualification. The provider never substitutes later joint poses for
    projected or stale input, and the Atomic caller owns closing the query.
    """

    def __init__(
        self,
        robot: Robot,
        simulation: SimulationManager,
        *,
        scene_binding: SimulationSceneBinding,
        required_owner_uids: tuple[str, ...],
        read_source_stamp: Callable[[], PhysicalObservationStamp],
    ) -> None:
        from .bindings import SimulationSceneBinding

        if type(scene_binding) is not SimulationSceneBinding:
            raise TypeError("Native handover requires an owned SimulationSceneBinding.")
        if not callable(read_source_stamp):
            raise TypeError("Native handover requires an observation stamp reader.")
        self.robot = robot
        self._simulation = simulation
        self._targets = {
            item.entity_id: item.simulation_uid for item in scene_binding.rigid_objects
        }
        self._required = tuple(required_owner_uids)
        if (
            not self._required
            or len(set(self._required)) != len(self._required)
            or any(
                type(uid) is not str or not uid or uid != uid.strip()
                for uid in self._required
            )
        ):
            raise ValueError(
                "Native handover requires a unique explicit scene inventory."
            )
        self._read_stamp = read_source_stamp

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> NativeHandSceneQuery:
        """Capture one fresh query only while the complete observation stamp matches."""
        if request.skill_id != "hand_over":
            raise ValueError("Native receiver geometry is owned by HandOver requests.")
        if context.robot.qpos.shape[0] != 1 or context.env_ids.tolist() != [0]:
            raise ValueError("Native receiver geometry requires B=1 and native env 0.")
        observed = context.observed_robot or context.robot
        if context.observed_robot is not None and not torch.equal(
            context.robot.qpos, observed.qpos
        ):
            raise ValueError("Native HandOver does not support projected joint starts.")
        stamp = context.world_source_stamp
        if (
            type(stamp) is not PhysicalObservationStamp
            or stamp.source_kind != "default.contacts"
        ):
            raise ValueError(
                "Native receiver geometry requires an observed Default source stamp."
            )
        if observed.root_pose is None:
            raise ValueError(
                "Native receiver geometry requires its observed robot root."
            )

        def read_revision() -> int:
            if self._read_stamp() != stamp:
                raise ValueError(
                    "Native receiver observation source changed during preparation."
                )
            return stamp.revision

        read_revision()
        if request.goal.semantics is None:
            raise ValueError(
                "Native receiver geometry requires explicit object semantics."
            )
        target = request.goal.semantics.entity_id
        target_uid = self._targets.get(target)
        if target_uid is None or target_uid not in self._required:
            raise ValueError(
                "Native receiver target must have one declared rigid owner in the inventory."
            )
        motion = request.binding.endpoint("destination", "motion").require_target(
            JointPositionTarget
        )
        hand = request.binding.endpoint("destination", "grasp").require_target(
            JointPositionTarget
        )
        if observed.timestamp != context.scene.timestamp:
            raise ValueError(
                "Native receiver robot and scene observations have different times."
            )
        return capture_native_hand_scene(
            self.robot,
            self._simulation,
            scope=NativeQueryScope(
                str(uuid4()), observed.timestamp, context.scene.version
            ),
            expected_qpos=observed.qpos,
            root_pose=observed.root_pose,
            receiver_control_part=motion.control_part,
            receiver_joint_ids=hand.joint_ids,
            required_owner_uids=self._required,
            excluded_owner_uid=target_uid,
            native_revision=stamp.revision,
            read_native_revision=read_revision,
        )
