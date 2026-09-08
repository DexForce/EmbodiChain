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

"""Pour atomic action implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv

from embodichain.lab.sim.atomic_actions.affordance import AxisAlignAffordance
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import AtomicAction
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import ActionPlan, TimedTrajectory
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    arm_qpos_from_state,
    require_shared_task_state_key,
)
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import build_pose_plan_states


@dataclass(frozen=True, slots=True, eq=False)
class PourGoal:
    """Rotate the object currently held by the bound manipulator."""


@dataclass(frozen=True, slots=True, eq=False)
class PourOptions(ActionOptions):
    """Per-invocation pouring behavior."""

    rotate_angle: float = math.pi / 4.0
    """Signed rotation about the held object's local internal axis, in radians."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.rotate_angle):
            raise ValueError("rotate_angle must be finite.")


class Pour(AtomicAction[PourGoal, PourOptions]):
    """Rotate and return an exclusively held object about its internal axis."""

    skill_id: ClassVar[str] = "pour"
    GoalType: ClassVar[type] = PourGoal
    OptionsType: ClassVar[type] = PourOptions
    open_loop: ClassVar[bool] = True
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "primary",
                motion_capabilities=frozenset(
                    {
                        CARTESIAN_POSE_CAPABILITY,
                        FORWARD_KINEMATICS_CAPABILITY,
                    }
                ),
                grasp_commands={GRASP_COMMAND: JointPositionCommand},
            ),
        ),
    )

    def _plan(
        self,
        request: ResolvedActionRequest[PourGoal, PourOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan a held-object rotation followed by the inverse rotation."""
        self.require_goal(request)
        options = request.skill_options
        motion_endpoint = request.binding.endpoint("primary", "motion")
        grasp_endpoint = request.binding.endpoint("primary", "grasp")
        manipulator = motion_endpoint.require_target(JointPositionTarget)
        end_effector = grasp_endpoint.require_target(JointPositionTarget)
        control_part = manipulator.control_part
        arm_joint_ids = list(manipulator.joint_ids)
        hand_joint_ids = list(end_effector.joint_ids)
        task_state_key = require_shared_task_state_key(
            motion_endpoint,
            grasp_endpoint,
            participant="Pour primary participant",
        )

        held_object = context.get_held_object(task_state_key)
        if held_object is None:
            raise ValueError(
                "Pour requires an object held by task-state resource "
                f"{task_state_key!r} - run PickUp first."
            )
        affordance = held_object.semantics.affordance
        if not isinstance(affordance, AxisAlignAffordance):
            raise ValueError(
                "Pour requires the held object to use an AxisAlignAffordance."
            )
        eligible = context.task.exclusive_held_object_mask(task_state_key)
        if not eligible.any():
            return self.failed_plan(
                request,
                context,
                message="Held object is not exclusive to the task-state resource.",
            )

        start_arm_qpos = arm_qpos_from_state(context, arm_joint_ids)
        current_eef_pose = self.robot.compute_fk(
            qpos=start_arm_qpos,
            name=control_part,
            to_matrix=True,
        )
        object_to_eef = held_object.object_to_eef.to(
            device=self.device,
            dtype=torch.float32,
        )
        current_object_pose = torch.bmm(current_eef_pose, pose_inv(object_to_eef))

        internal_axis = affordance.internal_axis.to(
            device=self.device,
            dtype=torch.float32,
        )
        internal_axis = internal_axis / torch.linalg.vector_norm(internal_axis)
        world_axis = torch.matmul(current_object_pose[:, :3, :3], internal_axis)
        rotation_delta = axis_angle_to_rotation_matrix(
            world_axis * options.rotate_angle
        )
        target_object_pose = current_object_pose.clone()
        target_object_pose[:, :3, :3] = torch.bmm(
            rotation_delta,
            current_object_pose[:, :3, :3],
        )
        poured_eef_pose = torch.bmm(target_object_pose, object_to_eef)

        # The object remains rigidly attached to the EEF throughout Pour.  The
        # first target performs the requested rotation about the object's
        # internal axis; the second target returns to the EEF pose observed by
        # FK at start_qpos, which is exactly the inverse rotation by the same
        # signed angle.  Supplying both targets in one call lets planners such
        # as cuRobo chain the two collision-aware legs without another public
        # MotionGenerator.generate invocation.
        eef_targets = torch.stack([poured_eef_pose, current_eef_pose], dim=1)

        result = self.motion_generator.generate(
            build_pose_plan_states(eef_targets),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_arm_qpos,
                control_part=control_part,
                interpolation_dt=context.require_control_dt(),
            ),
        )
        assert isinstance(result.success, torch.Tensor)
        assert result.positions is not None
        assert result.dt is not None
        success = result.success & eligible

        hand_grasp_qpos = grasp_endpoint.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        full = torch.empty(
            (self.num_envs, result.positions.shape[1], self.robot_dof),
            dtype=context.robot.qpos.dtype,
            device=self.device,
        )
        full[:] = context.last_qpos.unsqueeze(1)
        full[:, :, arm_joint_ids] = result.positions
        full[:, :, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_positions(
                full,
                env_ids=context.env_ids,
                dt=result.dt,
            ),
            segment_lengths={"pour": full.shape[1]},
        )


__all__ = ["Pour", "PourGoal", "PourOptions"]
