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

"""MoveHeldObject atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    arm_qpos_from_state,
    require_shared_task_state_key,
    resolve_object_target,
)
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import AtomicAction
from embodichain.lab.sim.atomic_actions.goals import (
    PoseGoalValue,
    resolve_pose_goal,
    validate_pose_goal,
)
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import ActionPlan, TimedTrajectory
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import build_pose_plan_states
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)


@dataclass(frozen=True, slots=True, eq=False)
class HeldObjectPoseGoal:
    """Desired pose for the object held by this action's control part."""

    object_target_pose: PoseGoalValue
    """Target object pose, shape ``(4, 4)`` or ``(num_envs, 4, 4)``."""

    def __post_init__(self) -> None:
        validate_pose_goal(
            self.object_target_pose,
            "object_target_pose",
            allow_waypoints=False,
        )


@dataclass(frozen=True, slots=True, eq=False)
class MoveHeldObjectOptions(ActionOptions):
    """Per-invocation held-object transport behavior."""


class MoveHeldObject(AtomicAction[HeldObjectPoseGoal, MoveHeldObjectOptions]):
    """Move the held object to the exact target object pose with a closed hand.

    The requested object orientation is preserved exactly. Callers that need a
    transport orientation must encode it in :class:`HeldObjectPoseGoal`; this
    action never substitutes an implicit end-effector orientation.
    """

    skill_id: ClassVar[str] = "move_held_object"
    GoalType: ClassVar[type] = HeldObjectPoseGoal
    OptionsType: ClassVar[type] = MoveHeldObjectOptions
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
        request: ResolvedActionRequest[HeldObjectPoseGoal, MoveHeldObjectOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan exact object-space transport without changing the attachment."""
        target = request.goal
        options = request.skill_options
        binding = request.binding
        motion = binding.endpoint("primary", "motion")
        grasp = binding.endpoint("primary", "grasp")
        motion_target = motion.require_target(JointPositionTarget)
        grasp_target = grasp.require_target(JointPositionTarget)
        task_state_key = require_shared_task_state_key(
            motion,
            grasp,
            participant="MoveHeldObject primary participant",
        )
        control_part = motion_target.control_part
        arm_joint_ids = list(motion_target.joint_ids)
        hand_joint_ids = list(grasp_target.joint_ids)
        hand_grasp_qpos = grasp.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        state = context
        held_object = state.get_held_object(task_state_key)
        if held_object is None:
            raise ValueError(
                "MoveHeldObject requires an object held by task-state resource "
                f"{task_state_key!r} - run PickUp first."
            )
        eligible = context.task.exclusive_held_object_mask(task_state_key)
        if not eligible.any():
            return self.failed_plan(
                request,
                context,
                message="Held object is not exclusive to the control part.",
            )
        object_target_pose = resolve_object_target(
            resolve_pose_goal(
                target.object_target_pose,
                context,
                name="object_target_pose",
            ),
            num_envs=self.num_envs,
            device=self.device,
        )
        start_arm_qpos = arm_qpos_from_state(state, arm_joint_ids)
        object_to_eef = held_object.object_to_eef.to(
            device=self.device, dtype=torch.float32
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0).repeat(self.num_envs, 1, 1)
        move_eef_xpos = torch.bmm(object_target_pose, object_to_eef)

        result = self.motion_generator.generate(
            build_pose_plan_states(move_eef_xpos),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_arm_qpos,
                control_part=control_part,
                interpolation_dt=context.control_dt,
            ),
        )
        assert isinstance(result.success, torch.Tensor)
        assert result.positions is not None
        success = result.success & eligible
        arm_traj = result.positions

        full = torch.empty(
            (self.num_envs, arm_traj.shape[1], self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :, arm_joint_ids] = arm_traj
        full[:, :, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
        assert result.dt is not None

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_positions(
                full,
                env_ids=context.env_ids,
                dt=result.dt,
            ),
            segment_lengths={"transport": full.shape[1]},
        )


__all__ = [
    "HeldObjectPoseGoal",
    "MoveHeldObject",
    "MoveHeldObjectOptions",
]
