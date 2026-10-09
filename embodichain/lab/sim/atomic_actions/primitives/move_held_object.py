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

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, ClassVar

import torch

from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    arm_qpos_from_state,
    require_shared_task_state_key,
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
from embodichain.lab.sim.atomic_actions.plans import ActionPlan, PlannerDiagnostics
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    resolve_pose_target,
    to_full_robot_trajectory,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)


@dataclass(frozen=True, slots=True, eq=False)
class HeldObjectPoseGoal:
    """Desired pose for the object held by this action's control part."""

    object_target_pose: PoseGoalValue
    """Object target pose or ordered waypoints.

    Accepts ``(4, 4)``, ``(num_envs, 4, 4)`` or
    ``(num_envs, n_waypoint, 4, 4)``; every waypoint retains the held grasp.
    """

    world_yaw_free: bool = False
    """Permit world-Z yaw changes at the final waypoint, retaining its position.

    Intermediate waypoints remain exact. Planning tries the nominal heading
    first, then seven 45-degree heading alternatives; each complete path must
    pass the motion generator's normal checks. This is not an exhaustive search.
    """

    world_yaw_free_path: bool = False
    """Apply one selected free yaw to every waypoint after the initial waypoint.

    Staged transports use this to select the yaw at the clearance waypoint and
    reuse it for the final descent.
    """

    def __post_init__(self) -> None:
        if type(self.world_yaw_free) is not bool:
            raise TypeError("world_yaw_free must be a boolean.")
        if type(self.world_yaw_free_path) is not bool:
            raise TypeError("world_yaw_free_path must be a boolean.")
        if self.world_yaw_free_path and not self.world_yaw_free:
            raise ValueError(
                "world_yaw_free_path requires world_yaw_free to be enabled."
            )
        validate_pose_goal(
            self.object_target_pose,
            "object_target_pose",
            allow_waypoints=True,
        )


@dataclass(frozen=True, slots=True, eq=False)
class MoveHeldObjectOptions(ActionOptions):
    """Per-invocation held-object transport behavior."""


class MoveHeldObject(AtomicAction[HeldObjectPoseGoal, MoveHeldObjectOptions]):
    """Move the held object through exact target poses with a closed hand.

    Requested orientations are exact unless final world yaw is explicitly free.
    Callers that need a
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
        object_target_pose = resolve_pose_target(
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
        if object_target_pose.ndim == 4:
            object_to_eef = object_to_eef.unsqueeze(1)
        motion_options = request.motion_policy.to_motion_gen_options(
            start_qpos=start_arm_qpos,
            control_part=control_part,
            interpolation_dt=context.control_dt,
        )

        if (
            target.world_yaw_free_path
            and target.world_yaw_free
            and object_target_pose.ndim == 4
            and object_target_pose.shape[1] == 3
        ):
            result, selected_yaws, attempts = self._plan_staged_yaw_free_path(
                object_target_pose,
                object_to_eef,
                motion_options,
                eligible,
            )
        else:
            result, selected_yaws, attempts = self._plan_single_yaw_free_path(
                object_target_pose,
                object_to_eef,
                motion_options,
                eligible,
                target.world_yaw_free,
            )
        if result.positions is None:
            return self.failed_plan(
                request, context, message="No feasible held-object transport path."
            )
        success = result.success & eligible

        base_qpos = state.last_qpos.clone()
        base_qpos[:, hand_joint_ids] = hand_grasp_qpos
        _, timed = to_full_robot_trajectory(
            result,
            base_qpos=base_qpos,
            joint_ids=arm_joint_ids,
            env_ids=context.env_ids,
            control_dt=context.require_control_dt(),
        )

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=timed,
            diagnostics=(
                PlannerDiagnostics(
                    backend=self.planning_services.planner_name,
                    metadata={
                        "world_yaw_offsets_rad": selected_yaws.tolist(),
                        "yaw_plan_attempts": attempts,
                    },
                )
                if target.world_yaw_free
                else None
            ),
            segment_lengths={"transport": timed.waypoint_count},
        )

    def _plan_single_yaw_free_path(
        self,
        object_target_pose: torch.Tensor,
        object_to_eef: torch.Tensor,
        motion_options: Any,
        eligible: torch.Tensor,
        world_yaw_free: bool,
    ) -> tuple[PlanResult, torch.Tensor, int]:
        """Plan one held-object path with optional final-yaw search."""

        def plan_target(poses: torch.Tensor) -> PlanResult:
            return self.motion_generator.generate(
                build_pose_plan_states(torch.matmul(poses, object_to_eef)),
                options=motion_options,
            )

        result = plan_target(object_target_pose)
        assert isinstance(result.success, torch.Tensor)
        selected_yaws = motion_options.start_qpos.new_zeros(
            motion_options.start_qpos.shape[0]
        )
        attempts = 1
        if world_yaw_free:
            for fraction in (0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0):
                missing = eligible & ~result.success
                if not missing.any():
                    break
                angle = math.pi * fraction
                cosine, sine = math.cos(angle), math.sin(angle)
                yaw = object_target_pose.new_tensor(
                    [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
                )
                poses = object_target_pose.clone()
                final_pose = poses[:, -1] if poses.ndim == 4 else poses
                final_pose[:, :3, :3] = yaw @ final_pose[:, :3, :3]
                candidate = plan_target(poses)
                attempts += 1
                accepted = missing & candidate.success
                if accepted.any():
                    result = _merge_yaw_paths(result, candidate, accepted)
                    selected_yaws[accepted] = angle
        return result, selected_yaws, attempts

    def _plan_staged_yaw_free_path(
        self,
        object_target_poses: torch.Tensor,
        object_to_eef: torch.Tensor,
        motion_options: Any,
        eligible: torch.Tensor,
    ) -> tuple[PlanResult, torch.Tensor, int]:
        """Reuse one yaw choice across clearance rotation and final descent."""
        first_poses = object_target_poses[:, :2]

        def plan_first(poses: torch.Tensor, options: Any) -> PlanResult:
            return self.motion_generator.generate(
                build_pose_plan_states(torch.matmul(poses, object_to_eef)),
                options=options,
            )

        first = plan_first(first_poses, motion_options)
        assert isinstance(first.success, torch.Tensor)
        start_qpos = motion_options.start_qpos
        assert start_qpos is not None
        selected_yaws = start_qpos.new_zeros(start_qpos.shape[0])
        attempts = 1
        for fraction in (0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0):
            missing = eligible & ~first.success
            if not missing.any():
                break
            angle = math.pi * fraction
            cosine, sine = math.cos(angle), math.sin(angle)
            yaw = first_poses.new_tensor(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
            )
            candidate_poses = first_poses.clone()
            candidate_poses[:, 1, :3, :3] = yaw @ candidate_poses[:, 1, :3, :3]
            candidate = plan_first(candidate_poses, motion_options)
            attempts += 1
            accepted = missing & candidate.success
            if accepted.any():
                first = _merge_yaw_paths(first, candidate, accepted)
                selected_yaws[accepted] = angle

        if first.positions is None:
            return first, selected_yaws, attempts

        final_poses = object_target_poses[:, 2].clone()
        cosine = torch.cos(selected_yaws)
        sine = torch.sin(selected_yaws)
        yaw = torch.zeros(
            (selected_yaws.shape[0], 3, 3),
            dtype=final_poses.dtype,
            device=final_poses.device,
        )
        yaw[:, 0, 0] = cosine
        yaw[:, 0, 1] = -sine
        yaw[:, 1, 0] = sine
        yaw[:, 1, 1] = cosine
        yaw[:, 2, 2] = 1.0
        final_poses[:, :3, :3] = yaw @ final_poses[:, :3, :3]

        second_options = deepcopy(motion_options)
        second_options.start_qpos = first.positions[:, -1].clone()
        second_object_to_eef = object_to_eef[:, 0]
        second = self.motion_generator.generate(
            build_pose_plan_states(torch.matmul(final_poses, second_object_to_eef)),
            options=second_options,
        )
        assert isinstance(second.success, torch.Tensor)
        missing_second = eligible & first.success & ~second.success
        if (
            missing_second.any()
            and second.positions is not None
            and second.dt is not None
        ):
            # A new heading needs a fresh clearance path as well as its descent.
            for fraction in (0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0):
                if not missing_second.any():
                    break
                angle = math.pi * fraction
                rows = missing_second & ~torch.isclose(
                    selected_yaws, selected_yaws.new_full(selected_yaws.shape, angle)
                )
                if not rows.any():
                    continue
                cosine, sine = math.cos(angle), math.sin(angle)
                candidate_yaw = first_poses.new_tensor(
                    [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
                )
                candidate_poses = first_poses.clone()
                candidate_poses[:, 1, :3, :3] = (
                    candidate_yaw @ candidate_poses[:, 1, :3, :3]
                )
                candidate_first = plan_first(candidate_poses, motion_options)
                attempts += 1
                if (
                    candidate_first.positions is None
                    or candidate_first.dt is None
                    or not (rows & candidate_first.success).any()
                ):
                    continue
                candidate_final = object_target_poses[:, 2].clone()
                candidate_final[:, :3, :3] = candidate_yaw @ candidate_final[:, :3, :3]
                candidate_options = deepcopy(motion_options)
                candidate_options.start_qpos = candidate_first.positions[:, -1].clone()
                candidate_second = self.motion_generator.generate(
                    build_pose_plan_states(candidate_final @ second_object_to_eef),
                    options=candidate_options,
                )
                accepted = rows & candidate_first.success & candidate_second.success
                if (
                    accepted.any()
                    and candidate_second.positions is not None
                    and candidate_second.dt is not None
                ):
                    first = _merge_yaw_paths(first, candidate_first, accepted)
                    second = _merge_yaw_paths(second, candidate_second, accepted)
                    selected_yaws[accepted] = angle
                    missing_second &= ~accepted
        success = first.success & second.success
        if second.positions is None or second.dt is None:
            return (
                PlanResult(success=success, positions=first.positions, dt=first.dt),
                selected_yaws,
                attempts,
            )
        assert first.dt is not None
        return (
            PlanResult(
                success=success,
                positions=torch.cat((first.positions, second.positions), dim=1),
                dt=torch.cat((first.dt, second.dt), dim=1),
            ),
            selected_yaws,
            attempts,
        )


def _merge_yaw_paths(
    previous: PlanResult, candidate: PlanResult, rows: torch.Tensor
) -> PlanResult:
    """Retain accepted rows while selecting newly feasible, independently timed paths."""
    assert candidate.positions is not None and candidate.dt is not None
    if previous.positions is None:
        return candidate
    assert previous.dt is not None
    count = max(previous.positions.shape[1], candidate.positions.shape[1])

    def pad_positions(value: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [value, value[:, -1:].expand(-1, count - value.shape[1], -1)], dim=1
        )

    # Zero-time terminal padding preserves each row's duration. The action
    # recomputes derivatives on the control grid in to_full_robot_trajectory.
    return PlanResult(
        success=torch.where(rows, candidate.success, previous.success),
        positions=torch.where(
            rows[:, None, None],
            pad_positions(candidate.positions),
            pad_positions(previous.positions),
        ),
        dt=torch.where(
            rows[:, None],
            torch.nn.functional.pad(candidate.dt, (0, count - candidate.dt.shape[1])),
            torch.nn.functional.pad(previous.dt, (0, count - previous.dt.shape[1])),
        ),
    )


__all__ = [
    "HeldObjectPoseGoal",
    "MoveHeldObject",
    "MoveHeldObjectOptions",
]
