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

"""Twist atomic action implementation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.utils.math import (
    axis_angle_to_rotation_matrix,
    pose_inv,
)

from embodichain.lab.sim.atomic_actions.affordance import TwistAffordance
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
    OPEN_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import AtomicAction, ObjectSemantics
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.goals import (
    ObjectActionGoal,
    PoseGoalValue,
    resolve_pose_goal,
    validate_pose_goal,
)
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlannerDiagnostics,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    find_symmetric_nearest_xpos,
    arm_qpos_from_state,
    hold_inactive_cartesian_rows,
    resample_planned_trajectory,
)
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    axis_translation_keyframes,
    build_pose_plan_states,
    interpolate_hand_qpos,
    resolve_pose_target,
    translate_pose_world,
)


@dataclass(frozen=True, slots=True, eq=False)
class TwistGoal(ObjectActionGoal):
    """Target object described by a twist affordance."""

    target_pose: PoseGoalValue
    """Target pose snapshot or late-bound stable scene-entity reference."""

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        validate_pose_goal(self.target_pose, "target_pose", allow_waypoints=False)


@dataclass(frozen=True, slots=True, eq=False)
class TwistOptions(ActionOptions):
    """Per-invocation twisting behavior."""

    hand_interp_steps: int = 5
    """Number of waypoints used for each close/open hand segment."""

    twist_waypoint_count: int = 8
    """Number of Cartesian keyframes along the target's circular twist arc."""

    pre_grasp_distance: float = 0.1
    """Distance from the grasp pose along its negative z-axis."""

    twist_angle: float = math.pi / 4
    """Requested twist rotation in radians."""

    def __post_init__(self) -> None:
        if self.hand_interp_steps < 1:
            raise ValueError("hand_interp_steps must be at least 1.")
        if self.twist_waypoint_count < 1:
            raise ValueError("twist_waypoint_count must be at least 1.")
        if not math.isfinite(self.pre_grasp_distance):
            raise ValueError("pre_grasp_distance must be finite.")
        if self.pre_grasp_distance < 0.0:
            raise ValueError("pre_grasp_distance must be non-negative.")
        if not math.isfinite(self.twist_angle):
            raise ValueError("twist_angle must be finite.")


class Twist(AtomicAction[TwistGoal, TwistOptions]):
    """Open-loop approach, grasp, twist, release, and retract motion."""

    skill_id: ClassVar[str] = "twist"
    GoalType: ClassVar[type] = TwistGoal
    OptionsType: ClassVar[type] = TwistOptions
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
                grasp_commands={
                    OPEN_COMMAND: JointPositionCommand,
                    GRASP_COMMAND: JointPositionCommand,
                },
            ),
        ),
    )

    def _on_bind(self) -> None:
        """Resolve dimensions owned by the engine's robot."""
        self.num_envs = self.robot.get_qpos().shape[0]
        self.robot_dof = self.robot.dof

    def _plan(
        self,
        request: ResolvedActionRequest[TwistGoal, TwistOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan all six twisting segments without stepping simulation."""
        if request.motion_policy.strategy != "ik_interp":
            raise ValueError(
                "Twist requires strategy='ik_interp' because the constrained "
                "twist path must preserve every Cartesian sample."
            )
        target = self.require_goal(request)
        affordance = self._require_twist_affordance(target.semantics)
        options = request.skill_options
        interpolation_dt = context.require_control_dt()
        binding = request.binding
        motion_target = binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        grasp = binding.endpoint("primary", "grasp")
        grasp_target = grasp.require_target(JointPositionTarget)
        control_part = motion_target.control_part
        arm_joint_ids = list(motion_target.joint_ids)
        hand_joint_ids = list(grasp_target.joint_ids)
        start_arm_qpos = arm_qpos_from_state(context, arm_joint_ids)
        hand_open_qpos = grasp.joint_positions(
            OPEN_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        hand_grasp_qpos = grasp.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )

        link_pose = resolve_pose_target(
            resolve_pose_goal(target.target_pose, context, name="target_pose"),
            num_envs=self.num_envs,
            device=self.device,
        )
        grasp_sample = affordance.sample_grasp_pose(
            link_pose,
            sampling=context.affordance_sampling,
            env_ids=context.env_ids,
            key=(request.invocation_id or self.skill_id) + ":grasp",
        )
        grasp_xpos = grasp_sample.poses.to(device=self.device, dtype=torch.float32)
        grasp_xpos = find_symmetric_nearest_xpos(
            grasp_xpos,
            reference_xpos=self.robot.compute_fk(
                qpos=start_arm_qpos, name=control_part, to_matrix=True
            ),
        )
        pre_grasp_xpos = translate_pose_world(
            grasp_xpos,
            -grasp_xpos[:, :3, 2] * options.pre_grasp_distance,
        )
        n_approach, n_reach, n_twist, n_retract = self._motion_segment_lengths(
            request.motion_policy.sample_count,
            options.hand_interp_steps,
            options.twist_waypoint_count,
        )
        twist_xpos = self._twisted_grasp_poses(
            link_pose,
            grasp_xpos,
            affordance.twist_axis,
            affordance.require_axis_origin(),
            options.twist_angle,
            n_twist - 1,
        )

        approach_success, approach_arm = self._plan_pose_segment(
            pre_grasp_xpos,
            start_arm_qpos,
            control_part,
            request,
            n_approach,
            interpolation_dt=interpolation_dt,
        )
        reach_keyframes = axis_translation_keyframes(
            pre_grasp_xpos,
            grasp_xpos,
            grasp_xpos[:, :3, 2],
            n_waypoints=n_reach - 1,
        )
        reach_active = grasp_sample.success & approach_success
        reach_hold_pose = self.robot.compute_fk(
            qpos=approach_arm[:, -1],
            name=control_part,
            to_matrix=True,
        ).to(reach_keyframes)
        reach_keyframes = hold_inactive_cartesian_rows(
            reach_keyframes,
            active_mask=reach_active,
            hold_pose=reach_hold_pose,
        )
        reach_success, reach_arm = self._plan_pose_segment(
            reach_keyframes,
            approach_arm[:, -1],
            control_part,
            request,
            n_reach,
            interpolation_dt=interpolation_dt,
            cartesian_samples=True,
        )
        twist_active = reach_active & reach_success
        twist_hold_pose = self.robot.compute_fk(
            qpos=reach_arm[:, -1],
            name=control_part,
            to_matrix=True,
        ).to(twist_xpos)
        twist_keyframes = hold_inactive_cartesian_rows(
            twist_xpos,
            active_mask=twist_active,
            hold_pose=twist_hold_pose,
        )

        def sample_twist_path(progress: torch.Tensor) -> torch.Tensor:
            sampled = self._twisted_grasp_poses_at_fractions(
                link_pose,
                grasp_xpos,
                affordance.twist_axis,
                affordance.require_axis_origin(),
                options.twist_angle,
                progress,
            )
            return hold_inactive_cartesian_rows(
                sampled,
                active_mask=twist_active,
                hold_pose=twist_hold_pose,
            )

        twist_success, twist_arm = self._plan_pose_segment(
            twist_keyframes,
            reach_arm[:, -1],
            control_part,
            request,
            n_twist,
            interpolation_dt=interpolation_dt,
            cartesian_samples=True,
            cartesian_path_sampler=sample_twist_path,
        )
        retract_success, retract_arm = self._plan_pose_segment(
            pre_grasp_xpos,
            twist_arm[:, -1],
            control_part,
            request,
            n_retract,
            interpolation_dt=interpolation_dt,
        )
        success = (
            grasp_sample.success
            & approach_success
            & reach_success
            & twist_success
            & retract_success
        )

        hand_close = interpolate_hand_qpos(
            hand_open_qpos,
            hand_grasp_qpos,
            n_waypoints=options.hand_interp_steps,
        )
        hand_open = interpolate_hand_qpos(
            hand_grasp_qpos,
            hand_open_qpos,
            n_waypoints=options.hand_interp_steps,
        )
        parts = (
            approach_arm,
            reach_arm,
            hand_close,
            twist_arm,
            hand_open,
            retract_arm,
        )
        lengths = tuple(part.shape[1] for part in parts)
        full = torch.empty(
            (self.num_envs, sum(lengths), self.robot_dof),
            dtype=context.robot.qpos.dtype,
            device=self.device,
        )
        full[:] = context.last_qpos.unsqueeze(1)
        offset = 0
        arm_parts = (approach_arm, reach_arm, twist_arm, retract_arm)
        arm_hands = (
            hand_open_qpos,
            hand_open_qpos,
            hand_grasp_qpos,
            hand_open_qpos,
        )
        for index, (arm, hand) in enumerate(zip(arm_parts[:2], arm_hands[:2])):
            stop = offset + arm.shape[1]
            full[:, offset:stop, arm_joint_ids] = arm
            full[:, offset:stop, hand_joint_ids] = (
                interpolate_hand_qpos(
                    context.last_qpos[:, hand_joint_ids],
                    hand,
                    n_waypoints=arm.shape[1],
                )
                if index == 0
                else hand.unsqueeze(1)
            )
            offset = stop
        stop = offset + hand_close.shape[1]
        full[:, offset:stop, arm_joint_ids] = reach_arm[:, -1].unsqueeze(1)
        full[:, offset:stop, hand_joint_ids] = hand_close
        offset = stop
        stop = offset + twist_arm.shape[1]
        full[:, offset:stop, arm_joint_ids] = twist_arm
        full[:, offset:stop, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
        offset = stop
        stop = offset + hand_open.shape[1]
        full[:, offset:stop, arm_joint_ids] = twist_arm[:, -1].unsqueeze(1)
        full[:, offset:stop, hand_joint_ids] = hand_open
        offset = stop
        full[:, offset:, arm_joint_ids] = retract_arm
        full[:, offset:, hand_joint_ids] = hand_open_qpos.unsqueeze(1)

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_uniform_step(
                full,
                env_ids=context.env_ids,
                step_dt=interpolation_dt,
            ),
            expected_effects=StateDelta(),
            diagnostics=PlannerDiagnostics(
                backend=self.planning_services.planner_name,
                metadata={"affordance_sample": grasp_sample.metadata},
            ),
            segment_lengths={
                "approach": lengths[0],
                "reach": lengths[1],
                "close": lengths[2],
                "twist": lengths[3],
                "open": lengths[4],
                "retract": lengths[5],
            },
        )

    @staticmethod
    def _require_twist_affordance(
        semantics: ObjectSemantics,
    ) -> TwistAffordance:
        affordance = semantics.affordance
        if not isinstance(affordance, TwistAffordance):
            raise ValueError("Twist requires a TwistAffordance.")
        return affordance

    @staticmethod
    def _motion_segment_lengths(
        sample_count: int,
        hand_interp_steps: int,
        twist_waypoint_count: int,
    ) -> tuple[int, int, int, int]:
        motion_count = sample_count - 2 * hand_interp_steps
        minimum_twist_count = twist_waypoint_count + 1
        if motion_count < minimum_twist_count + 6:
            raise ValueError(
                "Not enough waypoints for Twist. Increase sample_count or "
                "decrease hand_interp_steps or twist_waypoint_count."
            )
        base, remainder = divmod(motion_count, 4)
        values = [base + (index < remainder) for index in range(4)]
        if values[2] >= minimum_twist_count:
            return values[0], values[1], values[2], values[3]
        remaining_count = motion_count - minimum_twist_count
        other_base, other_remainder = divmod(remaining_count, 3)
        other = [other_base + (index < other_remainder) for index in range(3)]
        return other[0], other[1], minimum_twist_count, other[2]

    def _plan_pose_segment(
        self,
        target_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
        request: ResolvedActionRequest[TwistGoal, TwistOptions],
        sample_count: int,
        *,
        interpolation_dt: float,
        cartesian_samples: bool = False,
        cartesian_path_sampler: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.motion_generator.generate(
            build_pose_plan_states(target_pose),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_qpos,
                control_part=control_part,
                sample_count=sample_count,
                interpolation_dt=interpolation_dt,
                cartesian_linear=cartesian_samples,
                cartesian_path_sampler=cartesian_path_sampler,
            ),
        )
        assert isinstance(result.success, torch.Tensor)
        assert result.positions is not None
        assert result.dt is not None
        return result.success, resample_planned_trajectory(
            result.positions,
            sample_count,
            dt=result.dt,
            control_dt=interpolation_dt,
        )

    def _twisted_grasp_poses(
        self,
        link_pose: torch.Tensor,
        grasp_xpos: torch.Tensor,
        twist_axis: torch.Tensor,
        axis_origin: tuple[float, float, float],
        twist_angle: float,
        waypoint_count: int,
    ) -> torch.Tensor:
        """Build Cartesian EEF keyframes that follow the target's twist arc."""
        fractions = torch.linspace(
            1.0 / waypoint_count,
            1.0,
            waypoint_count,
            dtype=torch.float32,
            device=self.device,
        )
        return self._twisted_grasp_poses_at_fractions(
            link_pose,
            grasp_xpos,
            twist_axis,
            axis_origin,
            twist_angle,
            fractions,
        )

    def _twisted_grasp_poses_at_fractions(
        self,
        link_pose: torch.Tensor,
        grasp_xpos: torch.Tensor,
        twist_axis: torch.Tensor,
        axis_origin: tuple[float, float, float],
        twist_angle: float,
        fractions: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the rigid grasp path at normalized twist-angle fractions."""
        axis = twist_axis.to(device=self.device, dtype=torch.float32)
        axis = axis / torch.linalg.vector_norm(axis)
        fractions = fractions.to(device=self.device, dtype=torch.float32)
        angles = twist_angle * fractions
        rotations = (
            torch.eye(4, dtype=torch.float32, device=self.device)
            .reshape(1, 4, 4)
            .repeat(fractions.numel(), 1, 1)
        )
        rotations[:, :3, :3] = axis_angle_to_rotation_matrix(angles[:, None] * axis)
        link_to_eef = torch.bmm(pose_inv(link_pose), grasp_xpos)
        origin = torch.tensor(axis_origin, dtype=torch.float32, device=self.device)
        to_origin = torch.eye(4, dtype=torch.float32, device=self.device)
        from_origin = torch.eye(4, dtype=torch.float32, device=self.device)
        to_origin[:3, 3] = origin
        from_origin[:3, 3] = -origin
        local_rotations = torch.matmul(
            torch.matmul(to_origin[None], rotations), from_origin[None]
        )
        return torch.matmul(
            torch.matmul(link_pose[:, None], local_rotations[None]),
            link_to_eef[:, None],
        )


__all__ = ["Twist", "TwistGoal", "TwistOptions"]
