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
from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.utils.math import (
    axis_angle_to_rotation_matrix,
    pose_inv,
    get_relative_rotation,
)

from embodichain.lab.sim.atomic_actions.affordance import TwistAffordance
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
    OPEN_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import AtomicAction, ObjectSemantics
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.goals import ObjectActionGoal
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import ActionPlan
from embodichain.lab.sim.atomic_actions.primitives._helpers import arm_qpos_from_state
from embodichain.lab.sim.atomic_actions.requirements import (
    ActionBindingRoute,
    CARTESIAN_POSE_CAPABILITY,
    DisjointSlotEndpoints,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    interpolate_hand_qpos,
    translate_pose_world,
)


@dataclass(frozen=True, slots=True, eq=False)
class TwistGoal(ObjectActionGoal):
    """Target object described by a twist affordance."""

    goal_kind: ClassVar[str] = "twist"


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
        if self.pre_grasp_distance < 0.0:
            raise ValueError("pre_grasp_distance must be non-negative.")
        if not math.isfinite(self.twist_angle):
            raise ValueError("twist_angle must be finite.")


class Twist(AtomicAction[TwistGoal, TwistOptions]):
    """Approach, grasp, twist, release, and retract from a target."""

    skill_id: ClassVar[str] = "twist"
    GoalType: ClassVar[type] = TwistGoal
    OptionsType: ClassVar[type] = TwistOptions
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    end_effector_roles: ClassVar[tuple[str, ...]] = ("primary",)
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="primary",
                endpoints=(
                    SkillEndpointRequirement(
                        endpoint_id="motion",
                        capabilities=frozenset(
                            {
                                CARTESIAN_POSE_CAPABILITY,
                                FORWARD_KINEMATICS_CAPABILITY,
                            }
                        ),
                        route=ActionBindingRoute("manipulator", "primary"),
                    ),
                    SkillEndpointRequirement(
                        endpoint_id="grasp",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        required_commands={
                            OPEN_COMMAND: JointPositionCommand,
                            GRASP_COMMAND: JointPositionCommand,
                        },
                        route=ActionBindingRoute("end_effector", "primary"),
                    ),
                ),
                constraints=(DisjointSlotEndpoints(("motion", "grasp")),),
            ),
        ),
    )

    def __init__(self, default_options: TwistOptions | None = None) -> None:
        super().__init__(default_options)

    def _on_bind(self) -> None:
        """Resolve dimensions owned by the engine's robot."""
        self.num_envs = self.robot.get_qpos().shape[0]
        self.robot_dof = self.robot.dof

    def _find_symmetric_nearest_xpos(
        self, target_xpos: torch.Tensor, reference_xpos: torch.Tensor
    ) -> torch.Tensor:
        """Find the nearest symmetric pose to the reference pose."""
        symmetric_xpos = target_xpos.clone()
        symmetric_xpos[:, :3, 0] = -symmetric_xpos[:, :3, 0]
        symmetric_xpos[:, :3, 1] = -symmetric_xpos[:, :3, 1]
        angle_a = get_relative_rotation(
            reference_xpos[:, :3, :3], target_xpos[:, :3, :3]
        )
        angle_b = get_relative_rotation(
            reference_xpos[:, :3, :3], symmetric_xpos[:, :3, :3]
        )
        choose_target = (angle_a < angle_b)[..., None, None]
        target_xpos = torch.where(choose_target, target_xpos, symmetric_xpos)
        return target_xpos

    def _plan(
        self,
        request: ResolvedActionRequest[TwistGoal, TwistOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan all six twisting segments without stepping simulation."""
        target = self.require_goal(request)
        affordance = self._require_twist_affordance(target.semantics)
        options = request.skill_options
        manipulator = request.binding.manipulator()
        end_effector = request.binding.end_effector()
        arm_joint_ids = list(manipulator.joint_ids)
        hand_joint_ids = list(end_effector.joint_ids)
        start_arm_qpos = arm_qpos_from_state(context, arm_joint_ids)
        hand_open_qpos = end_effector.joint_positions(
            OPEN_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        hand_grasp_qpos = end_effector.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )

        link_pose = affordance.get_link_pose().to(
            device=self.device, dtype=torch.float32
        )
        if link_pose.shape != (self.num_envs, 4, 4):
            raise ValueError(
                "Twist target pose must have shape "
                f"({self.num_envs}, 4, 4), got {tuple(link_pose.shape)}."
            )
        grasp_xpos = affordance.get_grasp_pose(link_pose).to(
            device=self.device, dtype=torch.float32
        )
        grasp_xpos = self._find_symmetric_nearest_xpos(
            grasp_xpos,
            reference_xpos=self.robot.compute_fk(
                qpos=start_arm_qpos, name=manipulator.name, to_matrix=True
            ),
        )
        pre_grasp_xpos = translate_pose_world(
            grasp_xpos,
            -grasp_xpos[:, :3, 2] * options.pre_grasp_distance,
        )
        twist_xpos = self._twisted_grasp_poses(
            link_pose,
            grasp_xpos,
            affordance.twist_axis,
            options.twist_angle,
            options.twist_waypoint_count,
        )

        n_approach, n_reach, n_twist, n_retract = self._motion_segment_lengths(
            request.motion_policy.sample_count,
            options.hand_interp_steps,
        )

        approach_success, approach_arm = self._plan_pose_segment(
            pre_grasp_xpos,
            start_arm_qpos,
            manipulator.name,
            request,
            n_approach,
        )
        reach_success, reach_arm = self._plan_pose_segment(
            grasp_xpos,
            approach_arm[:, -1],
            manipulator.name,
            request,
            n_reach,
        )
        twist_success, twist_arm = self._plan_pose_segment(
            twist_xpos,
            reach_arm[:, -1],
            manipulator.name,
            request,
            n_twist,
        )
        retract_success, retract_arm = self._plan_pose_segment(
            pre_grasp_xpos,
            twist_arm[:, -1],
            manipulator.name,
            request,
            n_retract,
        )
        success = approach_success & reach_success & twist_success & retract_success

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
        for arm, hand in zip(arm_parts[:2], arm_hands[:2]):
            stop = offset + arm.shape[1]
            full[:, offset:stop, arm_joint_ids] = arm
            full[:, offset:stop, hand_joint_ids] = hand.unsqueeze(1)
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
            trajectory=full,
            expected_effects=StateDelta(),
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
    ) -> tuple[int, int, int, int]:
        motion_count = sample_count - 2 * hand_interp_steps
        if motion_count < 8:
            raise ValueError(
                "Not enough waypoints for Twist. Increase sample_count or "
                "decrease hand_interp_steps."
            )
        base, remainder = divmod(motion_count, 4)
        values = [base + (index < remainder) for index in range(4)]
        return values[0], values[1], values[2], values[3]

    def _plan_pose_segment(
        self,
        target_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
        request: ResolvedActionRequest[TwistGoal, TwistOptions],
        sample_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.motion_generator.generate(
            build_pose_plan_states(target_pose),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_qpos,
                control_part=control_part,
                sample_count=sample_count,
            ),
        )
        assert isinstance(result.success, torch.Tensor)
        assert result.positions is not None
        return result.success, result.positions

    def _twisted_grasp_poses(
        self,
        link_pose: torch.Tensor,
        grasp_xpos: torch.Tensor,
        twist_axis: torch.Tensor,
        twist_angle: float,
        waypoint_count: int,
    ) -> torch.Tensor:
        """Build Cartesian EEF keyframes that follow the target's twist arc."""
        axis = twist_axis.to(device=self.device, dtype=torch.float32)
        axis = axis / torch.linalg.vector_norm(axis)
        angles = torch.linspace(
            twist_angle / waypoint_count,
            twist_angle,
            waypoint_count,
            dtype=torch.float32,
            device=self.device,
        )
        rotations = (
            torch.eye(4, dtype=torch.float32, device=self.device)
            .reshape(1, 4, 4)
            .repeat(waypoint_count, 1, 1)
        )
        rotations[:, :3, :3] = axis_angle_to_rotation_matrix(angles[:, None] * axis)
        link_to_eef = torch.bmm(pose_inv(link_pose), grasp_xpos)
        return torch.matmul(
            torch.matmul(link_pose[:, None], rotations[None]),
            link_to_eef[:, None],
        )


__all__ = ["Twist", "TwistGoal", "TwistOptions"]
