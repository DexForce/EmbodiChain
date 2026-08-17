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

"""PressButton atomic action implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch

from ._helpers import arm_qpos_from_state
from embodichain.utils.math import get_relative_rotation
from ..affordance import PressButtonAffordance
from ..control import GRASP_COMMAND
from ..core import AtomicAction, ObjectSemantics
from ..effects import StateDelta
from ..goals import ObjectActionGoal
from ..invocation import ActionOptions, ResolvedActionRequest
from ..plans import ActionPlan
from ..state import PlanningContext
from ..trajectory_ops import (
    build_pose_plan_states,
    interpolate_hand_qpos,
    translate_pose_world,
)


@dataclass(frozen=True, slots=True, eq=False)
class PressButtonGoal(ObjectActionGoal):
    """Articulation-link button described by a press affordance."""

    goal_kind: ClassVar[str] = "press_button"


@dataclass(frozen=True, slots=True, eq=False)
class PressButtonOptions(ActionOptions):
    """Per-invocation button-pressing behavior."""

    hand_interp_steps: int = 5
    """Number of waypoints used to close the hand."""

    approach_distance: float = 0.1
    """Distance from the button surface opposite the press direction."""

    press_distance: float = 0.05
    """Distance traveled into the button along its press axis."""

    def __post_init__(self) -> None:
        if self.hand_interp_steps < 1:
            raise ValueError("hand_interp_steps must be at least 1.")
        if not math.isfinite(self.approach_distance):
            raise ValueError("approach_distance must be finite.")
        if self.approach_distance < 0.0:
            raise ValueError("approach_distance must be non-negative.")
        if not math.isfinite(self.press_distance):
            raise ValueError("press_distance must be finite.")
        if self.press_distance <= 0.0:
            raise ValueError("press_distance must be positive.")


class PressButton(AtomicAction[PressButtonGoal, PressButtonOptions]):
    """Close the gripper, approach and press a button, then retract."""

    skill_id: ClassVar[str] = "press_button"
    GoalType: ClassVar[type] = PressButtonGoal
    OptionsType: ClassVar[type] = PressButtonOptions
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    end_effector_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def __init__(self, default_options: PressButtonOptions | None = None) -> None:
        super().__init__(default_options)

    def _on_bind(self) -> None:
        """Resolve dimensions owned by the engine's robot."""
        self.n_envs = self.robot.get_qpos().shape[0]
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
        request: ResolvedActionRequest[PressButtonGoal, PressButtonOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan close, approach, press, and retract without stepping simulation."""
        target = self.require_goal(request)
        affordance = self._require_press_button_affordance(target.semantics)
        options = request.skill_options
        manipulator = request.binding.manipulator()
        end_effector = request.binding.end_effector()
        arm_joint_ids = list(manipulator.joint_ids)
        hand_joint_ids = list(end_effector.joint_ids)
        start_arm_qpos = arm_qpos_from_state(context, arm_joint_ids)
        start_hand_qpos = context.last_qpos[:, hand_joint_ids]
        hand_grasp_qpos = end_effector.joint_positions(
            GRASP_COMMAND,
            n_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )

        link_pose = affordance.get_link_pose().to(
            device=self.device, dtype=torch.float32
        )
        if link_pose.shape != (self.n_envs, 4, 4):
            raise ValueError(
                "Articulation link pose must have shape "
                f"({self.n_envs}, 4, 4), got {tuple(link_pose.shape)}."
            )
        contact_xpos = affordance.get_press_pose(link_pose).to(
            device=self.device, dtype=torch.float32
        )
        contact_xpos = self._find_symmetric_nearest_xpos(
            contact_xpos,
            reference_xpos=self.robot.compute_fk(
                qpos=start_arm_qpos, name=manipulator.name, to_matrix=True
            ),
        )
        approach_xpos = translate_pose_world(
            contact_xpos,
            -contact_xpos[:, :3, 2] * options.approach_distance,
        )
        pressed_xpos = translate_pose_world(
            contact_xpos,
            contact_xpos[:, :3, 2] * options.press_distance,
        )
        n_approach, n_press, n_retract = self._motion_segment_lengths(
            request.motion_policy.sample_count,
            options.hand_interp_steps,
        )
        hand_close = interpolate_hand_qpos(
            start_hand_qpos,
            hand_grasp_qpos,
            n_waypoints=options.hand_interp_steps,
        )
        approach_success, approach_arm = self._plan_pose_segment(
            approach_xpos,
            start_arm_qpos,
            manipulator.name,
            request,
            n_approach,
        )
        press_success, press_arm = self._plan_pose_segment(
            pressed_xpos,
            approach_arm[:, -1],
            manipulator.name,
            request,
            n_press,
        )
        retract_success, retract_arm = self._plan_pose_segment(
            approach_xpos,
            press_arm[:, -1],
            manipulator.name,
            request,
            n_retract,
        )
        success = approach_success & press_success & retract_success

        parts = (hand_close, approach_arm, press_arm, retract_arm)
        lengths = tuple(part.shape[1] for part in parts)
        full = torch.empty(
            (self.n_envs, sum(lengths), self.robot_dof),
            dtype=context.robot.qpos.dtype,
            device=self.device,
        )
        full[:] = context.last_qpos.unsqueeze(1)
        offset = 0

        stop = offset + hand_close.shape[1]
        full[:, offset:stop, arm_joint_ids] = start_arm_qpos.unsqueeze(1)
        full[:, offset:stop, hand_joint_ids] = hand_close
        offset = stop

        for arm in (approach_arm, press_arm, retract_arm):
            stop = offset + arm.shape[1]
            full[:, offset:stop, arm_joint_ids] = arm
            full[:, offset:stop, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
            offset = stop

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=full,
            expected_effects=StateDelta(),
            segment_lengths={
                "close": lengths[0],
                "approach": lengths[1],
                "press": lengths[2],
                "retract": lengths[3],
            },
        )

    @staticmethod
    def _require_press_button_affordance(
        semantics: ObjectSemantics,
    ) -> PressButtonAffordance:
        affordance = semantics.affordance
        if not isinstance(affordance, PressButtonAffordance):
            raise ValueError("PressButton requires a PressButtonAffordance.")
        return affordance

    @staticmethod
    def _motion_segment_lengths(
        sample_count: int,
        hand_interp_steps: int,
    ) -> tuple[int, int, int]:
        motion_count = sample_count - hand_interp_steps
        if motion_count < 6:
            raise ValueError(
                "Not enough waypoints for PressButton. Increase sample_count or "
                "decrease hand_interp_steps."
            )
        base, remainder = divmod(motion_count, 3)
        values = [base + (index < remainder) for index in range(3)]
        return values[0], values[1], values[2]

    def _plan_pose_segment(
        self,
        target_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
        request: ResolvedActionRequest[PressButtonGoal, PressButtonOptions],
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


__all__ = ["PressButton", "PressButtonGoal", "PressButtonOptions"]
