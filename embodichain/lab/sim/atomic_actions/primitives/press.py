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

"""Press atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import configclass, logger

from ._helpers import arm_qpos_from_state
from ..core import (
    ActionCfg,
    AtomicAction,
)
from ..goals import PoseGoalValue, resolve_pose_goal, validate_pose_goal
from ..invocation import ActionInvocation
from ..plans import ActionPlan
from ..state import PlanningContext
from ..trajectory import TrajectoryBuilder


@dataclass(frozen=True, slots=True, eq=False)
class PressGoal:
    """Single end-effector contact pose used by :class:`Press`."""

    goal_kind: ClassVar[str] = "press_pose"

    xpos: PoseGoalValue
    """Contact pose, shape ``(4, 4)`` or ``(n_envs, 4, 4)``."""

    def __post_init__(self) -> None:
        validate_pose_goal(self.xpos, "xpos", allow_waypoints=False)


@configclass
class PressCfg(ActionCfg):
    name: str = "press"
    """Name of the action, used for identification and logging."""

    control_part: str = "arm"
    """Manipulator resource used by this configured action instance."""

    hand_interp_steps: int = 5
    """Number of waypoints for closing the gripper before pressing."""

    hand_control_part: str = "hand"
    """Name of the robot part that controls the hand joints."""

    hand_close_qpos: torch.Tensor | None = None
    """Joint positions for the closed hand state, shape ``[hand_dof,]``."""


class Press(AtomicAction[PressGoal]):
    """Close the gripper, press down to a target pose, then return."""

    skill_id: ClassVar[str] = "press"
    GoalType: ClassVar[type] = PressGoal
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    end_effector_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def __init__(
        self,
        motion_generator,
        cfg: PressCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or PressCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.hand_dof = len(self.hand_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_close_qpos is None:
            logger.log_error(
                "hand_close_qpos must be specified in PressCfg", ValueError
            )
        self.hand_close_qpos = self.builder.expand_hand_qpos(
            self.cfg.hand_close_qpos,
            n_envs=self.n_envs,
            hand_dof=self.hand_dof,
        )

    def _plan(
        self,
        invocation: ActionInvocation[PressGoal],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan a close, press, and retract sequence."""
        target = self.require_goal(invocation)
        if invocation.binding.manipulator() != self.cfg.control_part:
            raise ValueError("Press manipulator binding does not match its config.")
        if invocation.binding.end_effector() != self.cfg.hand_control_part:
            raise ValueError("Press end-effector binding does not match its config.")
        state = context
        press_xpos = self.builder.resolve_pose_target(
            resolve_pose_goal(target.xpos, context, name="xpos"),
            n_envs=self.n_envs,
        )
        start_arm_qpos = self.builder.resolve_start_qpos(
            arm_qpos_from_state(state, self.arm_joint_ids),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        start_hand_qpos = state.last_qpos[:, self.hand_joint_ids]

        n_close, n_down, n_back = self._compute_phase_waypoints(
            invocation.motion_policy.sample_count
        )

        hand_close_path = self.builder.interpolate_hand_qpos(
            start_hand_qpos,
            self.hand_close_qpos,
            n_waypoints=n_close,
        )

        target_states_list = [
            [PlanState(xpos=press_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        down_success, down_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_down,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
            cfg=invocation.motion_policy,
        )

        press_arm_qpos = down_arm[:, -1, :]
        back_success, back_arm = self.builder.plan_joint_motion(
            press_arm_qpos,
            start_arm_qpos,
            n_back,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
            cfg=invocation.motion_policy,
        )
        success = down_success & back_success

        # Allocate from the actually-returned phase lengths so collision-aware
        # planners (which preserve their own sample count) are accommodated.
        n_down_actual = down_arm.shape[1]
        n_back_actual = back_arm.shape[1]
        full = torch.empty(
            (self.n_envs, n_close + n_down_actual + n_back_actual, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_close, self.arm_joint_ids] = start_arm_qpos.unsqueeze(1)
        full[:, :n_close, self.hand_joint_ids] = hand_close_path
        full[:, n_close : n_close + n_down_actual, self.arm_joint_ids] = down_arm
        full[:, n_close : n_close + n_down_actual, self.hand_joint_ids] = (
            self.hand_close_qpos.unsqueeze(1)
        )
        full[:, n_close + n_down_actual :, self.arm_joint_ids] = back_arm
        full[:, n_close + n_down_actual :, self.hand_joint_ids] = (
            self.hand_close_qpos.unsqueeze(1)
        )

        return self.build_plan(
            invocation,
            context,
            success=success,
            trajectory=full,
            phase_name="press",
        )

    def _compute_phase_waypoints(self, sample_count: int) -> tuple[int, int, int]:
        """Split the invocation sample budget across press phases."""
        n_close = self.cfg.hand_interp_steps
        if n_close < 1:
            logger.log_error(
                "hand_interp_steps must be at least 1 for PressCfg.", ValueError
            )

        motion_waypoints = sample_count - n_close
        n_down = motion_waypoints // 2
        n_back = motion_waypoints - n_down
        if n_down < 2 or n_back < 2:
            logger.log_error(
                "Not enough waypoints for press trajectory. Increase "
                "MotionPolicy.sample_count or decrease hand_interp_steps.",
                ValueError,
            )
        return n_close, n_down, n_back


__all__ = ["Press", "PressCfg", "PressGoal"]
