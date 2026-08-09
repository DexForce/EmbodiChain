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

"""Expert demonstration environment for opening a drawer."""

from __future__ import annotations

from typing import Any

import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenerator,
    MotionGenOptions,
    MoveType,
    PlanResult,
    PlanState,
    ToppraPlannerCfg,
    ToppraPlanOptions,
    TrajectorySampleMethod,
)
from embodichain.lab.sim.utility.action_utils import interpolate_with_nums

__all__ = ["OpenDrawerEnv"]


def _require_plan_positions(result: PlanResult, *, phase: str) -> torch.Tensor:
    """Return a successful single-environment trajectory.

    Args:
        result: Motion-planning result to validate.
        phase: Human-readable planning phase for error reporting.

    Returns:
        Joint positions for the task's single environment.

    Raises:
        RuntimeError: If planning failed or returned no joint positions.
    """
    if not result.is_all_success():
        raise RuntimeError(f"Motion planning failed during {phase}.")
    if result.positions is None:
        raise RuntimeError(
            f"Motion planning returned no joint positions during {phase}."
        )
    return result.positions[0]


@register_env("OpenDrawer-v1", max_episode_steps=300)
class OpenDrawerEnv(EmbodiedEnv):
    """Open a sliding drawer with the right arm of a CobotMagic robot."""

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        """Initialize the environment and its TOPPRA motion generator.

        Args:
            cfg: Declarative environment configuration.
            **kwargs: Additional arguments forwarded to :class:`EmbodiedEnv`.
        """
        super().__init__(cfg, **kwargs)

        self.motion_gen = MotionGenerator(
            cfg=MotionGenCfg(
                planner_cfg=ToppraPlannerCfg(
                    robot_uid=self.robot.uid,
                )
            )
        )
        self.eef_open = self.robot.get_qpos_limits(name="right_eef")[:, :, 1]
        self.eef_close = self.robot.get_qpos_limits(name="right_eef")[:, :, 0]

    def _generate_eef_motion(
        self, num_steps: int = 10, *, opening: bool = True
    ) -> torch.Tensor:
        """Interpolate the right gripper between its closed and open limits.

        Args:
            num_steps: Number of trajectory samples.
            opening: Whether to open rather than close the gripper.

        Returns:
            Gripper joint trajectory with shape ``(num_steps, eef_dof)``.
        """
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2.")

        current_qpos = self.eef_close if opening else self.eef_open
        target_qpos = self.eef_open if opening else self.eef_close
        return interpolate_with_nums(
            torch.stack([current_qpos, target_qpos], dim=1),
            interp_nums=[num_steps - 1],
            device=self.device,
        ).squeeze(0)

    def create_demo_action_list(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Generate an expert trajectory that grasps and pulls the drawer handle.

        The demonstration is defined for the single-environment CobotMagic task
        configuration and consists of four phases: move to the start pose,
        approach the handle, close the gripper, and pull the drawer open.

        Returns:
            Joint-position actions with shape ``(num_steps, action_dof)``.

        Raises:
            ValueError: If the environment contains more than one arena.
            RuntimeError: If any motion-planning phase fails.
        """
        if self.num_envs != 1:
            raise ValueError(
                "OpenDrawerEnv expert demonstrations currently require num_envs=1."
            )

        qpos_start = torch.tensor(
            [[0.0, 2.06, -0.75, 0.0, -1.20, 1.6]],
            dtype=torch.float32,
            device=self.device,
        )

        options_to_start = MotionGenOptions(
            control_part="right_arm",
            is_interpolate=True,
            start_qpos=self.robot.get_qpos("right_arm")[0],
            plan_opts=ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=50,
            ),
        )
        plan_to_start_result = self.motion_gen.generate(
            target_states=[
                PlanState.single(move_type=MoveType.JOINT_MOVE, qpos=qpos_start[0])
            ],
            options=options_to_start,
        )
        plan_to_start = _require_plan_positions(
            plan_to_start_result, phase="move to start"
        )

        xpos_begin = self.robot.compute_fk(
            name="right_arm", qpos=qpos_start, to_matrix=True
        )[0]
        xpos_mid = xpos_begin.clone()
        xpos_mid[0, 3] += 0.11

        options_to_handle = MotionGenOptions(
            control_part="right_arm",
            is_interpolate=True,
            is_linear=True,
            start_qpos=qpos_start[0],
            plan_opts=ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=50,
            ),
        )
        plan_to_handle_result = self.motion_gen.generate(
            target_states=[
                PlanState.single(move_type=MoveType.EEF_MOVE, xpos=xpos)
                for xpos in (xpos_begin, xpos_mid)
            ],
            options=options_to_handle,
        )
        plan_to_handle = _require_plan_positions(
            plan_to_handle_result, phase="handle approach"
        )

        options_leave_handle = MotionGenOptions(
            control_part="right_arm",
            is_interpolate=True,
            is_linear=True,
            start_qpos=plan_to_handle[-1],
            plan_opts=ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=50,
            ),
        )
        plan_leave_handle_result = self.motion_gen.generate(
            target_states=[
                PlanState.single(move_type=MoveType.EEF_MOVE, xpos=xpos)
                for xpos in (xpos_mid, xpos_begin)
            ],
            options=options_leave_handle,
        )
        plan_leave_handle = _require_plan_positions(
            plan_leave_handle_result, phase="drawer pull"
        )

        num_grasp_steps = 20
        eef_grasp_motion = self._generate_eef_motion(
            num_steps=num_grasp_steps, opening=False
        )

        len_to_start = plan_to_start.shape[0]
        len_to_handle = plan_to_handle.shape[0]
        len_leave_handle = plan_leave_handle.shape[0]
        total_len = len_to_start + len_to_handle + num_grasp_steps + len_leave_handle
        trajectory = torch.zeros(
            (total_len, self.robot.dof),
            dtype=torch.float32,
            device=self.device,
        )

        right_arm_ids = self.robot.get_joint_ids("right_arm")
        right_eef_ids = self.robot.get_joint_ids("right_eef")
        idx = 0

        trajectory[idx : idx + len_to_start, right_arm_ids] = plan_to_start
        trajectory[idx : idx + len_to_start, right_eef_ids] = self._generate_eef_motion(
            num_steps=len_to_start, opening=True
        )
        idx += len_to_start

        trajectory[idx : idx + len_to_handle, right_arm_ids] = plan_to_handle
        trajectory[idx : idx + len_to_handle, right_eef_ids] = self.eef_open.expand(
            len_to_handle, -1
        )
        idx += len_to_handle

        trajectory[idx : idx + num_grasp_steps, right_arm_ids] = (
            plan_to_handle[-1].unsqueeze(0).expand(num_grasp_steps, -1)
        )
        trajectory[idx : idx + num_grasp_steps, right_eef_ids] = eef_grasp_motion
        idx += num_grasp_steps

        trajectory[idx : idx + len_leave_handle, right_arm_ids] = plan_leave_handle
        trajectory[idx : idx + len_leave_handle, right_eef_ids] = self.eef_close.expand(
            len_leave_handle, -1
        )

        return trajectory[:, self.active_joint_ids]
