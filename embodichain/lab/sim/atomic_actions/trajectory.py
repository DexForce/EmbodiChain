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

"""Stateless trajectory builder utilities for atomic actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.lab.sim.planners import PlanState
from embodichain.lab.sim.planners.motion_generator import MotionGenOptions
from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator


class TrajectoryBuilder:
    """Stateless trajectory utilities shared by every atomic action.

    Holds a reference to the motion generator (and through it, the robot and
    device) so callers don't have to thread those through each helper call.
    All methods are pure: no per-call state is kept on the builder.
    """

    def __init__(self, motion_generator: MotionGenerator) -> None:
        self.motion_generator = motion_generator
        self.robot = motion_generator.robot
        self.device = self.robot.device

    # ------------------------------------------------------------------
    # Success / shape helpers
    # ------------------------------------------------------------------

    def all_envs_success(self, is_success: bool | torch.Tensor) -> bool:
        """Return true only when all environments report success."""
        if isinstance(is_success, torch.Tensor):
            return bool(torch.all(is_success).item())
        return bool(is_success)

    def resolve_pose_target(self, target: torch.Tensor, *, n_envs: int) -> torch.Tensor:
        """Broadcast a (4, 4) pose to (n_envs, 4, 4) or validate batched shape."""
        if not isinstance(target, torch.Tensor):
            logger.log_error(
                f"target must be torch.Tensor of shape (4, 4) or ({n_envs}, 4, 4)",
                TypeError,
            )
        target = target.to(device=self.device, dtype=torch.float32)
        if target.shape == (4, 4):
            target = target.unsqueeze(0).repeat(n_envs, 1, 1)
        if target.shape != (n_envs, 4, 4):
            logger.log_error(
                f"target tensor must have shape (4, 4) or ({n_envs}, 4, 4), "
                f"but got {target.shape}",
                ValueError,
            )
        return target

    def resolve_start_qpos(
        self,
        start_qpos: torch.Tensor | None,
        *,
        n_envs: int,
        arm_dof: int,
        control_part: str,
    ) -> torch.Tensor:
        """Resolve planning start joint positions into batched arm joint positions."""
        if start_qpos is None:
            start_qpos = self.robot.get_qpos(name=control_part)
        if start_qpos.shape == (arm_dof,):
            start_qpos = start_qpos.unsqueeze(0).repeat(n_envs, 1)
        if start_qpos.shape != (n_envs, arm_dof):
            logger.log_error(
                f"start_qpos must have shape ({n_envs}, {arm_dof}), "
                f"but got {start_qpos.shape}",
                ValueError,
            )
        return start_qpos

    def resolve_joint_target(
        self,
        target_qpos: torch.Tensor,
        *,
        n_envs: int,
        joint_dof: int,
        control_part: str,
    ) -> torch.Tensor:
        """Resolve a joint-space target into batched control-part joint positions."""
        if not isinstance(target_qpos, torch.Tensor):
            logger.log_error(
                f"target qpos for '{control_part}' must be a torch.Tensor with shape "
                f"({joint_dof},) or ({n_envs}, {joint_dof})",
                TypeError,
            )
        target_qpos = target_qpos.to(device=self.device, dtype=torch.float32)
        if target_qpos.shape == (joint_dof,):
            target_qpos = target_qpos.unsqueeze(0).repeat(n_envs, 1)
        if target_qpos.shape != (n_envs, joint_dof):
            logger.log_error(
                f"target qpos for '{control_part}' must have shape ({joint_dof},) "
                f"or ({n_envs}, {joint_dof}), but got {target_qpos.shape}",
                ValueError,
            )
        return target_qpos

    # ------------------------------------------------------------------
    # Pose math
    # ------------------------------------------------------------------

    def apply_local_offset(
        self, pose: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """Apply a world-frame translational offset to a batched pose.

        Despite the historical method name, ``offset`` is added directly to the
        translation column and is not rotated by each pose's orientation.
        """
        if not (pose.dim() == 3 and pose.shape[1:] == (4, 4)):
            logger.log_error("pose must have shape [N, 4, 4]", ValueError)
        offset = offset.to(device=pose.device, dtype=pose.dtype)
        if offset.dim() == 1:
            offset = offset.unsqueeze(0)
        if not (offset.dim() == 2 and offset.shape[1] == 3):
            logger.log_error("offset must have shape [N, 3] or [3]", ValueError)
        if offset.shape[0] not in (1, pose.shape[0]):
            logger.log_error(
                f"offset batch size must be 1 or match pose batch size {pose.shape[0]}, "
                f"but got {offset.shape[0]}",
                ValueError,
            )
        result = pose.clone()
        result[:, :3, 3] += offset
        return result

    # ------------------------------------------------------------------
    # IK / FK convenience
    # ------------------------------------------------------------------

    def ik_solve(
        self,
        target_pose: torch.Tensor,
        *,
        control_part: str,
        qpos_seed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve IK for a single (unbatched) target pose."""
        if qpos_seed is None:
            qpos_seed = self.robot.get_qpos(name=control_part)[0]
        elif qpos_seed.dim() == 2:
            qpos_seed = qpos_seed[0]
        elif qpos_seed.dim() != 1:
            logger.log_error(
                f"qpos_seed must be 1D or 2D, but got shape {qpos_seed.shape}",
                ValueError,
            )
        success, qpos = self.robot.compute_ik(
            pose=target_pose.unsqueeze(0),
            joint_seed=qpos_seed.unsqueeze(0),
            name=control_part,
            env_ids=[0],
        )
        if not success.all():
            logger.log_error(f"IK failed for target pose: {target_pose}", RuntimeError)
        return qpos.squeeze(0)

    def fk_compute(self, qpos: torch.Tensor, *, control_part: str) -> torch.Tensor:
        """Compute forward kinematics for a joint configuration."""
        is_unbatched = qpos.dim() == 1
        if is_unbatched:
            qpos = qpos.unsqueeze(0)
        xpos = self.robot.compute_fk(qpos=qpos, name=control_part, to_matrix=True)
        return xpos.squeeze(0) if is_unbatched else xpos

    # ------------------------------------------------------------------
    # Waypoint splitting
    # ------------------------------------------------------------------

    def split_three_phase(
        self,
        sample_interval: int,
        hand_interp_steps: int,
        *,
        first_phase_ratio: float = 0.6,
        first_phase_name: str = "first",
        third_phase_name: str = "third",
    ) -> tuple[int, int, int]:
        """Split total sample interval into motion, hand-interp, and motion phases."""
        first = int(np.round(sample_interval - hand_interp_steps) * first_phase_ratio)
        if first < 2:
            logger.log_error(
                f"Not enough waypoints for {first_phase_name} trajectory. "
                "Increase sample_interval or decrease hand_interp_steps.",
                ValueError,
            )
        second = hand_interp_steps
        third = sample_interval - first - second
        if third < 2:
            logger.log_error(
                f"Not enough waypoints for {third_phase_name} trajectory. "
                "Increase sample_interval or decrease hand_interp_steps.",
                ValueError,
            )
        return first, second, third

    # ------------------------------------------------------------------
    # MotionGen options
    # ------------------------------------------------------------------

    def build_motion_gen_options(
        self,
        start_qpos: torch.Tensor,
        *,
        sample_interval: int,
        control_part: str,
    ) -> MotionGenOptions:
        """Build planner options. Reads ``start_qpos[0]`` because the planner shares options across envs; pass the full batched tensor for type uniformity with other helpers."""
        return MotionGenOptions(
            start_qpos=start_qpos[0],
            control_part=control_part,
            is_interpolate=True,
            is_linear=False,
            interpolate_position_step=0.001,
            plan_opts=ToppraPlanOptions(sample_interval=sample_interval),
        )

    # ------------------------------------------------------------------
    # Arm trajectory planning
    # ------------------------------------------------------------------

    def plan_arm_traj(
        self,
        target_states_list: list[list[PlanState]],
        start_qpos: torch.Tensor,
        n_waypoints: int,
        *,
        control_part: str,
        arm_dof: int,
    ) -> tuple[bool, torch.Tensor]:
        """Plan batched arm trajectories for all environments."""
        n_envs = start_qpos.shape[0]
        n_state = len(target_states_list[0])
        xpos_traj = torch.zeros(
            (n_envs, n_state, 4, 4), dtype=torch.float32, device=self.device
        )
        for i, target_states in enumerate(target_states_list):
            for j, target_state in enumerate(target_states):
                xpos_traj[i, j] = target_state.xpos

        trajectory = torch.zeros(
            (n_envs, n_state, arm_dof), dtype=torch.float32, device=self.device
        )
        qpos_seed = start_qpos
        for j in range(n_state):
            is_success, qpos = self.robot.compute_ik(
                pose=xpos_traj[:, j], name=control_part, joint_seed=qpos_seed
            )
            if not self.all_envs_success(is_success):
                logger.log_warning(
                    f"Failed to compute IK for target state {j} in some environments."
                )
                return False, trajectory
            trajectory[:, j] = qpos
            qpos_seed = qpos
        trajectory = torch.concatenate([start_qpos.unsqueeze(1), trajectory], dim=1)
        interp = interpolate_with_distance(
            trajectory=trajectory, interp_num=n_waypoints, device=self.device
        )
        return True, interp

    def plan_joint_traj(
        self,
        start_qpos: torch.Tensor,
        target_qpos: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor:
        """Interpolate a joint-space trajectory from ``start_qpos`` to ``target_qpos``."""
        trajectory = torch.stack([start_qpos, target_qpos], dim=1)
        return interpolate_with_distance(
            trajectory=trajectory, interp_num=n_waypoints, device=self.device
        )

    # ------------------------------------------------------------------
    # Hand qpos helpers
    # ------------------------------------------------------------------

    def expand_hand_qpos(
        self, hand_qpos: torch.Tensor, *, n_envs: int, hand_dof: int
    ) -> torch.Tensor:
        """Resolve hand qpos to batched shape ``(n_envs, hand_dof)``."""
        hand_qpos = hand_qpos.to(device=self.device, dtype=torch.float32)
        if hand_qpos.shape == (hand_dof,):
            return hand_qpos.unsqueeze(0).repeat(n_envs, 1)
        if hand_qpos.shape == (n_envs, hand_dof):
            return hand_qpos
        logger.log_error(
            f"hand_qpos must have shape ({hand_dof},) or ({n_envs}, {hand_dof}), "
            f"but got {hand_qpos.shape}",
            ValueError,
        )
        raise AssertionError("unreachable")  # logger.log_error already raised

    def broadcast_hand_qpos_to_waypoints(
        self,
        hand_qpos: torch.Tensor,
        *,
        n_envs: int,
        hand_dof: int,
        n_waypoints: int,
    ) -> torch.Tensor:
        """Expand hand qpos to (n_envs, n_waypoints, hand_dof) by broadcasting the per-env value across all waypoints."""
        return (
            self.expand_hand_qpos(hand_qpos, n_envs=n_envs, hand_dof=hand_dof)
            .unsqueeze(1)
            .repeat(1, n_waypoints, 1)
        )

    def interpolate_hand_qpos(
        self,
        start_hand_qpos: torch.Tensor,
        end_hand_qpos: torch.Tensor,
        *,
        n_waypoints: int,
    ) -> torch.Tensor:
        """Interpolate hand joint positions between two gripper states."""
        is_unbatched = start_hand_qpos.dim() == 1 and end_hand_qpos.dim() == 1
        start_hand_qpos = start_hand_qpos.to(self.device)
        end_hand_qpos = end_hand_qpos.to(self.device)
        if start_hand_qpos.dim() == 1:
            start_hand_qpos = start_hand_qpos.unsqueeze(0)
        if end_hand_qpos.dim() == 1:
            end_hand_qpos = end_hand_qpos.unsqueeze(0)
        weights = torch.linspace(
            0, 1, steps=n_waypoints, device=self.device, dtype=start_hand_qpos.dtype
        )
        result = torch.lerp(
            start_hand_qpos.unsqueeze(1),
            end_hand_qpos.unsqueeze(1),
            weights[None, :, None],
        )
        if is_unbatched:
            return result.squeeze(0)
        return result


__all__ = ["TrajectoryBuilder"]
