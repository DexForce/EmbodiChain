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

import torch
import numpy as np

from dataclasses import MISSING
from typing import Dict, List, Tuple, Union, Any
from scipy.spatial.transform import Rotation, Slerp

from embodichain.lab.sim.planners import (
    BasePlannerCfg,
    BasePlannerRuntimeCfg,
    BasePlanner,
    ToppraPlanner,
    ToppraPlannerCfg,
)
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod
from embodichain.lab.sim.objects.robot import Robot
from embodichain.utils import logger, configclass
from .utils import MovePart, MoveType, PlanState, PlanResult


__all__ = ["MotionGenerator", "MotionGenCfg"]


@configclass
class MotionGenCfg:

    planner_cfg: BasePlannerCfg = MISSING


class MotionGenerator:
    r"""Unified motion generator for robot trajectory planning.

    This class provides a unified interface for trajectory planning with and without
    collision checking. It supports V3 environment interfaces and can use different
    types of planners (ToppraPlanner, RRT, PRM, etc.) for trajectory generation.

    Args:
        robot: Robot agent object (must support compute_fk, compute_ik, dof, get_joint_ids)
        uid: Unique identifier for the robot (optional)
        sim: Simulation environment object (optional, reserved for future collision checking)
        planner_type: Type of planner to use (default: "toppra")
        default_velocity: Default velocity limits for each joint (rad/s)
        default_acceleration: Default acceleration limits for each joint (rad/s²)
        collision_margin: Safety margin for collision checking (meters, reserved for future use)
        **kwargs: Additional arguments passed to planner initialization
    """

    _support_planner_dict = {
        "toppra": (ToppraPlanner, ToppraPlannerCfg),
    }

    @classmethod
    def register_planner_type(cls, name: str, planner_class, planner_cfg_class):
        """
        Register a new planner type.
        """
        cls._support_planner_dict[name] = (planner_class, planner_cfg_class)

    def __init__(self, cfg: MotionGenCfg):

        # Create planner based on planner_type
        self.planner: BasePlanner = self._create_planner(cfg.planner_cfg)

        self.robot = self.planner.robot

    def _create_planner(
        self,
        planner_cfg: BasePlannerCfg,
    ) -> BasePlanner:
        r"""Create planner instance based on planner type.

        Args:
            planner_cfg: Configuration object for the planner, must include 'planner_type' attribute

        Returns:
            Planner instance
        """
        planner_type = planner_cfg.planner_type
        if planner_type not in self._support_planner_dict.keys():
            logger.log_error(
                f"Unsupported planner type: {planner_type}. "
                f"Supported types: {list(self._support_planner_dict.keys())}"
            )
        cls = self._support_planner_dict[planner_type][0](cfg=planner_cfg)
        return cls

    def plan(
        self,
        target_states: List[PlanState],
        runtime_cfg: BasePlannerRuntimeCfg = BasePlannerRuntimeCfg(),
    ) -> PlanResult:
        r"""Plan trajectory without collision checking.

        This method generates a smooth trajectory using the selected planner that satisfies
        velocity and acceleration constraints, but does not check for collisions.

        Args:
            current_state: PlanState
            target_states: List of PlanState
            cfg:  Planner runtime configuration.

        Returns:
            PlanResult containing the planned trajectory details.
        """
        if runtime_cfg.is_pre_interpolate:
            # interpolate trajectory to generate more waypoints for smoother motion and better constraint handling
            if target_states[0].move_type == MoveType.TCP_MOVE:
                xpos_list = []
                for state in target_states:
                    if state.move_type != MoveType.TCP_MOVE:
                        logger.log_error(
                            f"All states must be the same. First state is {target_states[0].move_type}, but got {state.move_type}"
                        )
                    xpos_list.append(state.xpos)
                    qpos_list = None
            elif target_states[0].move_type == MoveType.JOINT_MOVE:
                qpos_list = []
                for state in target_states:
                    if state.move_type != MoveType.JOINT_MOVE:
                        logger.log_error(
                            f"All states must be the same. First state is {target_states[0].move_type}, but got {state.move_type}"
                        )
                    qpos_list.append(state.qpos)
                    xpos_list = None
            else:
                logger.log_error(
                    f"Unsupported move type for pre-interpolation: {target_states[0].move_type}"
                )
            init_plan_state, target_plan_states = self.planner.interpolate_trajectory(
                control_part=runtime_cfg.control_part,
                xpos_list=xpos_list,
                qpos_list=qpos_list,
                cfg=runtime_cfg,
            )
        else:
            target_plan_states = target_states

        result = self.planner.plan(
            target_states=target_plan_states, runtime_cfg=runtime_cfg
        )
        return result

    def estimate_trajectory_sample_count(
        self,
        xpos_list: torch.Tensor | list[torch.Tensor] | None = None,
        qpos_list: torch.Tensor | list[torch.Tensor] | None = None,
        step_size: float | torch.Tensor = 0.01,
        angle_step: float | torch.Tensor = np.pi / 90,
        control_part: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Estimate the number of trajectory sampling points required.

        This function estimates the total number of sampling points needed to generate
        a trajectory based on the given waypoints and sampling parameters. Supports
        parallel computation for batched input trajectories.

        Args:
            xpos_list: Tensor of 4x4 transformation matrices, shape [B, N, 4, 4] or [N, 4, 4]
            qpos_list: Tensor of joint positions, shape [B, N, D] or [N, D] (optional)
            step_size: Maximum allowed distance between points (meters). Float or Tensor [B]
            angle_step: Maximum allowed angular difference between points (radians). Float or Tensor [B]
            **kwargs: Additional parameters for further customization

        Returns:
            torch.Tensor: Estimated number of sampling points per trajectory, shape [B]
                          (or scalar tensor if single trajectory)
        """
        # Input validation
        if xpos_list is None and qpos_list is None:
            return torch.tensor(0)

        # Handle lists gracefully if passed by legacy code
        if isinstance(xpos_list, list):
            xpos_list = torch.stack(
                [
                    x if isinstance(x, torch.Tensor) else torch.tensor(x)
                    for x in xpos_list
                ]
            ).float()
        elif isinstance(xpos_list, np.ndarray):
            xpos_list = torch.as_tensor(xpos_list, dtype=torch.float32)

        if isinstance(qpos_list, list):
            qpos_list = torch.stack(
                [
                    q if isinstance(q, torch.Tensor) else torch.tensor(q)
                    for q in qpos_list
                ]
            ).float()
        elif isinstance(qpos_list, np.ndarray):
            qpos_list = torch.as_tensor(qpos_list, dtype=torch.float32)

        device = qpos_list.device if qpos_list is not None else xpos_list.device

        original_dim = qpos_list.dim() if qpos_list is not None else xpos_list.dim()

        # If joint position list is provided but end effector position list is not,
        # convert through forward kinematics
        if qpos_list is not None and xpos_list is None:
            if original_dim == 2:  # [N, D]
                qpos_list = qpos_list.unsqueeze(0)  # [1, N, D]

            B, N, D = qpos_list.shape

            if N < 2:
                return torch.ones((B,), dtype=torch.int32, device=device)

            xpos_list = self.robot.compute_batch_fk(
                qpos=qpos_list,
                name=control_part,
                to_matrix=True,
            )
        else:
            if original_dim == 3:  # [N, 4, 4]
                xpos_list = xpos_list.unsqueeze(0)
            B, N, _, _ = xpos_list.shape

            if N < 2:
                return torch.ones((B,), dtype=torch.int32, device=device)

        # Convert step metrics to tensors
        if not isinstance(step_size, torch.Tensor):
            step_size = torch.full((B,), step_size, device=device, dtype=torch.float32)
        else:
            step_size = step_size.to(device)

        if not isinstance(angle_step, torch.Tensor):
            angle_step = torch.full(
                (B,), angle_step, device=device, dtype=torch.float32
            )
        else:
            angle_step = angle_step.to(device)

        # Calculate position distances
        start_poses = xpos_list[:, :-1]  # [B, N-1, 4, 4]
        end_poses = xpos_list[:, 1:]  # [B, N-1, 4, 4]

        pos_diffs = end_poses[:, :, :3, 3] - start_poses[:, :, :3, 3]
        pos_dists = torch.norm(pos_diffs, dim=-1)  # [B, N-1]
        total_pos_dist = pos_dists.sum(dim=-1)  # [B]

        # Calculate rotation angles
        start_rot = start_poses[:, :, :3, :3]  # [B, N-1, 3, 3]
        end_rot = end_poses[:, :, :3, :3]  # [B, N-1, 3, 3]

        start_rot_T = start_rot.transpose(-1, -2)
        rel_rot = torch.matmul(start_rot_T, end_rot)

        trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
        cos_angle = (trace - 1.0) / 2.0
        # Add epsilon to prevent NaN in acos at boundaries
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6)

        angles = torch.acos(cos_angle)  # [B, N-1]
        total_angle = angles.sum(dim=-1)  # [B]

        # Compute sampling points
        pos_samples = torch.clamp((total_pos_dist / step_size).int(), min=1)
        rot_samples = torch.clamp((total_angle / angle_step).int(), min=1)

        total_samples = torch.max(pos_samples, rot_samples)

        out_samples = torch.clamp(total_samples, min=2)

        if original_dim in (2, 3):  # Reshape back to scalar tensor if not batched
            return out_samples[0]

        return out_samples
