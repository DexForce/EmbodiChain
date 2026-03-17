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
        self.uid = self.planner.cfg.control_part
        self.dof = self.planner.dofs

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

    def _create_state_dict(
        self, position: np.ndarray, velocity: np.ndarray | None = None
    ) -> Dict:
        r"""Create a state dictionary for trajectory planning.

        Args:
            position: Joint positions
            velocity: Joint velocities (optional, defaults to zeros)
            acceleration: Joint accelerations (optional, defaults to zeros)

        Returns:
            State dictionary with 'position', 'velocity', 'acceleration'
        """
        if velocity is None:
            velocity = np.zeros(self.dof)

        if isinstance(position, torch.Tensor) | isinstance(position, np.ndarray):
            position = position.squeeze()

        return {
            "position": (
                position.tolist() if isinstance(position, np.ndarray) else position
            ),
            "velocity": (
                velocity.tolist() if isinstance(velocity, np.ndarray) else velocity
            ),
            "acceleration": [0.0] * self.dof,
        }

    def plan(
        self,
        current_state: PlanState,
        target_states: List[PlanState],
        cfg: BasePlannerRuntimeCfg = BasePlannerRuntimeCfg(),
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
        # Plan trajectory using selected planner
        result = self.planner.plan(
            current_state=current_state, target_states=target_states, cfg=cfg
        )
        return result

    def create_discrete_trajectory(
        self,
        xpos_list: torch.Tensor | None = None,
        qpos_list: torch.Tensor | None = None,
        cfg: BasePlannerRuntimeCfg = BasePlannerRuntimeCfg(),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Generate a discrete trajectory between waypoints using cartesian or joint space interpolation.

        This method supports two trajectory planning approaches:
        1. Linear interpolation: Fast, uniform spacing, no dynamics constraints
        2. Planner-based: Smooth, considers velocity/acceleration limits, realistic motion

        Args:
            xpos_list: Waypoints as a tensor of 4x4 transformation matrices [N, 4, 4] (optional)
            qpos_list: Joint configurations as a tensor [N, dof] (optional)
            cfg:  Planner runtime configuration.

        Returns:
            A tuple containing:
            - torch.Tensor: Joint space trajectory tensor [N, dof]
            - torch.Tensor: Cartesian space trajectory tensor [N, 4, 4]
        """
        init_plan_state, target_plan_states = self.planner.interpolate_trajectory(
            control_part=self.uid, xpos_list=xpos_list, qpos_list=qpos_list, cfg=cfg
        )

        if init_plan_state is None or target_plan_states is None:
            empty_qpos = torch.empty((0, self.dof), dtype=torch.float32)
            empty_xpos = torch.empty((0, 4, 4), dtype=torch.float32)
            return empty_qpos, empty_xpos

        # Plan trajectory using internal plan method
        plan_result = self.plan(
            current_state=init_plan_state, target_states=target_plan_states, cfg=cfg
        )

        if not plan_result.success or plan_result.positions is None:
            logger.log_error("Failed to plan trajectory")

        # Convert outputs to tensor format
        out_qpos_tensor = (
            plan_result.positions.to(dtype=torch.float32, device=self.robot.device)
            if isinstance(plan_result.positions, torch.Tensor)
            else torch.as_tensor(
                plan_result.positions, dtype=torch.float32, device=self.robot.device
            )
        )
        if out_qpos_tensor.dim() == 1:
            out_qpos_tensor = out_qpos_tensor.unsqueeze(0)

        out_xpos_tensor = self.robot.compute_batch_fk(
            qpos=out_qpos_tensor.unsqueeze(0),
            name=self.uid,
            to_matrix=True,
        ).squeeze_(0)

        return out_qpos_tensor, out_xpos_tensor

    def estimate_trajectory_sample_count(
        self,
        xpos_list: torch.Tensor | list[torch.Tensor] | None = None,
        qpos_list: torch.Tensor | list[torch.Tensor] | None = None,
        step_size: float | torch.Tensor = 0.01,
        angle_step: float | torch.Tensor = np.pi / 90,
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
                name=self.uid,
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
