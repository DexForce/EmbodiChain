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
        from embodichain.utils.utility import get_class_instance

        cls = get_class_instance(
            "embodichain.lab.sim.planners", f"{planner_cfg.planner_type}Planner"
        )(cfg=planner_cfg)

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
        sample_method: TrajectorySampleMethod = TrajectorySampleMethod.TIME,
        sample_interval: Union[float, int] = 0.01,
        **kwargs,
    ) -> PlanResult:
        r"""Plan trajectory without collision checking.

        This method generates a smooth trajectory using the selected planner that satisfies
        velocity and acceleration constraints, but does not check for collisions.

        Args:
            current_state: PlanState
            target_states: List of PlanState
            sample_method: Sampling method (TIME or QUANTITY)
            sample_interval: Sampling interval (time in seconds for TIME method, or number of points for QUANTITY)
            **kwargs: Additional arguments

        Returns:
            PlanResult containing the planned trajectory details.
        """
        # Plan trajectory using selected planner
        result = self.planner.plan(
            current_state=current_state,
            target_states=target_states,
            sample_method=sample_method,
            sample_interval=sample_interval,
            **kwargs,
        )

        return result

    def create_discrete_trajectory(
        self,
        xpos_list: torch.Tensor | None = None,
        qpos_list: torch.Tensor | None = None,
        is_use_current_qpos: bool = True,
        is_linear: bool = False,
        sample_method: TrajectorySampleMethod = TrajectorySampleMethod.QUANTITY,
        sample_num: float | int = 20,
        qpos_seed: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Generate a discrete trajectory between waypoints using cartesian or joint space interpolation.

        This method supports two trajectory planning approaches:
        1. Linear interpolation: Fast, uniform spacing, no dynamics constraints
        2. Planner-based: Smooth, considers velocity/acceleration limits, realistic motion

        Args:
            xpos_list: Waypoints as a tensor of 4x4 transformation matrices [N, 4, 4] (optional)
            qpos_list: Joint configurations as a tensor [N, dof] (optional)
            is_use_current_qpos: Whether to use current joint angles as starting point
            is_linear: If True, use cartesian linear interpolation, else joint space
            sample_method: Sampling method (QUANTITY or TIME)
            sample_num: Number of interpolated points for final trajectory
            qpos_seed: Initial joint configuration for IK solving
            **kwargs: Additional arguments

        Returns:
            A tuple containing:
            - torch.Tensor: Joint space trajectory tensor [N, dof]
            - torch.Tensor: Cartesian space trajectory tensor [N, 4, 4]
        """

        def interpolate_xpos(
            current_xpos: np.ndarray, target_xpos: np.ndarray, num_samples: int
        ) -> np.ndarray:
            """Interpolate between two poses using vectorized Slerp + linear translation."""
            num_samples = max(2, int(num_samples))

            interp_ratios = np.linspace(0.0, 1.0, num_samples)
            slerp = Slerp(
                [0.0, 1.0],
                Rotation.from_matrix([current_xpos[:3, :3], target_xpos[:3, :3]]),
            )
            interp_rots = slerp(interp_ratios).as_matrix()
            interp_trans = (1.0 - interp_ratios[:, None]) * current_xpos[
                :3, 3
            ] + interp_ratios[:, None] * target_xpos[:3, 3]

            interp_poses = np.repeat(np.eye(4)[None, :, :], num_samples, axis=0)
            interp_poses[:, :3, :3] = interp_rots
            interp_poses[:, :3, 3] = interp_trans
            return interp_poses

        def calculate_point_allocations(
            xpos_list: torch.Tensor | np.ndarray,
            step_size: float = 0.002,
            angle_step: float = np.pi / 90,
        ) -> List[int]:
            """Calculate interpolation points for each segment with vectorized tensor ops."""
            if not isinstance(xpos_list, torch.Tensor):
                xpos_tensor = torch.as_tensor(
                    np.asarray(xpos_list),
                    dtype=torch.float32,
                    device=self.robot.device,
                )
            else:
                xpos_tensor = xpos_list.to(
                    dtype=torch.float32, device=self.robot.device
                )

            if xpos_tensor.dim() != 3 or xpos_tensor.shape[0] < 2:
                return []

            start_poses = xpos_tensor[:-1]  # [N-1, 4, 4]
            end_poses = xpos_tensor[1:]  # [N-1, 4, 4]

            pos_dists = torch.norm(end_poses[:, :3, 3] - start_poses[:, :3, 3], dim=-1)
            pos_points = torch.clamp((pos_dists / step_size).int(), min=1)

            rel_rot = torch.matmul(
                start_poses[:, :3, :3].transpose(-1, -2), end_poses[:, :3, :3]
            )
            trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
            cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
            angles = torch.acos(cos_angle)
            rot_points = torch.clamp((angles / angle_step).int(), min=1)

            return torch.maximum(pos_points, rot_points).tolist()

        # Handle input arguments
        if qpos_list is not None:
            if not isinstance(qpos_list, torch.Tensor):
                qpos_list = np.asarray(qpos_list)
            qpos_tensor = torch.as_tensor(
                qpos_list, dtype=torch.float32, device=self.robot.device
            )
            if qpos_tensor.dim() == 1:
                qpos_tensor = qpos_tensor.unsqueeze(0)

            qpos_batch = qpos_tensor.unsqueeze(0)  # [n_env=1, n_batch=N, dof]
            xpos_batch = self.robot.compute_batch_fk(
                qpos=qpos_batch,
                name=self.uid,
                to_matrix=True,
            )
            xpos_list = xpos_batch.squeeze(0)
            qpos_list = qpos_tensor

        if xpos_list is None:
            logger.log_warning("Either xpos_list or qpos_list must be provided")
            empty_qpos = torch.empty((0, self.dof), dtype=torch.float32)
            empty_xpos = torch.empty((0, 4, 4), dtype=torch.float32)
            return empty_qpos, empty_xpos

        if not isinstance(xpos_list, torch.Tensor):
            xpos_list = torch.as_tensor(
                np.asarray(xpos_list),
                dtype=torch.float32,
                device=self.robot.device,
            )
        else:
            xpos_list = xpos_list.to(dtype=torch.float32, device=self.robot.device)

        # Get current position if needed
        if is_use_current_qpos:
            joint_ids = self.robot.get_joint_ids(self.uid)
            qpos_tensor = self.robot.get_qpos()
            # qpos_tensor shape: (batch, dof), usually batch=1
            current_qpos = qpos_tensor[0, joint_ids]

            current_xpos = self.robot.compute_fk(
                qpos=current_qpos, name=self.uid, to_matrix=True
            ).squeeze(0)

            if not isinstance(xpos_list, torch.Tensor):
                xpos_tensor = torch.as_tensor(
                    np.asarray(xpos_list),
                    dtype=torch.float32,
                    device=self.robot.device,
                )
            else:
                xpos_tensor = xpos_list.to(
                    dtype=torch.float32, device=self.robot.device
                )

            # Check if current position is significantly different from first waypoint
            pos_diff = torch.norm(current_xpos[:3, 3] - xpos_tensor[0, :3, 3]).item()
            rot_diff = torch.norm(current_xpos[:3, :3] - xpos_tensor[0, :3, :3]).item()

            if pos_diff > 0.001 or rot_diff > 0.01:
                xpos_list = torch.cat([current_xpos.unsqueeze(0), xpos_tensor], dim=0)
                if qpos_list is not None:
                    if not isinstance(qpos_list, torch.Tensor):
                        qpos_list = np.asarray(qpos_list)
                    qpos_tensor = torch.as_tensor(
                        qpos_list, dtype=torch.float32, device=self.robot.device
                    )
                    qpos_list = torch.cat(
                        [current_qpos.unsqueeze(0), qpos_tensor], dim=0
                    )
            else:
                xpos_list = xpos_tensor

        if qpos_seed is None and qpos_list is not None:
            qpos_seed = qpos_list[0]

        # Input validation
        if len(xpos_list) < 2:
            logger.log_warning("xpos_list must contain at least 2 points")
            empty_qpos = torch.empty((0, self.dof), dtype=torch.float32)
            empty_xpos = torch.empty((0, 4, 4), dtype=torch.float32)
            return empty_qpos, empty_xpos

        # Calculate point allocations for interpolation
        interpolated_point_allocations = calculate_point_allocations(
            xpos_list, step_size=0.002, angle_step=np.pi / 90
        )

        # Generate trajectory
        interpolate_qpos_list = []
        if is_linear or qpos_list is None:
            # Linear cartesian interpolation
            feasible_pose_targets = []
            for i in range(len(xpos_list) - 1):
                interpolated_poses = interpolate_xpos(
                    (
                        xpos_list[i].detach().cpu().numpy()
                        if isinstance(xpos_list, torch.Tensor)
                        else xpos_list[i]
                    ),
                    (
                        xpos_list[i + 1].detach().cpu().numpy()
                        if isinstance(xpos_list, torch.Tensor)
                        else xpos_list[i + 1]
                    ),
                    interpolated_point_allocations[i],
                )
                for xpos in interpolated_poses:
                    success, qpos = self.robot.compute_ik(
                        pose=xpos, joint_seed=qpos_seed, name=self.uid
                    )

                    if isinstance(success, torch.Tensor):
                        is_success = bool(success.all())
                    elif isinstance(success, np.ndarray):
                        is_success = bool(np.all(success))
                    elif isinstance(success, (list, tuple)):
                        is_success = all(success)
                    else:
                        is_success = bool(success)

                    if isinstance(qpos, torch.Tensor):
                        has_nan = torch.isnan(qpos).any().item()
                    else:
                        has_nan = np.isnan(qpos).any()

                    if not is_success or qpos is None or has_nan:
                        logger.log_debug(
                            f"IK failed or returned nan at pose, skipping this point."
                        )
                        continue

                    q_entry = qpos[0] if isinstance(qpos, (np.ndarray, list)) else qpos
                    if isinstance(q_entry, torch.Tensor) and q_entry.dim() > 1:
                        q_entry = q_entry.squeeze(0)
                    interpolate_qpos_list.append(q_entry)
                    feasible_pose_targets.append(xpos)
                    qpos_seed = q_entry

            # Vectorized FK feasibility check to keep only physically consistent IK outputs.
            if len(interpolate_qpos_list) > 0:
                qpos_tensor = torch.stack(
                    [
                        (
                            q.to(dtype=torch.float32, device=self.robot.device)
                            if isinstance(q, torch.Tensor)
                            else torch.as_tensor(
                                q, dtype=torch.float32, device=self.robot.device
                            )
                        )
                        for q in interpolate_qpos_list
                    ]
                )
                fk_batch = self.robot.compute_batch_fk(
                    qpos=qpos_tensor.unsqueeze(0),
                    name=self.uid,
                    to_matrix=True,
                ).squeeze(0)
                target_pose_tensor = torch.as_tensor(
                    np.asarray(feasible_pose_targets),
                    dtype=torch.float32,
                    device=self.robot.device,
                )
                pos_err = torch.norm(
                    fk_batch[:, :3, 3] - target_pose_tensor[:, :3, 3], dim=-1
                )
                rot_err = torch.norm(
                    fk_batch[:, :3, :3] - target_pose_tensor[:, :3, :3],
                    dim=(-2, -1),
                )
                valid_mask = (pos_err < 0.02) & (rot_err < 0.2)
                interpolate_qpos_list = [
                    q
                    for q, is_valid in zip(interpolate_qpos_list, valid_mask)
                    if bool(is_valid.item())
                ]
        else:
            # Joint space interpolation
            interpolate_qpos_list = [q for q in qpos_list]

        if len(interpolate_qpos_list) < 2:
            logger.log_error("Need at least 2 waypoints for trajectory planning")
            empty_qpos = torch.empty((0, self.dof), dtype=torch.float32)
            empty_xpos = torch.empty((0, 4, 4), dtype=torch.float32)
            return empty_qpos, empty_xpos

        # Create trajectory dictionary
        current_state = self._create_state_dict(interpolate_qpos_list[0])
        target_states = [
            self._create_state_dict(pos) for pos in interpolate_qpos_list[1:]
        ]

        init_plan_state = PlanState(
            move_type=MoveType.JOINT_MOVE,
            move_part=MovePart.ALL,
            qpos=current_state["position"],
            qvel=current_state["velocity"],
            qacc=current_state["acceleration"],
        )
        target_plan_states = []
        for state in target_states:
            plan_state = PlanState(
                move_type=MoveType.JOINT_MOVE, qpos=state["position"]
            )
            target_plan_states.append(plan_state)

        # Plan trajectory using internal plan method
        plan_result = self.plan(
            current_state=init_plan_state,
            target_states=target_plan_states,
            sample_method=sample_method,
            sample_interval=sample_num,
            **kwargs,
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
            qpos=out_qpos_tensor.unsqueeze_(0),
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
