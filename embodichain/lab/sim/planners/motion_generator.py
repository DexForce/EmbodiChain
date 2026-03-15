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
        xpos_list: list[np.ndarray] | None = None,
        qpos_list: list[np.ndarray] | None = None,
        is_use_current_qpos: bool = True,
        is_linear: bool = False,
        sample_method: TrajectorySampleMethod = TrajectorySampleMethod.QUANTITY,
        sample_num: float | int = 20,
        qpos_seed: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        r"""Generate a discrete trajectory between waypoints using cartesian or joint space interpolation.

        This method supports two trajectory planning approaches:
        1. Linear interpolation: Fast, uniform spacing, no dynamics constraints
        2. Planner-based: Smooth, considers velocity/acceleration limits, realistic motion

        Args:
            xpos_list: List of waypoints as 4x4 transformation matrices (optional)
            qpos_list: List of joint configurations (optional)
            is_use_current_qpos: Whether to use current joint angles as starting point
            is_linear: If True, use cartesian linear interpolation, else joint space
            sample_method: Sampling method (QUANTITY or TIME)
            sample_num: Number of interpolated points for final trajectory
            qpos_seed: Initial joint configuration for IK solving
            **kwargs: Additional arguments

        Returns:
            A tuple containing:
            - List[np.ndarray]: Joint space trajectory as a list of joint configurations
            - List[np.ndarray]: Cartesian space trajectory as a list of 4x4 matrices
        """

        def interpolate_xpos(
            current_xpos: np.ndarray, target_xpos: np.ndarray, num_samples: int
        ) -> List[np.ndarray]:
            """Interpolate between two poses using Slerp for rotation and linear for translation."""
            if num_samples < 2:
                num_samples = 2

            slerp = Slerp(
                [0, 1],
                Rotation.from_matrix([current_xpos[:3, :3], target_xpos[:3, :3]]),
            )
            interpolated_poses = []
            for s in np.linspace(0, 1, num_samples):
                interp_rot = slerp(s).as_matrix()
                interp_trans = (1 - s) * current_xpos[:3, 3] + s * target_xpos[:3, 3]
                interp_pose = np.eye(4)
                interp_pose[:3, :3] = interp_rot
                interp_pose[:3, 3] = interp_trans
                interpolated_poses.append(interp_pose)
            return interpolated_poses

        def calculate_point_allocations(
            xpos_list: List[np.ndarray],
            step_size: float = 0.002,
            angle_step: float = np.pi / 90,
        ) -> List[int]:
            """Calculate number of interpolation points between each pair of waypoints."""
            point_allocations = []

            for i in range(len(xpos_list) - 1):
                start_pose = xpos_list[i]
                end_pose = xpos_list[i + 1]

                if isinstance(start_pose, torch.Tensor):
                    start_pose = start_pose.squeeze().cpu().numpy()
                if isinstance(end_pose, torch.Tensor):
                    end_pose = end_pose.squeeze().cpu().numpy()

                pos_dist = np.linalg.norm(end_pose[:3, 3] - start_pose[:3, 3])
                pos_points = max(1, int(pos_dist / step_size))

                angle_diff = Rotation.from_matrix(
                    start_pose[:3, :3].T @ end_pose[:3, :3]
                )
                angle = abs(angle_diff.as_rotvec()).max()
                rot_points = max(1, int(angle / angle_step))

                num_points = max(pos_points, rot_points)
                point_allocations.append(num_points)

            return point_allocations

        # Handle input arguments
        if qpos_list is not None:
            qpos_list = np.asarray(qpos_list)
            qpos_tensor = (
                torch.tensor(qpos_list)
                if not isinstance(qpos_list, torch.Tensor)
                else qpos_list
            )
            xpos_list = [
                self.robot.compute_fk(qpos=q, name=self.uid, to_matrix=True)
                .squeeze(0)
                .cpu()
                .numpy()
                for q in qpos_tensor
            ]

        if xpos_list is None:
            logger.log_warning("Either xpos_list or qpos_list must be provided")
            return [], []

        # Get current position if needed
        if is_use_current_qpos:
            joint_ids = self.robot.get_joint_ids(self.uid)
            qpos_tensor = self.robot.get_qpos()
            # qpos_tensor shape: (batch, dof), usually batch=1
            current_qpos = qpos_tensor[0, joint_ids]

            current_xpos = (
                self.robot.compute_fk(qpos=current_qpos, name=self.uid, to_matrix=True)
                .squeeze(0)
                .cpu()
                .numpy()
            )

            # Check if current position is significantly different from first waypoint
            pos_diff = np.linalg.norm(current_xpos[:3, 3] - xpos_list[0][:3, 3])
            rot_diff = np.linalg.norm(current_xpos[:3, :3] - xpos_list[0][:3, :3])

            if pos_diff > 0.001 or rot_diff > 0.01:
                xpos_list = np.concatenate(
                    [current_xpos[None, :, :], xpos_list], axis=0
                )
                if qpos_list is not None:
                    qpos_list = np.concatenate(
                        [current_qpos[None, :], qpos_list], axis=0
                    )

        if qpos_seed is None and qpos_list is not None:
            qpos_seed = qpos_list[0]

        # Input validation
        if len(xpos_list) < 2:
            logger.log_warning("xpos_list must contain at least 2 points")
            return [], []

        # Calculate point allocations for interpolation
        interpolated_point_allocations = calculate_point_allocations(
            xpos_list, step_size=0.002, angle_step=np.pi / 90
        )

        # Generate trajectory
        interpolate_qpos_list = []
        if is_linear or qpos_list is None:
            # Linear cartesian interpolation
            for i in range(len(xpos_list) - 1):
                interpolated_poses = interpolate_xpos(
                    xpos_list[i], xpos_list[i + 1], interpolated_point_allocations[i]
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

                    interpolate_qpos_list.append(
                        qpos[0] if isinstance(qpos, (np.ndarray, list)) else qpos
                    )
                    qpos_seed = (
                        qpos[0] if isinstance(qpos, (np.ndarray, list)) else qpos
                    )
        else:
            # Joint space interpolation
            interpolate_qpos_list = (
                qpos_list.tolist() if isinstance(qpos_list, np.ndarray) else qpos_list
            )

        if len(interpolate_qpos_list) < 2:
            logger.log_error("Need at least 2 waypoints for trajectory planning")

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

        # Convert positions to list
        out_qpos_list = plan_result.positions.to("cpu").numpy().tolist()
        out_qpos_list = (
            torch.tensor(out_qpos_list)
            if not isinstance(out_qpos_list, torch.Tensor)
            else out_qpos_list
        )
        out_xpos_list = [
            self.robot.compute_fk(qpos=q.unsqueeze(0), name=self.uid, to_matrix=True)
            .squeeze(0)
            .cpu()
            .numpy()
            for q in out_qpos_list
        ]

        return out_qpos_list, out_xpos_list

    def estimate_trajectory_sample_count(
        self,
        xpos_list: List[np.ndarray] = None,
        qpos_list: List[np.ndarray] = None,
        step_size: float = 0.01,
        angle_step: float = np.pi / 90,
        **kwargs,
    ) -> int:
        """Estimate the number of trajectory sampling points required.

        This function estimates the total number of sampling points needed to generate
        a trajectory based on the given waypoints and sampling parameters. It can be
        used to predict computational load and memory requirements before actual
        trajectory generation.

        Args:
            xpos_list: List of 4x4 transformation matrices representing waypoints
            qpos_list: List of joint positions (optional)
            is_linear: Whether to use linear interpolation
            step_size: Maximum allowed distance between consecutive points (in meters)
            angle_step: Maximum allowed angular difference between consecutive points (in radians)
            **kwargs: Additional parameters for further customization

        Returns:
            int: Estimated number of trajectory sampling points
        """

        def rotation_matrix_to_angle(self, rot_matrix: np.ndarray) -> float:
            cos_angle = (np.trace(rot_matrix) - 1) / 2
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.arccos(cos_angle)

        # Input validation
        if xpos_list is None and qpos_list is None:
            return 0

        # If joint position list is provided but end effector position list is not,
        # convert through forward kinematics
        if qpos_list is not None and xpos_list is None:
            if len(qpos_list) < 2:
                return 1 if len(qpos_list) == 1 else 1
            xpos_list = [
                self.robot.compute_fk(
                    qpos=torch.tensor(q, dtype=torch.float32),
                    name=self.uid,
                    to_matrix=True,
                )
                .squeeze(0)
                .cpu()
                .numpy()
                for q in qpos_list
            ]

        if xpos_list is None or len(xpos_list) == 0:
            return 1

        if len(xpos_list) == 1:
            return 1

        total_samples = 1  # Starting point

        total_pos_dist = 0.0
        total_angle = 0.0

        for i in range(len(xpos_list) - 1):
            start_pose = xpos_list[i]
            end_pose = xpos_list[i + 1]

            pos_diff = end_pose[:3, 3] - start_pose[:3, 3]
            total_pos_dist += np.linalg.norm(pos_diff)

            try:
                rot_matrix = start_pose[:3, :3].T @ end_pose[:3, :3]
                angle = rotation_matrix_to_angle(rot_matrix)
                total_angle += angle
            except Exception:
                pass

        pos_samples = max(1, int(total_pos_dist / step_size))
        rot_samples = max(1, int(total_angle / angle_step))

        total_samples = max(pos_samples, rot_samples)

        return max(2, total_samples)
