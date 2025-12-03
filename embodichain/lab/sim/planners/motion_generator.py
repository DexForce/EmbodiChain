# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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
from typing import Dict, List, Tuple, Union, Optional, Any
from enum import Enum
from scipy.spatial.transform import Rotation, Slerp

from embodichain.lab.sim.planners.toppra_planner import ToppraPlanner
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod
from embodichain.lab.sim.objects.robot import Robot
from embodichain.utils import logger


class PlannerType(Enum):
    r"""Enumeration for different planner types."""

    TOPPRA = "toppra"
    """TOPPRA planner for time-optimal trajectory planning."""


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

    def __init__(
        self,
        robot: Robot,
        uid: str,
        sim=None,
        planner_type: Union[str, PlannerType] = "toppra",
        default_velocity: float = 0.2,
        default_acceleration: float = 0.5,
        collision_margin: float = 0.01,
        **kwargs,
    ):
        self.robot = robot
        self.sim = sim
        self.collision_margin = collision_margin
        self.uid = uid

        # Get robot DOF using get_joint_ids for specified control part (None for whole body)
        self.dof = len(robot.get_joint_ids(uid))

        # Create planner based on planner_type
        self.planner_type = self._parse_planner_type(planner_type)
        self.planner = self._create_planner(
            self.planner_type, default_velocity, default_acceleration, **kwargs
        )

    def _parse_planner_type(self, planner_type: Union[str, PlannerType]) -> str:
        r"""Parse planner type from string or enum.

        Args:
            planner_type: Planner type as string or PlannerType enum

        Returns:
            Planner type as string
        """
        if isinstance(planner_type, PlannerType):
            return planner_type.value
        elif isinstance(planner_type, str):
            planner_type_lower = planner_type.lower()
            # Validate planner type
            valid_types = [e.value for e in PlannerType]
            if planner_type_lower not in valid_types:
                logger.log_warning(
                    f"Unknown planner type '{planner_type}', using 'toppra'. "
                    f"Valid types: {valid_types}"
                )
                return "toppra"
            return planner_type_lower
        else:
            logger.log_error(
                f"planner_type must be str or PlannerType, got {type(planner_type)}",
                TypeError,
            )

    def _create_planner(
        self,
        planner_type: str,
        default_velocity: float,
        default_acceleration: float,
        **kwargs,
    ) -> Any:
        r"""Create planner instance based on planner type.

        Args:
            planner_type: Type of planner to create
            default_velocity: Default velocity limit
            default_acceleration: Default acceleration limit
            **kwargs: Additional arguments for planner initialization

        Returns:
            Planner instance
        """
        # Get constraints from robot or use defaults
        max_constraints = self._get_constraints(
            default_velocity, default_acceleration, **kwargs
        )

        if planner_type == "toppra":
            return ToppraPlanner(self.dof, max_constraints)
        else:
            logger.log_error(
                f"Unknown planner type '{planner_type}'. "
                f"Supported types: {[e.value for e in PlannerType]}",
                ValueError,
            )

    def _get_constraints(
        self, default_velocity: float, default_acceleration: float, **kwargs
    ) -> Dict[str, List[float]]:
        r"""Get velocity and acceleration constraints for the robot.

        Priority:
        1. kwargs['max_constraints'] if provided
        2. Robot's built-in constraints (if available)
        3. Default values

        Args:
            default_velocity: Default velocity limit
            default_acceleration: Default acceleration limit
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'velocity' and 'acceleration' constraints
        """
        # Check if constraints are provided in kwargs
        if "max_constraints" in kwargs and kwargs["max_constraints"] is not None:
            constraints = kwargs["max_constraints"]
            if isinstance(constraints, dict) and "velocity" in constraints:
                return constraints

        # Try to get constraints from robot (if available)
        # TODO: Add robot.get_joint_limits() or similar if available in future

        # Use default constraints
        return {
            "velocity": [default_velocity] * self.dof,
            "acceleration": [default_acceleration] * self.dof,
        }

    def _create_state_dict(
        self, position: np.ndarray, velocity: Optional[np.ndarray] = None
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
        current_state: Dict,
        target_states: List[Dict],
        sample_method: TrajectorySampleMethod = TrajectorySampleMethod.TIME,
        sample_interval: Union[float, int] = 0.01,
        **kwargs,
    ) -> Tuple[
        bool,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        float,
    ]:
        r"""Plan trajectory without collision checking.

        This method generates a smooth trajectory using the selected planner that satisfies
        velocity and acceleration constraints, but does not check for collisions.

        Args:
            current_state: Dictionary containing current state:
                - "position": Current joint positions (required)
                - "velocity": Current joint velocities (optional, defaults to zeros)
                - "acceleration": Current joint accelerations (optional, defaults to zeros)
            target_states: List of target state dictionaries, each with same format as current_state
            sample_method: Sampling method (TIME or QUANTITY)
            sample_interval: Sampling interval (time in seconds for TIME method, or number of points for QUANTITY)
            **kwargs: Additional arguments

        Returns:
            Tuple of (success, positions, velocities, accelerations, times, duration):
                - success: bool, whether planning succeeded
                - positions: np.ndarray (N, DOF), joint positions along trajectory
                - velocities: np.ndarray (N, DOF), joint velocities along trajectory
                - accelerations: np.ndarray (N, DOF), joint accelerations along trajectory
                - times: np.ndarray (N,), time stamps for each point
                - duration: float, total trajectory duration
        """
        # Validate inputs
        if len(current_state["position"]) != self.dof:
            logger.log_warning(
                f"Current state position dimension {len(current_state['position'])} "
                f"does not match robot DOF {self.dof}"
            )
            return False, None, None, None, None, 0.0

        for i, target in enumerate(target_states):
            if len(target["position"]) != self.dof:
                logger.log_warning(
                    f"Target state {i} position dimension {len(target['position'])} "
                    f"does not match robot DOF {self.dof}"
                )
                return False, None, None, None, None, 0.0

        # Plan trajectory using selected planner
        (
            success,
            positions,
            velocities,
            accelerations,
            times,
            duration,
        ) = self.planner.plan(
            current_state=current_state,
            target_states=target_states,
            sample_method=sample_method,
            sample_interval=sample_interval,
        )

        return success, positions, velocities, accelerations, times, duration

    def plan_with_collision(
        self,
        current_state: Dict,
        target_states: List[Dict],
        sample_method: TrajectorySampleMethod = TrajectorySampleMethod.TIME,
        sample_interval: Union[float, int] = 0.01,
        collision_check_interval: float = 0.01,
        **kwargs,
    ) -> None:
        r"""Plan trajectory with collision checking.

        TODO: This method is not yet implemented. It should:
        1. Generate a trajectory using the selected planner
        2. Check for collisions along the trajectory
        3. Return failure if collisions are detected
        """
        pass

    def create_discrete_trajectory(
        self,
        xpos_list: Optional[List[np.ndarray]] = None,
        qpos_list: Optional[List[np.ndarray]] = None,
        is_use_current_qpos: bool = True,
        is_linear: bool = False,
        sample_method: TrajectorySampleMethod = TrajectorySampleMethod.QUANTITY,
        sample_num: Union[float, int] = 20,
        qpos_seed: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

        # Plan trajectory using internal plan method
        success, positions, velocities, accelerations, times, duration = self.plan(
            current_state=current_state,
            target_states=target_states,
            sample_method=sample_method,
            sample_interval=sample_num,
            **kwargs,
        )

        if not success or positions is None:
            logger.log_error("Failed to plan trajectory")

        # Convert positions to list
        out_qpos_list = (
            positions.tolist() if isinstance(positions, np.ndarray) else positions
        )

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
