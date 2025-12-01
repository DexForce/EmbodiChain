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
import warp as wp
from itertools import product
from typing import Optional, Union, Tuple, Any, Literal, TYPE_CHECKING
from embodichain.utils import configclass, logger
from embodichain.lab.sim.solvers import SolverCfg, BaseSolver

from embodichain.utils.warp.kinematics.srs_solver import (
    transform_pose_kernel,
    compute_ik_kernel,
    sort_ik_kernel,
    nearest_ik_kernel,
    check_success_kernel,
)
from embodichain.utils.device_utils import standardize_device_string

if TYPE_CHECKING:
    from typing import Self
    from embodichain.lab.sim.robots.dexforce_w1.params import W1ArmKineParams


all = ["SRSSolver", "SRSSolverCfg"]


@configclass
class SRSSolverCfg(SolverCfg):
    """Configuration for SRS inverse kinematics controller."""

    class_type: str = "SRSSolver"
    """Type of the solver class."""

    # kine_params: "W1ArmKineParams"
    # SRS-specific parameters
    dh_params = []
    """Denavit-Hartenberg parameters for the robot's kinematic chain."""

    qpos_limits = []
    """Joint position limits for the robot."""

    T_b_ob = np.eye(4)
    """Base to observed base transform."""

    T_e_oe = np.eye(4)
    """End-effector to observed end-effector transform."""

    link_lengths = []
    """Link lengths of the robot arm."""

    rotation_directions = []
    """Rotation directions for each joint."""

    num_samples: int = 100
    """Number of samples for elbow angle during IK computation."""

    sort_ik: bool = True
    """Whether to sort IK solutions based on proximity to seed joint positions."""

    # TODO: Each target pose may have multiple IK solutions; weights can help select the best one.
    ik_nearest_weight: np.array = np.ones(7)
    """Weights for each joint when finding the nearest IK solution."""

    def init_solver(
        self, num_envs: int = 1, device: torch.device = torch.device("cpu"), **kwargs
    ) -> "SRSSolver":
        """Initialize the solver with the configuration.

        Args:
            device (torch.device): The device to use for the solver. Defaults to CPU.
            num_envs (int): The number of environments for which the solver is initialized.
            **kwargs: Additional keyword arguments that may be used for solver initialization.

        Returns:
            SRSSolver: An initialized solver instance.
        """

        solver = SRSSolver(cfg=self, num_envs=num_envs, device=device, **kwargs)

        # Set the Tool Center Point (TCP) for the solver
        if isinstance(self.tcp, torch.Tensor):
            tcp = self.tcp.cpu().numpy()
        else:
            tcp = self.tcp
        solver.set_tcp(tcp)

        return solver


class _BaseSRSSolverImpl:
    """Base implementation for the SRS inverse kinematics solver."""

    def __init__(self, cfg: SRSSolverCfg, device: torch.device):
        # Initialize configuration and device
        self.cfg = cfg
        self.device = device
        self.dofs = 7
        self.dh_params = cfg.dh_params
        self.qpos_limits = cfg.qpos_limits
        self.tcp_xpos = np.eye(4)

        # Initialize transformation matrices
        self._parse_params()

    def _parse_params(self):
        # Compute the inverse transformation matrices for TCP, end-effector, and base.
        self.tcp_xpos = self.cfg.tcp
        self.tcp_inv_np = np.linalg.inv(self.tcp_xpos)
        self.T_e_oe_inv_np = np.linalg.inv(self.cfg.T_e_oe)
        self.T_b_ob_inv_np = np.linalg.inv(self.cfg.T_b_ob)

        # Convert configuration parameters to numpy arrays for efficient computation.
        self.dh_params_np = np.asarray(self.cfg.dh_params)
        self.qpos_limits_np = np.asarray(self.cfg.qpos_limits)
        self.link_lengths_np = np.asarray(self.cfg.link_lengths)
        self.rotation_directions_np = np.asarray(self.cfg.rotation_directions)


class _CPUSRSSolverImpl(_BaseSRSSolverImpl):
    """CPU implementation of the SRS inverse kinematics solver."""

    def __init__(self, cfg: SRSSolverCfg, device: torch.device):
        super().__init__(cfg, device)

    def _parse_params(self):
        super()._parse_params()

        # Generate all possible configuration combinations for shoulder, elbow, and wrist.
        # Each configuration is represented by a vector of three elements, each being +1 or -1.
        # This covers all 8 possible sign combinations for the three joints.
        self.configs = [
            np.array([x, y, z]) for x, y, z in product([1.0, -1.0], repeat=3)
        ]

        # Generate a set of elbow angles sampled uniformly from -π to π.
        # The number of samples is determined by self.cfg.num_samples.
        # These angles are used for searching possible IK solutions.
        self.elbow_angles = torch.linspace(
            -torch.pi, torch.pi, self.cfg.num_samples, device=self.device
        )

        # Convert ik_nearest_weight to a tensor for efficient computation.
        self.ik_nearest_weight_tensor = torch.tensor(
            self.cfg.ik_nearest_weight, dtype=torch.float32, device=self.device
        )

    def _get_fk(self, target_joint: np.ndarray) -> np.ndarray:
        """
        Compute the forward kinematics (FK) for a given joint state.

        Args:
            target_joint (np.ndarray): Joint angles (shape: [7,]).

        Returns:
            np.ndarray: 4x4 transformation matrix representing the end-effector pose.
        """
        # Initialize pose as identity matrix
        pose = np.eye(4)

        # Iterate through the DH parameters and compute the transformation
        for i in range(self.dh_params.shape[0]):
            d = self.dh_params[i, 0]
            alpha = self.dh_params[i, 1]
            a = self.dh_params[i, 2]
            theta = self.dh_params[i, 3]

            # Add joint angle contribution if within bounds
            if i < target_joint.size:
                theta += target_joint[i] * self.cfg.rotation_directions[i]

            # Compute the transformation matrix for this joint
            T = self._dh_transform(d, alpha, a, theta)
            pose = pose @ T

        # Apply additional transformations: user frame, base, and tool center point (TCP)
        pose = (
            self.cfg.T_b_ob
            @ pose
            @ self.cfg.T_e_oe  # End-effector-to-observed-end-effector transform
            @ self.tcp_xpos  # Tool center point transform
        )

        return pose

    def _calculate_arm_joint_angles(
        self,
        P26: np.ndarray,
        elbow_config: int,
        joints: np.ndarray,
        link_lengths: np.ndarray,
    ) -> bool:
        """
        Calculate joint angles based on the position vector P26.

        Args:
            P26 (np.ndarray): Vector from shoulder to wrist.
            elbow_config (int): Elbow configuration (+1 or -1).
            joints (np.ndarray): Joint angles to be updated.
            link_lengths (np.ndarray): Link lengths of the robot.

        Returns:
            bool: True if successful, False otherwise.
        """
        d_bs, d_se, d_ew = link_lengths[:3]

        norm_P26 = np.linalg.norm(P26)
        if norm_P26 < np.abs(d_bs + d_ew):
            logger.log_warning("Specified pose outside reachable workspace.")
            return False

        elbow_cos_angle = (norm_P26**2 - d_se**2 - d_ew**2) / (2 * d_se * d_ew)
        if abs(elbow_cos_angle) > 1.0:
            logger.log_debug("Elbow singularity. End effector at limit.")
            return False

        joints[3] = elbow_config * np.arccos(elbow_cos_angle)

        if abs(P26[2]) > 1e-6:
            joints[0] = np.arctan2(P26[1], P26[0])
        else:
            joints[0] = 0

        euclidean_norm = np.hypot(P26[0], P26[1])
        angle_phi = np.arccos((d_se**2 + norm_P26**2 - d_ew**2) / (2 * d_se * norm_P26))
        joints[1] = np.arctan2(euclidean_norm, P26[2]) + elbow_config * angle_phi

        return True

    def _dh_transform(
        self, d: float, alpha: float, a: float, theta: float
    ) -> np.ndarray:
        """
        Compute the Denavit-Hartenberg transformation matrix.

        Args:
            d (float): Link offset.
            alpha (float): Link twist.
            a (float): Link length.
            theta (float): Joint angle.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)

        # fmt: off
        return np.array(
            [
                [cos_theta,  -sin_theta * cos_alpha, sin_theta * sin_alpha,  a * cos_theta],
                [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha,  a * sin_theta],
                [0,          sin_alpha,              cos_alpha,              d],
                [0, 0, 0, 1],
            ]
        )
        # fmt: on

    def _skew(self, vector: np.ndarray) -> np.ndarray:
        """
        Compute the skew-symmetric matrix of a vector.

        Args:
            vector (np.ndarray): Input vector (3,).

        Returns:
            np.ndarray: Skew-symmetric matrix (3x3).
        """
        return np.array(
            [
                [0, -vector[2], vector[1]],
                [vector[2], 0, -vector[0]],
                [-vector[1], vector[0], 0],
            ]
        )

    def _compute_reference_plane(
        self, target_pose: np.ndarray, elbow_config: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calculate the reference plane vector, rotation matrix, and joint values.

        Args:
            target_pose (np.ndarray): Transformed target pose (4x4).
            elbow_config (int): Elbow configuration (+1 or -1).

        Returns:
            tuple: (plane_normal, base_to_elbow_rotation, joint_angles) or (None, None, None) if failed.
        """
        dh_params = self.dh_params
        link_lengths = self.cfg.link_lengths

        P_target = target_pose[:3, 3]
        P02 = np.array([0, 0, link_lengths[0]])
        P67 = np.array([0, 0, dh_params[6, 0]])
        P06 = P_target - target_pose[:3, :3] @ P67
        P26 = P06 - P02

        joint_angles = np.zeros(7)
        if not self._calculate_arm_joint_angles(
            P26, elbow_config, joint_angles, link_lengths
        ):
            return None, None, None

        T34_v = self._dh_transform(
            dh_params[3, 0], dh_params[3, 1], dh_params[3, 2], joint_angles[3]
        )
        P34_v = T34_v[:3, 3]

        norm_P34_P02 = np.linalg.norm(P34_v - P02)
        if norm_P34_P02 > 1e-6:
            v1 = (P34_v - P02) / norm_P34_P02
        else:
            v1 = np.zeros_like(P34_v - P02)
        v2 = (P06 - P02) / np.linalg.norm(P06 - P02)
        plane_normal = np.cross(v1, v2)

        base_to_elbow_rotation = np.eye(3)
        for i in range(3):
            T = self._dh_transform(
                dh_params[i, 0], dh_params[i, 1], dh_params[i, 2], joint_angles[i]
            )
            base_to_elbow_rotation = base_to_elbow_rotation @ T[:3, :3]

        return plane_normal, base_to_elbow_rotation, joint_angles

    def _process_all_solutions(
        self,
        ik_qpos_tensor: torch.Tensor,
        qpos_seed: torch.Tensor,
        valid_mask: torch.Tensor,
        success_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns all valid IK solutions (optionally sorted).

        Args:
            ik_qpos_tensor (torch.Tensor): The IK joint position tensor.
            qpos_seed (torch.Tensor): The seed joint position tensor.
            valid_mask (torch.Tensor): The mask indicating valid solutions.
            success_tensor (torch.Tensor): The tensor indicating success of IK solutions.

        Returns:
            torch.Tensor: The success tensor.
            torch.Tensor: The IK solutions tensor (sorted if specified).
        """
        if self.cfg.sort_ik:
            weighted_diff = (
                ik_qpos_tensor - qpos_seed.unsqueeze(1)
            ) * self.ik_nearest_weight_tensor
            distances = torch.norm(weighted_diff, dim=2)
            distances[~valid_mask] = float("inf")
            sorted_indices = torch.argsort(distances, dim=1)
            sorted_ik_qpos_tensor = torch.gather(
                ik_qpos_tensor, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 7)
            )
            return success_tensor, sorted_ik_qpos_tensor
        else:
            return success_tensor, ik_qpos_tensor

    def _process_single_solution(
        self,
        ik_qpos_tensor: torch.Tensor,
        qpos_seed: torch.Tensor,
        valid_mask: torch.Tensor,
        success_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the nearest valid IK solution (optionally sorted).

        Args:
            ik_qpos_tensor (torch.Tensor): The IK joint position tensor.
            qpos_seed (torch.Tensor): The seed joint position tensor.
            valid_mask (torch.Tensor): The mask indicating valid solutions.
            success_tensor (torch.Tensor): The tensor indicating success of IK solutions.

        Returns:
            torch.Tensor: The success tensor.
            torch.Tensor: The nearest valid IK solution tensor.
        """
        num_targets = ik_qpos_tensor.shape[0]
        if self.cfg.sort_ik:
            weighted_diff = (
                ik_qpos_tensor - qpos_seed.unsqueeze(1)
            ) * self.ik_nearest_weight_tensor
            distances = torch.norm(weighted_diff, dim=2)
            mask = success_tensor.unsqueeze(1) & valid_mask
            distances[~mask] = float("inf")
            nearest_indices = torch.argmin(distances, dim=1)
            nearest_solutions = torch.zeros(
                (num_targets, 7), dtype=qpos_seed.dtype, device=self.device
            )
            has_solution = distances.min(dim=1).values != float("inf")
            if has_solution.any():
                nearest_solutions[has_solution] = ik_qpos_tensor[
                    torch.arange(num_targets)[has_solution],
                    nearest_indices[has_solution],
                ]
            return success_tensor, nearest_solutions.unsqueeze(1)
        else:
            # Return first solution only
            return success_tensor, ik_qpos_tensor[:, :1, :]

    def _get_each_ik(
        self, target_pose: np.ndarray, nsparam: float, config: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Computes the inverse kinematics for a given target pose, normalization parameter, and configuration.

        Args:
            target_pose (np.ndarray): 4x4 target pose matrix.
            nsparam (float): Normalization parameter (angle).
            config (np.ndarray): Configuration index.

        Returns:
            bool: Success flag.
            np.ndarray: List of joint solutions (7) or None if no solution is found.
        """
        # Validate the target pose matrix
        target_pose = np.array(target_pose)
        if target_pose.ndim == 3 and target_pose.shape[0] == 1:
            target_pose = target_pose[0]  # Extract the first matrix
        if target_pose.shape != (4, 4):
            logger.log_error(
                f"Invalid xpos shape: {target_pose.shape}, expected (4,4)."
            )
            return False, None

        shoulder_config, elbow_config, wrist_config = config[0], config[1], config[2]

        dof = self.dofs
        joints_output = np.zeros(dof)

        # Extract parameters
        dh_params = self.dh_params
        link_lengths = self.cfg.link_lengths
        rotation_directions = self.cfg.rotation_directions

        # Transform target pose
        target_xpos = (
            self.T_b_ob_inv_np @ target_pose @ self.tcp_inv_np @ self.T_e_oe_inv_np
        )
        P_target = target_xpos[:3, 3]
        R_target = target_xpos[:3, :3]
        P02 = np.array([0, 0, link_lengths[0]])  # Base to shoulder
        P67 = np.array([0, 0, dh_params[6, 0]])  # Hand to end-effector
        P06 = P_target - R_target @ P67
        P26 = P06 - P02

        # Calculate joint angles
        joints = np.zeros(dof)
        if not self._calculate_arm_joint_angles(
            P26, elbow_config, joints, link_lengths
        ):
            return False, None

        # Calculate transformations
        T34 = self._dh_transform(
            dh_params[3, 0], dh_params[3, 1], dh_params[3, 2], joints[3]
        )
        R34 = T34[:3, :3]

        # Calculate reference plane
        V_v_to_sew, R03_o, joint_v = self._compute_reference_plane(
            target_xpos, config[1]
        )
        if V_v_to_sew is None:
            return False, None

        # Calculate shoulder joint rotation matrices
        usw = P26 / np.linalg.norm(P26)
        skew_usw = self._skew(usw)
        angle_psi = nsparam
        s_psi = wp.sin(angle_psi)
        c_psi = wp.cos(angle_psi)

        # Calculate rotation matrix R03
        A_s = skew_usw @ R03_o
        B_s = -skew_usw @ skew_usw @ R03_o
        C_s = (usw[:, None] @ usw[None, :]) @ R03_o
        R03 = A_s * s_psi + B_s * c_psi + C_s

        # Calculate shoulder joint angles
        angle1 = np.arctan2(R03[1, 1] * shoulder_config, R03[0, 1] * shoulder_config)
        angle2 = np.arccos(R03[2, 1]) * shoulder_config
        angle3 = np.arctan2(-R03[2, 2] * shoulder_config, -R03[2, 0] * shoulder_config)

        # Calculate wrist joint angles
        A_w = R34.T @ A_s.T @ R_target
        B_w = R34.T @ B_s.T @ R_target
        C_w = R34.T @ C_s.T @ R_target
        R47 = A_w * s_psi + B_w * c_psi + C_w

        angle5 = np.arctan2(R47[1, 2] * wrist_config, R47[0, 2] * wrist_config)
        angle6 = np.arccos(R47[2, 2]) * wrist_config
        angle7 = np.arctan2(R47[2, 1] * wrist_config, -R47[2, 0] * wrist_config)

        joints_output[0] = (angle1 - dh_params[0, 3]) * rotation_directions[0]
        joints_output[1] = (angle2 - dh_params[1, 3]) * rotation_directions[1]
        joints_output[2] = (angle3 - dh_params[2, 3]) * rotation_directions[2]
        joints_output[3] = (joints[3] - dh_params[3, 3]) * rotation_directions[3]
        joints_output[4] = (angle5 - dh_params[4, 3]) * rotation_directions[4]
        joints_output[5] = (angle6 - dh_params[5, 3]) * rotation_directions[5]
        joints_output[6] = (angle7 - dh_params[6, 3]) * rotation_directions[6]

        return True, joints_output

    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute inverse kinematics (IK) for the given target pose using CPU.

        Args:
            target_xpos: Target end-effector pose (4x4).
            qpos_seed: Initial joint positions (rad).
            return_all_solutions: Whether to return all solutions. Default is False.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Success flag and joint positions.
        """
        num_targets = target_xpos.shape[0]
        # Validate and normalize qpos_seed
        if qpos_seed is None:
            qpos_seed = torch.zeros(
                (target_xpos.shape[0], 7), dtype=torch.float32, device=self.device
            )

        # Prepare to collect results
        max_possible_solutions = len(self.elbow_angles) * len(self.configs)
        all_solutions = np.zeros(
            (num_targets, max_possible_solutions, 7), dtype=np.float32
        )
        solution_counts = np.zeros(num_targets, dtype=np.int32)

        # Iterate over target poses
        for target_idx, xpos in enumerate(target_xpos):
            sol_idx = 0
            for psi in self.elbow_angles:
                for config in self.configs:
                    success, qpos = self._get_each_ik(xpos, psi.item(), config)
                    if success:
                        fk_xpos = self._get_fk(qpos)
                        if np.allclose(fk_xpos, xpos, atol=1e-4):
                            all_solutions[target_idx, sol_idx, :] = qpos
                            sol_idx += 1
            solution_counts[target_idx] = sol_idx

        has_solution = solution_counts > 0
        if not any(has_solution):
            logger.log_warning(
                f"Failed to calculate IK solutions.\n"
                f"Target pose: {target_xpos}\nSeed: {qpos_seed}"
            )
            return (
                torch.zeros(num_targets, dtype=torch.bool, device=self.device),
                torch.zeros(
                    (num_targets, num_targets, 7),
                    dtype=qpos_seed.dtype,
                    device=self.device,
                ),
            )
        max_solutions = solution_counts.max()

        # Convert results to tensors
        ik_qpos_tensor = torch.zeros(
            (num_targets, max_solutions, 7),
            dtype=qpos_seed.dtype,
            device=self.device,
        )
        for target_idx in range(num_targets):
            count = solution_counts[target_idx]
            if count > 0:
                ik_qpos_tensor[target_idx, :count] = torch.from_numpy(
                    all_solutions[target_idx, :count]
                ).to(self.device, dtype=qpos_seed.dtype)

        valid_mask = ik_qpos_tensor.abs().sum(dim=2) > 0  # (num_targets, max_solutions)
        success_tensor = torch.from_numpy(has_solution).to(self.device)
        if return_all_solutions:
            return self._process_all_solutions(
                ik_qpos_tensor, qpos_seed, valid_mask, success_tensor
            )
        else:
            return self._process_single_solution(
                ik_qpos_tensor, qpos_seed, valid_mask, success_tensor
            )


class _CUDASRSSolverImpl(_BaseSRSSolverImpl):
    """CUDA implementation of the SRS inverse kinematics solver."""

    def __init__(self, cfg: SRSSolverCfg, device: torch.device):
        super().__init__(cfg, device)

    def _parse_params(self):
        super()._parse_params()

        # Convert numpy transformation matrices to Warp mat44 format for CUDA computation.
        self.tcp_inv_wp = wp.mat44(*self.tcp_inv_np.flatten())
        self.T_b_ob_inv_wp = wp.mat44(*self.T_b_ob_inv_np.flatten())
        self.T_e_oe_inv_wp = wp.mat44(*self.T_e_oe_inv_np.flatten())

        # Convert DH parameters, joint limits, link lengths, and rotation directions to Warp arrays.
        self.dh_params_wp = wp.array(
            self.dh_params_np.flatten(),
            dtype=float,
            device=standardize_device_string(self.device),
        )
        self.qpos_limits_wp = wp.array(
            self.qpos_limits_np,
            dtype=wp.vec2,
            device=standardize_device_string(self.device),
        )
        self.link_lengths_wp = wp.array(
            self.link_lengths_np.flatten(),
            dtype=float,
            device=standardize_device_string(self.device),
        )
        self.rotation_directions_wp = wp.array(
            self.rotation_directions_np.flatten(),
            dtype=float,
            device=standardize_device_string(self.device),
        )

        # Generate all possible configuration combinations for shoulder, elbow, and wrist.
        # Each configuration is represented by a vector of three elements, each being +1 or -1.
        # This covers all 8 possible sign combinations for the three joints.
        self.configs = [wp.vec3(x, y, z) for x, y, z in product([1.0, -1.0], repeat=3)]
        self.configs_wp = wp.array(
            self.configs, dtype=wp.vec3, device=standardize_device_string(self.device)
        )

        # Generate a set of elbow angles sampled uniformly from -π to π.
        # The number of samples is determined by self.cfg.num_samples.
        # These angles are used for searching possible IK solutions.
        joint_reference_limits = [-wp.pi, wp.pi]
        self.elbow_angles = np.linspace(
            joint_reference_limits[0], joint_reference_limits[1], self.cfg.num_samples
        ).tolist()

        # Convert elbow angles to Warp array for CUDA computation.
        self.elbow_angles_wp = wp.array(
            self.elbow_angles,
            dtype=float,
            device=standardize_device_string(self.device),
        )

    def _sort_ik_solutions(
        self, qpos_out_wp, success_wp, qpos_seed, num_targets, num_configs, num_angles
    ):
        """
        Sort IK solutions based on weighted distance.

        Args:
            qpos_out_wp: Warp array of IK solutions (shape: [num_targets * num_configs * num_angles, 7]).
            success_wp: Warp array of validity flags (shape: [num_targets * num_configs * num_angles]).
            qpos_seed: Warp array of seed positions (shape: [num_targets, 7]).
            num_targets: Number of targets.
            num_configs: Number of configurations.
            num_angles: Number of angles.

        Returns:
            Tuple[wp.array, wp.array]: Sorted IK solutions and their validity flags.
        """
        N = num_targets
        N_SOL = num_configs * num_angles
        DOF = 7

        sorted_ik_solutions = wp.zeros(
            N * N_SOL * DOF, dtype=float, device=standardize_device_string(self.device)
        )
        sorted_ik_valid_flags = wp.zeros(
            N * N_SOL, dtype=int, device=standardize_device_string(self.device)
        )
        distances = wp.zeros(
            N * N_SOL, dtype=float, device=standardize_device_string(self.device)
        )
        indices = wp.zeros(
            N * N_SOL, dtype=int, device=standardize_device_string(self.device)
        )

        wp.launch(
            kernel=sort_ik_kernel,
            dim=num_targets,
            inputs=[
                qpos_out_wp,
                success_wp,
                qpos_seed,
                wp.array(
                    self.cfg.ik_nearest_weight,
                    dtype=float,
                    device=standardize_device_string(self.device),
                ),
                distances,
                indices,
                N_SOL,
            ],
            outputs=[
                sorted_ik_solutions,
                sorted_ik_valid_flags,
            ],
            device=standardize_device_string(self.device),
        )
        return sorted_ik_solutions, sorted_ik_valid_flags

    def _nearest_ik_solution(
        self, qpos_out_wp, success_wp, qpos_seed, num_targets, num_configs, num_angles
    ):
        """
        Find the nearest valid IK solution for each target pose.

        Selects the IK solution closest to the seed configuration among all valid solutions.

        Args:
            qpos_out_wp: IK solutions array of shape [num_targets * num_configs * num_angles, 7]
            success_wp: Validity flags array of shape [num_targets * num_configs * num_angles]
            qpos_seed: Seed configurations array of shape [num_targets, 7]
            num_targets: Number of target poses
            num_configs: Number of IK configurations
            num_angles: Number of sampling angles

        Returns:
            Tuple[wp.array, wp.array]:
                - Nearest IK solutions array of shape [num_targets, 7]
                - Validity flags array of shape [num_targets] indicating solution feasibility
        """
        N = num_targets
        N_SOL = num_configs * num_angles
        DOF = 7

        nearest_ik_solutions = wp.zeros(
            N * DOF, dtype=float, device=standardize_device_string(self.device)
        )
        nearest_ik_valid_flags = wp.zeros(
            N, dtype=int, device=standardize_device_string(self.device)
        )

        wp.launch(
            kernel=nearest_ik_kernel,
            dim=num_targets,
            inputs=[
                qpos_out_wp,
                success_wp,
                qpos_seed.flatten(),
                wp.array(
                    self.cfg.ik_nearest_weight,
                    dtype=float,
                    device=standardize_device_string(self.device),
                ),
                N_SOL,
            ],
            outputs=[
                nearest_ik_solutions,
                nearest_ik_valid_flags,
            ],
            device=standardize_device_string(self.device),
        )
        return nearest_ik_solutions, nearest_ik_valid_flags

    def _process_all_solutions(
        self,
        qpos_out_wp: wp.array,
        success_wp: wp.array,
        qpos_seed: wp.array,
        num_targets: int,
        num_configs: int,
        num_angles: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process and return all valid IK solutions.

        Args:
            qpos_out_wp: Warp array of IK solutions.
            success_wp: Warp array of success flags.
            qpos_seed: Seed joint positions.
            num_targets: Number of target poses.
            num_configs: Number of configurations.
            num_angles: Number of angles.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Success flags and all valid joint positions.
        """
        num_per_target = num_configs * num_angles

        if self.cfg.sort_ik:
            sorted_ik_solutions, sorted_ik_valid_flags = self._sort_ik_solutions(
                qpos_out_wp,
                success_wp,
                qpos_seed.flatten(),
                num_targets,
                num_configs,
                num_angles,
            )

            ik_solutions_tensor = wp.to_torch(sorted_ik_solutions).view(
                num_targets, num_per_target, 7
            )
            ik_valid_flags_tensor = (
                wp.to_torch(sorted_ik_valid_flags)
                .view(num_targets, num_per_target)
                .bool()
            )
        else:
            ik_solutions_tensor = wp.to_torch(qpos_out_wp).view(
                num_targets, num_per_target, 7
            )
            ik_valid_flags_tensor = (
                wp.to_torch(success_wp).view(num_targets, num_per_target).bool()
            )

        success_flags = ik_valid_flags_tensor.any(dim=1)

        valid_qpos_list = [
            ik_solutions_tensor[i][ik_valid_flags_tensor[i]] for i in range(num_targets)
        ]
        max_solutions = max(q.shape[0] for q in valid_qpos_list)
        valid_qpos_tensor = torch.zeros(
            (num_targets, max_solutions, 7),
            dtype=torch.float32,
            device=self.device,
        )
        for i, q in enumerate(valid_qpos_list):
            valid_qpos_tensor[i, : q.shape[0]] = q.to(self.device)

        return success_flags.to(self.device), valid_qpos_tensor

    def _process_single_solution(
        self,
        qpos_out_wp: wp.array,
        success_wp: wp.array,
        qpos_seed: wp.array,
        num_targets: int,
        num_configs: int,
        num_angles: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process and return the nearest valid IK solution for each target.

        Args:
            qpos_out_wp: Warp array of IK solutions.
            success_wp: Warp array of success flags.
            qpos_seed: Seed joint positions.
            num_targets: Number of target poses.
            num_configs: Number of configurations.
            num_angles: Number of angles.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Success flags and nearest valid joint positions.
        """
        num_per_target = num_configs * num_angles

        if self.cfg.sort_ik:
            nearest_ik_solutions, nearest_ik_valid_flags = self._nearest_ik_solution(
                qpos_out_wp,
                success_wp,
                qpos_seed,
                num_targets,
                num_configs,
                num_angles,
            )

            nearest_ik_solutions_tensor = wp.to_torch(nearest_ik_solutions).view(
                num_targets, 7
            )
            nearest_ik_valid_flags_tensor = (
                wp.to_torch(nearest_ik_valid_flags).view(num_targets).bool()
            )

            first_valid_qpos = torch.zeros(
                (num_targets, 1, 7), dtype=torch.float32, device=self.device
            )
            for i in range(num_targets):
                if nearest_ik_valid_flags_tensor[i]:
                    first_valid_qpos[i, 0] = nearest_ik_solutions_tensor[i].to(
                        self.device
                    )

            return nearest_ik_valid_flags_tensor.to(self.device), first_valid_qpos
        else:
            ik_solutions_tensor = wp.to_torch(qpos_out_wp).view(
                num_targets, num_per_target, 7
            )
            ik_valid_flags_tensor = (
                wp.to_torch(success_wp).view(num_targets, num_per_target).bool()
            )

            first_valid_qpos = torch.zeros(
                (num_targets, 1, 7), dtype=torch.float32, device=self.device
            )
            valid_flags = torch.zeros(num_targets, dtype=torch.bool, device=self.device)
            for i in range(num_targets):
                valid_indices = torch.where(ik_valid_flags_tensor[i])[0]
                if len(valid_indices) > 0:
                    first_valid_qpos[i, 0] = ik_solutions_tensor[
                        i, valid_indices[0]
                    ].to(self.device)
                    valid_flags[i] = True

            return valid_flags, first_valid_qpos

    def _check_success_flags(
        self,
        success_wp: wp.array,
        num_targets: int,
        num_configs: int,
        num_angles: int,
    ) -> torch.Tensor:
        """
        Check success flags for IK solutions.

        Args:
            success_wp: Warp array of success flags.
            num_targets: Number of target poses.
            num_configs: Number of configurations.
            num_angles: Number of angles.

        Returns:
            torch.Tensor: Success flags as a boolean tensor.
        """
        num_solutions = num_configs * num_angles
        success_flags_wp = wp.empty(
            num_targets, dtype=int, device=standardize_device_string(self.device)
        )
        wp.launch(
            kernel=check_success_kernel,
            dim=num_targets,
            inputs=[
                success_wp,
                num_solutions,
            ],
            outputs=[
                success_flags_wp,
            ],
            device=standardize_device_string(self.device),
        )
        return wp.to_torch(success_flags_wp).bool().to(self.device)

    def _compute_ik_solutions(
        self,
        combinations_wp: wp.array,
        xpos_wp: wp.array,
        qpos_out_wp: wp.array,
        success_wp: wp.array,
        num_combinations: int,
    ) -> None:
        """
        Compute IK solutions using the provided combinations.

        Args:
            combinations_wp: Warp array of combinations for parallel processing.
            xpos_wp: Transformed target poses.
            qpos_out_wp: Output array for joint positions.
            success_wp: Output array for success flags.
            num_combinations: Total number of combinations to process.
        """
        # Temporary arrays
        res_arm_angles = wp.zeros(
            num_combinations, dtype=int, device=standardize_device_string(self.device)
        )
        joints_arm = wp.zeros(
            num_combinations,
            dtype=wp.vec4,
            device=standardize_device_string(self.device),
        )
        res_plane_normal = wp.zeros(
            num_combinations, dtype=int, device=standardize_device_string(self.device)
        )
        plane_normal = wp.zeros(
            num_combinations,
            dtype=wp.vec3,
            device=standardize_device_string(self.device),
        )
        base_to_elbow_rotation = wp.zeros(
            num_combinations,
            dtype=wp.mat33,
            device=standardize_device_string(self.device),
        )
        joints_plane = wp.zeros(
            num_combinations,
            dtype=wp.vec4,
            device=standardize_device_string(self.device),
        )

        # Launch kernel to compute IK solutions
        wp.launch(
            kernel=compute_ik_kernel,
            dim=num_combinations,
            inputs=(
                combinations_wp,
                xpos_wp,
                self.elbow_angles_wp,
                self.qpos_limits_wp,
                self.configs_wp,
                self.dh_params_wp,
                self.link_lengths_wp,
                self.rotation_directions_wp,
                res_arm_angles,
                joints_arm,
                res_plane_normal,
                plane_normal,
                base_to_elbow_rotation,
                joints_plane,
            ),
            outputs=[success_wp, qpos_out_wp],
            device=standardize_device_string(self.device),
        )

    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute inverse kinematics (IK) for the given target pose.

        Args:
            target_xpos: Target end-effector pose (4x4).
            qpos_seed: Initial joint positions (rad).
            return_all_solutions: Whether to return all solutions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Success flag and joint positions.
        """
        # Prepare inputs
        target_xpos = target_xpos.to(self.device)
        target_xpos = target_xpos.view(-1, 4, 4)
        target_xpos_wp = wp.from_torch(target_xpos, dtype=wp.mat44)

        # transform pose
        xpos_wp = wp.zeros(
            target_xpos_wp.shape[0],
            dtype=wp.mat44,
            device=standardize_device_string(self.device),
        )
        wp.launch(
            kernel=transform_pose_kernel,
            dim=target_xpos_wp.shape[0],
            inputs=[
                target_xpos_wp,
                self.T_b_ob_inv_wp,
                self.T_e_oe_inv_wp,
                self.tcp_inv_wp,
            ],
            outputs=[xpos_wp],
            device=standardize_device_string(self.device),
        )

        # Define configurations and angles
        if qpos_seed is None:
            qpos_seed = wp.zeros(
                (target_xpos.shape[0], 7),
                dtype=float,
                device=standardize_device_string(self.device),
            )
            # TODO: Currently, full-space sampling is used to temporarily address situations
            # where joint space discontinuities or solution failures occur in different user scenarios.
            # Future plans include reducing the sampling space and adjusting the configuration.
            #
            # self.configs = [wp.vec3(*np.sign(qpos_seed[[1, 3, 5]].cpu().numpy()))]

        # Prepare output arrays
        num_targets = target_xpos_wp.shape[0]
        num_configs = len(self.configs)
        num_angles = len(self.elbow_angles)
        # num_solutions = num_configs * num_angles
        num_combinations = num_targets * num_configs * num_angles

        # Generate combinations for parallel processing
        combinations_np = np.stack(
            np.meshgrid(
                np.arange(num_targets),
                np.arange(num_configs),
                np.arange(num_angles),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        combinations_wp = wp.array(
            combinations_np,
            dtype=wp.vec3,
            device=standardize_device_string(self.device),
        )

        # Output arrays
        qpos_out_wp = wp.zeros(
            num_combinations * 7,
            dtype=float,
            device=standardize_device_string(self.device),
        )
        success_wp = wp.zeros(
            num_combinations, dtype=int, device=standardize_device_string(self.device)
        )

        # Compute IK solutions
        self._compute_ik_solutions(
            combinations_wp, xpos_wp, qpos_out_wp, success_wp, num_combinations
        )

        # Check for successful solutions
        success_flags_tensor = self._check_success_flags(
            success_wp, num_targets, num_configs, num_angles
        )

        if success_flags_tensor.any():
            if return_all_solutions:
                return self._process_all_solutions(
                    qpos_out_wp,
                    success_wp,
                    qpos_seed,
                    num_targets,
                    num_configs,
                    num_angles,
                )
            else:
                return self._process_single_solution(
                    qpos_out_wp,
                    success_wp,
                    qpos_seed,
                    num_targets,
                    num_configs,
                    num_angles,
                )
        else:
            return (
                torch.zeros(num_targets, dtype=torch.bool, device=self.device),
                torch.zeros(
                    (num_targets, num_targets, 7),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )


class SRSSolver(BaseSolver):
    r"""SRS inverse kinematics (IK) controller.

    This controller implements SRS inverse kinematics using various methods for
    computing the inverse of the Jacobian matrix.
    """

    def __init__(self, cfg: SRSSolverCfg, num_envs: int, device: str, **kwargs):
        r"""Initializes the SRS kinematics solver.

            This constructor sets up the kinematics solver using SRS methods,
            allowing for efficient computation of robot kinematics based on
            the specified URDF model.

        Args:
            cfg: The configuration for the solver.
            num_envs (int): The number of environments for the solver.
            device (str, optional): The device to use for the solver (e.g., "cpu" or "cuda").
            **kwargs: Additional keyword arguments passed to the base solver.

        """
        super().__init__(cfg=cfg, num_envs=num_envs, device=device, **kwargs)

        # Degrees of freedom
        self.dofs = 7

        # Tool Center Point (TCP) position
        self.tcp_xpos = np.eye(4)

        # Compute root base transform
        fk_dict = self.pk_serial_chain.forward_kinematics(
            th=np.zeros(7), end_only=False
        )
        root_tf = fk_dict[list(fk_dict.keys())[0]]
        self.root_base_xpos = root_tf.get_matrix().cpu().numpy()

        # Initialize implementation based on device
        if self.device.type == "cuda":
            self.impl = _CUDASRSSolverImpl(cfg, self.device)
        else:
            self.impl = _CPUSRSSolverImpl(cfg, self.device)

    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor = None,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute inverse kinematics (IK) for the given target pose.

        Args:
            target_xpos: Target end-effector pose (4x4).
            qpos_seed: Initial joint positions (rad). Default is None.
            return_all_solutions: Whether to return all solutions. Default is False.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Success flag and joint positions.
        """
        return self.impl.get_ik(
            target_xpos=target_xpos,
            qpos_seed=qpos_seed,
            return_all_solutions=return_all_solutions,
            **kwargs,
        )
