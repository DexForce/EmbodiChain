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

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import warp as wp

from embodichain.lab.sim.solvers import BaseSolver, SolverCfg
from embodichain.utils import configclass, logger
from embodichain.utils.device_utils import standardize_device_string
from embodichain.utils.warp.kinematics.srs_solver import (
    check_success_kernel,
    compute_arm_angle_kernel,
    compute_ik_kernel,
    nearest_ik_kernel,
    transform_pose_kernel,
)

if TYPE_CHECKING:
    pass


__all__ = ["SRSSolver", "SRSSolverCfg"]


@configclass
class SRSSolverCfg(SolverCfg):
    """Configuration for SRS inverse kinematics controller."""

    class_type: str = "SRSSolver"
    """Type of the solver class."""

    # kine_params: "W1ArmKineParams"
    # SRS-specific parameters
    dh_params = []
    """Denavit-Hartenberg parameters for the robot's kinematic chain."""

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

    search_mode: Literal["seeded", "full"] = "seeded"
    """Redundancy search strategy.

    ``"seeded"`` searches the seed arm angle first and then expands radially;
    ``"full"`` samples the complete ``[-pi, pi)`` interval.
    """

    redundancy_step: float = np.pi / 18.0
    """Angular step in radians for seed-centered redundancy search."""

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
        solver.set_tcp(self._get_tcp_as_numpy())

        return solver


class _BaseSRSSolverImpl:
    """Base implementation for the SRS inverse kinematics solver."""

    def __init__(self, cfg: SRSSolverCfg, device: torch.device):
        # Initialize configuration and device
        self.cfg = cfg
        self.device = device
        self.dofs = 7
        self.dh_params = cfg.dh_params
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
        self.link_lengths_np = np.asarray(self.cfg.link_lengths)
        self.rotation_directions_np = np.asarray(self.cfg.rotation_directions)

        if self.cfg.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if self.cfg.search_mode not in ("seeded", "full"):
            raise ValueError("search_mode must be 'seeded' or 'full'")
        if (
            not np.isfinite(self.cfg.redundancy_step)
            or self.cfg.redundancy_step <= 0
            or self.cfg.redundancy_step > np.pi
        ):
            raise ValueError("redundancy_step must be finite and in the range (0, pi]")

    def _sample_elbow_angles(
        self,
        qpos_seed: torch.Tensor,
        *,
        force_full: bool = False,
    ) -> torch.Tensor:
        """Build redundancy samples for every target.

        Seeded sampling follows a radial order ``0, +step, -step, ...`` so a
        nearby redundancy branch is evaluated before distant branches.
        """
        batch_size = qpos_seed.shape[0]
        if force_full or self.cfg.search_mode == "full":
            samples = torch.arange(
                self.cfg.num_samples,
                dtype=qpos_seed.dtype,
                device=self.device,
            )
            samples = -torch.pi + samples * (2.0 * torch.pi / self.cfg.num_samples)
            return samples.unsqueeze(0).expand(batch_size, -1).contiguous()

        offsets = [0.0]
        layer = 1
        while len(offsets) < self.cfg.num_samples:
            offset = layer * self.cfg.redundancy_step
            if offset > np.pi + 1e-12:
                break
            offsets.append(offset)
            # +pi and -pi normalize to the same angle, so keep only one.
            if offset < np.pi - 1e-12 and len(offsets) < self.cfg.num_samples:
                offsets.append(-offset)
            layer += 1
        offset_tensor = torch.tensor(offsets, dtype=qpos_seed.dtype, device=self.device)
        seed_arm_angles = self._get_seed_arm_angles(qpos_seed)
        angles = seed_arm_angles.unsqueeze(1) + offset_tensor.unsqueeze(0)
        return torch.remainder(angles + torch.pi, 2.0 * torch.pi) - torch.pi

    def _get_seed_arm_angles(self, qpos_seed: torch.Tensor) -> torch.Tensor:
        """Compute geometric arm angles for seed joint configurations."""
        raise NotImplementedError

    @staticmethod
    def _wrap_to_limits(
        joints: np.ndarray,
        limits: np.ndarray,
        seed: np.ndarray,
    ) -> np.ndarray | None:
        """Map revolute joints to equivalent values inside limits near the seed."""
        wrapped = np.empty_like(joints)
        two_pi = 2.0 * np.pi
        for index, value in enumerate(joints):
            lower, upper = limits[index]
            k_min = int(np.ceil((lower - value) / two_pi))
            k_max = int(np.floor((upper - value) / two_pi))
            if k_min > k_max:
                return None
            nearest_k = int(np.rint((seed[index] - value) / two_pi))
            nearest_k = min(max(nearest_k, k_min), k_max)
            wrapped[index] = value + nearest_k * two_pi
        return wrapped

    @staticmethod
    def _deduplicate_solutions(
        solutions: torch.Tensor, tolerance: float = 1e-5
    ) -> torch.Tensor:
        """Greedily retain periodic-unique representatives in input order."""
        if solutions.shape[0] < 2:
            return solutions

        cpu_solutions = solutions.detach().cpu().numpy()
        retained = np.empty_like(cpu_solutions)
        retained_indices = np.empty(cpu_solutions.shape[0], dtype=np.int64)
        retained_count = 0
        for index, candidate in enumerate(cpu_solutions):
            if retained_count:
                difference = candidate - retained[:retained_count]
                wrapped = np.remainder(difference + np.pi, 2.0 * np.pi) - np.pi
                if np.any(np.max(np.abs(wrapped), axis=1) <= tolerance):
                    continue
            retained[retained_count] = candidate
            retained_indices[retained_count] = index
            retained_count += 1

        index_tensor = torch.as_tensor(
            retained_indices[:retained_count],
            dtype=torch.long,
            device=solutions.device,
        )
        return solutions[index_tensor]


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

    def _get_model_fk(
        self, target_joint: np.ndarray, end_joint_index: int = 6
    ) -> np.ndarray:
        """Compute DH-model FK without base, end-effector, or TCP transforms."""
        pose = np.eye(4)
        for index in range(end_joint_index + 1):
            d, alpha, a, theta_offset = self.dh_params[index]
            theta = (
                theta_offset + target_joint[index] * self.rotation_directions_np[index]
            )
            pose = pose @ self._dh_transform(d, alpha, a, theta)
        return pose

    def _get_seed_arm_angles(self, qpos_seed: torch.Tensor) -> torch.Tensor:
        """Compute seed arm angles from shoulder-elbow-wrist geometry."""
        arm_angles = np.zeros(qpos_seed.shape[0], dtype=np.float32)
        for target_index, seed in enumerate(qpos_seed.detach().cpu().numpy()):
            full_pose = self._get_model_fk(seed)
            elbow_pose = self._get_model_fk(seed, end_joint_index=2)
            shoulder = np.array([0.0, 0.0, self.link_lengths_np[0]])
            wrist_offset = np.array([0.0, 0.0, self.dh_params_np[6, 0]])
            wrist = full_pose[:3, 3] - full_pose[:3, :3] @ wrist_offset
            shoulder_to_wrist = wrist - shoulder
            distance = np.linalg.norm(shoulder_to_wrist)
            if distance < 1e-10:
                continue

            elbow_model = (
                self.dh_params_np[3, 3] + seed[3] * self.rotation_directions_np[3]
            )
            elbow_config = -1.0 if elbow_model < 0.0 else 1.0
            _, _, reference_joints = self._compute_reference_plane(
                full_pose, elbow_config
            )
            if reference_joints is None:
                continue

            reference_pose = np.eye(4)
            for index in range(3):
                reference_pose = reference_pose @ self._dh_transform(
                    self.dh_params_np[index, 0],
                    self.dh_params_np[index, 1],
                    self.dh_params_np[index, 2],
                    reference_joints[index],
                )

            axis = shoulder_to_wrist / distance
            reference_upper = reference_pose[:3, 3] - shoulder
            actual_upper = elbow_pose[:3, 3] - shoulder
            reference_radial = reference_upper - axis * np.dot(reference_upper, axis)
            actual_radial = actual_upper - axis * np.dot(actual_upper, axis)
            reference_norm = np.linalg.norm(reference_radial)
            actual_norm = np.linalg.norm(actual_radial)
            if reference_norm < 1e-10 or actual_norm < 1e-10:
                continue

            reference_radial /= reference_norm
            actual_radial /= actual_norm
            arm_angles[target_index] = np.arctan2(
                np.dot(axis, np.cross(reference_radial, actual_radial)),
                np.dot(reference_radial, actual_radial),
            )
        return torch.from_numpy(arm_angles).to(self.device)

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

        joints[3] = elbow_config * np.arccos(np.clip(elbow_cos_angle, -1.0, 1.0))

        euclidean_norm = np.hypot(P26[0], P26[1])
        if euclidean_norm > 1e-6:
            joints[0] = np.arctan2(P26[1], P26[0])
        else:
            joints[0] = 0

        angle_phi_cos = (d_se**2 + norm_P26**2 - d_ew**2) / (2 * d_se * norm_P26)
        angle_phi = np.arccos(np.clip(angle_phi_cos, -1.0, 1.0))
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
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
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

        base_to_elbow_pose = np.eye(4)
        for i in range(3):
            T = self._dh_transform(
                dh_params[i, 0], dh_params[i, 1], dh_params[i, 2], joint_angles[i]
            )
            base_to_elbow_pose = base_to_elbow_pose @ T

        reference_upper = base_to_elbow_pose[:3, 3] - P02
        shoulder_to_wrist = P06 - P02
        upper_norm = np.linalg.norm(reference_upper)
        wrist_norm = np.linalg.norm(shoulder_to_wrist)
        if upper_norm < 1e-10 or wrist_norm < 1e-10:
            return None, None, None
        plane_normal = np.cross(
            reference_upper / upper_norm, shoulder_to_wrist / wrist_norm
        )
        plane_norm = np.linalg.norm(plane_normal)
        if plane_norm < 1e-10:
            return None, None, None
        plane_normal /= plane_norm

        return plane_normal, base_to_elbow_pose[:3, :3], joint_angles

    def _process_all_solutions(
        self,
        ik_qpos_tensor: torch.Tensor,
        qpos_seed: torch.Tensor,
        valid_mask: torch.Tensor,
        success_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            diff = ik_qpos_tensor - qpos_seed.unsqueeze(1)
            wrapped_diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            distances = torch.sum(
                wrapped_diff.square() * self.ik_nearest_weight_tensor, dim=2
            )
            distances[~valid_mask] = float("inf")
            sorted_indices = torch.argsort(distances, dim=1)
            sorted_ik_qpos_tensor = torch.gather(
                ik_qpos_tensor, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 7)
            )
            sorted_valid_mask = torch.gather(valid_mask, 1, sorted_indices)
            ik_qpos_tensor = sorted_ik_qpos_tensor
            valid_mask = sorted_valid_mask
        valid_qpos = [
            self._deduplicate_solutions(ik_qpos_tensor[index][valid_mask[index]])
            for index in range(ik_qpos_tensor.shape[0])
        ]
        max_solutions = max(solution.shape[0] for solution in valid_qpos)
        compact_qpos = torch.zeros(
            (ik_qpos_tensor.shape[0], max_solutions, 7),
            dtype=ik_qpos_tensor.dtype,
            device=self.device,
        )
        for index, solution in enumerate(valid_qpos):
            compact_qpos[index, : solution.shape[0]] = solution
        return success_tensor, compact_qpos

    def _process_single_solution(
        self,
        ik_qpos_tensor: torch.Tensor,
        qpos_seed: torch.Tensor,
        valid_mask: torch.Tensor,
        success_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            diff = ik_qpos_tensor - qpos_seed.unsqueeze(1)
            wrapped_diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            distances = torch.sum(
                wrapped_diff.square() * self.ik_nearest_weight_tensor, dim=2
            )
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
        self,
        target_pose: np.ndarray | torch.Tensor,
        nsparam: float,
        config: np.ndarray,
        qpos_seed: np.ndarray,
        prepared: (
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
        ) = None,
    ) -> tuple[bool, np.ndarray | None]:
        """
        Computes the inverse kinematics for a given target pose, normalization parameter, and configuration.

        Args:
            target_pose (np.ndarray | torch.Tensor): 4x4 target pose matrix.
            nsparam (float): Normalization parameter (angle).
            config (np.ndarray): Configuration index.

        Returns:
            bool: Success flag.
            np.ndarray: List of joint solutions (7) or None if no solution is found.
        """
        # Validate the target pose matrix
        if isinstance(target_pose, torch.Tensor):
            target_pose = target_pose.detach().cpu().numpy()
        else:
            target_pose = np.asarray(target_pose)
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

        if prepared is None:
            target_xpos = (
                self.T_b_ob_inv_np @ target_pose @ self.tcp_inv_np @ self.T_e_oe_inv_np
            )
            P_target = target_xpos[:3, 3]
            R_target = target_xpos[:3, :3]
            P02 = np.array([0, 0, link_lengths[0]])
            P67 = np.array([0, 0, dh_params[6, 0]])
            P26 = P_target - R_target @ P67 - P02
            _, R03_o, joints = self._compute_reference_plane(target_xpos, elbow_config)
            if R03_o is None or joints is None:
                return False, None
        else:
            target_xpos, R_target, P26, R03_o, joints = prepared

        # Calculate transformations
        T34 = self._dh_transform(
            dh_params[3, 0], dh_params[3, 1], dh_params[3, 2], joints[3]
        )
        R34 = T34[:3, :3]

        # Calculate shoulder joint rotation matrices
        usw = P26 / np.linalg.norm(P26)
        skew_usw = self._skew(usw)
        angle_psi = nsparam
        s_psi = np.sin(angle_psi)
        c_psi = np.cos(angle_psi)

        # Calculate rotation matrix R03
        A_s = skew_usw @ R03_o
        B_s = -skew_usw @ skew_usw @ R03_o
        C_s = (usw[:, None] @ usw[None, :]) @ R03_o
        R03 = A_s * s_psi + B_s * c_psi + C_s

        # Calculate shoulder joint angles
        angle1 = np.arctan2(R03[1, 1] * shoulder_config, R03[0, 1] * shoulder_config)
        angle2 = np.arccos(np.clip(R03[2, 1], -1.0, 1.0)) * shoulder_config
        angle3 = np.arctan2(-R03[2, 2] * shoulder_config, -R03[2, 0] * shoulder_config)

        # Calculate wrist joint angles
        A_w = R34.T @ A_s.T @ R_target
        B_w = R34.T @ B_s.T @ R_target
        C_w = R34.T @ C_s.T @ R_target
        R47 = A_w * s_psi + B_w * c_psi + C_w

        angle5 = np.arctan2(R47[1, 2] * wrist_config, R47[0, 2] * wrist_config)
        angle6 = np.arccos(np.clip(R47[2, 2], -1.0, 1.0)) * wrist_config
        angle7 = np.arctan2(R47[2, 1] * wrist_config, -R47[2, 0] * wrist_config)

        joints_output[0] = (angle1 - dh_params[0, 3]) * rotation_directions[0]
        joints_output[1] = (angle2 - dh_params[1, 3]) * rotation_directions[1]
        joints_output[2] = (angle3 - dh_params[2, 3]) * rotation_directions[2]
        joints_output[3] = (joints[3] - dh_params[3, 3]) * rotation_directions[3]
        joints_output[4] = (angle5 - dh_params[4, 3]) * rotation_directions[4]
        joints_output[5] = (angle6 - dh_params[5, 3]) * rotation_directions[5]
        joints_output[6] = (angle7 - dh_params[6, 3]) * rotation_directions[6]

        joints_output = self._wrap_to_limits(
            joints_output, self.qpos_limits_np, qpos_seed
        )
        if joints_output is None:
            return False, None

        return True, joints_output

    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        target_xpos = target_xpos.to(self.device, dtype=torch.float32).view(-1, 4, 4)
        num_targets = target_xpos.shape[0]
        # Validate and normalize qpos_seed
        if qpos_seed is None:
            qpos_seed = torch.zeros(
                (target_xpos.shape[0], 7), dtype=torch.float32, device=self.device
            )
        else:
            qpos_seed = (
                qpos_seed.to(self.device, dtype=torch.float32)
                .reshape(num_targets, -1, 7)[:, 0]
                .contiguous()
            )

        # Prepare to collect results
        elbow_angles = self._sample_elbow_angles(
            qpos_seed, force_full=return_all_solutions
        ).cpu()
        max_possible_solutions = elbow_angles.shape[1] * len(self.configs)
        all_solutions = np.zeros(
            (num_targets, max_possible_solutions, 7), dtype=np.float32
        )
        solution_counts = np.zeros(num_targets, dtype=np.int32)
        qpos_seed_np = qpos_seed.detach().cpu().numpy()
        target_xpos_np = target_xpos.detach().cpu().numpy()

        # Iterate over target poses
        for target_idx, xpos in enumerate(target_xpos):
            transformed = (
                self.T_b_ob_inv_np
                @ target_xpos_np[target_idx]
                @ self.tcp_inv_np
                @ self.T_e_oe_inv_np
            )
            rotation = transformed[:3, :3]
            shoulder = np.array([0.0, 0.0, self.link_lengths_np[0]])
            wrist_offset = np.array([0.0, 0.0, self.dh_params_np[6, 0]])
            shoulder_to_wrist = transformed[:3, 3] - rotation @ wrist_offset - shoulder
            prepared_by_elbow = {}
            for elbow_config in (1.0, -1.0):
                _, reference_rotation, reference_joints = self._compute_reference_plane(
                    transformed, elbow_config
                )
                if reference_rotation is not None and reference_joints is not None:
                    prepared_by_elbow[elbow_config] = (
                        transformed,
                        rotation,
                        shoulder_to_wrist,
                        reference_rotation,
                        reference_joints,
                    )
            sol_idx = 0
            for psi in elbow_angles[target_idx]:
                for config in self.configs:
                    prepared = prepared_by_elbow.get(config[1])
                    if prepared is None:
                        continue
                    success, qpos = self._get_each_ik(
                        xpos,
                        psi.item(),
                        config,
                        qpos_seed_np[target_idx],
                        prepared,
                    )
                    if success:
                        fk_xpos = self._get_fk(qpos)
                        target_np = xpos.detach().cpu().numpy()
                        if np.linalg.norm(fk_xpos - target_np) <= 1e-4:
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
                    (num_targets, 0 if return_all_solutions else 1, 7),
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

        valid_mask = torch.arange(max_solutions, device=self.device).unsqueeze(
            0
        ) < torch.from_numpy(solution_counts).to(self.device).unsqueeze(1)
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

        self.ik_nearest_weight_wp = wp.array(
            self.cfg.ik_nearest_weight,
            dtype=float,
            device=standardize_device_string(self.device),
        )
        self._temporary_workspace: dict[tuple[int, str], wp.array] = {}

    def _temporary_array(self, count: int, dtype: type, name: str) -> wp.array:
        """Return a zeroed reusable Warp scratch array."""
        key = (count, name)
        array = self._temporary_workspace.get(key)
        if array is None:
            array = wp.zeros(
                count,
                dtype=dtype,
                device=standardize_device_string(self.device),
            )
            self._temporary_workspace[key] = array
        else:
            array.zero_()
        return array

    def _get_seed_arm_angles(self, qpos_seed: torch.Tensor) -> torch.Tensor:
        """Compute seed arm angles with the Warp geometric implementation."""
        batch_size = qpos_seed.shape[0]
        arm_angles_wp = self._temporary_array(batch_size, float, "arm_angles")
        success_wp = self._temporary_array(batch_size, int, "arm_angle_success")
        wp.launch(
            kernel=compute_arm_angle_kernel,
            dim=batch_size,
            inputs=[
                wp.from_torch(qpos_seed.contiguous().flatten()),
                self.dh_params_wp,
                self.link_lengths_wp,
                self.rotation_directions_wp,
            ],
            outputs=[arm_angles_wp, success_wp],
            device=standardize_device_string(self.device),
        )
        arm_angles = wp.to_torch(arm_angles_wp)
        success = wp.to_torch(success_wp).bool()
        return torch.where(success, arm_angles, torch.zeros_like(arm_angles))

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
                self.ik_nearest_weight_wp,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        ik_solutions_tensor = wp.to_torch(qpos_out_wp).view(
            num_targets, num_per_target, 7
        )
        ik_valid_flags_tensor = (
            wp.to_torch(success_wp).view(num_targets, num_per_target).bool()
        )
        if self.cfg.sort_ik:
            diff = ik_solutions_tensor - qpos_seed.unsqueeze(1)
            wrapped_diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            weights = torch.as_tensor(
                self.cfg.ik_nearest_weight,
                dtype=ik_solutions_tensor.dtype,
                device=self.device,
            )
            distances = torch.sum(wrapped_diff.square() * weights, dim=-1)
            distances.masked_fill_(~ik_valid_flags_tensor, float("inf"))
            indices = torch.argsort(distances, dim=1)
            ik_solutions_tensor = torch.gather(
                ik_solutions_tensor, 1, indices.unsqueeze(-1).expand(-1, -1, 7)
            )
            ik_valid_flags_tensor = torch.gather(ik_valid_flags_tensor, 1, indices)

        success_flags = ik_valid_flags_tensor.any(dim=1)

        valid_qpos_list = [
            self._deduplicate_solutions(
                ik_solutions_tensor[i][ik_valid_flags_tensor[i]]
            )
            for i in range(num_targets)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        xpos_wp: wp.array,
        qpos_seed: torch.Tensor,
        qpos_out_wp: wp.array,
        success_wp: wp.array,
        num_combinations: int,
        num_configs: int,
        num_angles: int,
    ) -> None:
        """
        Compute IK solutions using the provided combinations.

        Args:
            xpos_wp: Transformed target poses.
            qpos_out_wp: Output array for joint positions.
            success_wp: Output array for success flags.
            num_combinations: Total number of combinations to process.
        """
        # Temporary arrays
        res_arm_angles = self._temporary_array(num_combinations, int, "res_arm_angles")
        joints_arm = self._temporary_array(num_combinations, wp.vec4, "joints_arm")
        res_plane_normal = self._temporary_array(
            num_combinations, int, "res_plane_normal"
        )
        plane_normal = self._temporary_array(num_combinations, wp.vec3, "plane_normal")
        base_to_elbow_rotation = self._temporary_array(
            num_combinations, wp.mat33, "base_to_elbow_rotation"
        )
        joints_plane = self._temporary_array(num_combinations, wp.vec4, "joints_plane")

        # Launch kernel to compute IK solutions
        wp.launch(
            kernel=compute_ik_kernel,
            dim=num_combinations,
            inputs=(
                xpos_wp,
                self.elbow_angles_wp,
                wp.from_torch(qpos_seed.contiguous().flatten()),
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
                num_configs,
                num_angles,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        target_xpos = target_xpos.to(self.device, dtype=torch.float32)
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
            qpos_seed = torch.zeros(
                (target_xpos.shape[0], 7),
                dtype=target_xpos.dtype,
                device=self.device,
            )
        else:
            qpos_seed = (
                qpos_seed.to(self.device, dtype=torch.float32)
                .reshape(target_xpos.shape[0], -1, 7)[:, 0]
                .contiguous()
            )

        # Prepare output arrays
        num_targets = target_xpos_wp.shape[0]
        num_configs = len(self.configs)
        elbow_angles = self._sample_elbow_angles(
            qpos_seed, force_full=return_all_solutions
        ).contiguous()
        num_angles = elbow_angles.shape[1]
        self.elbow_angles_wp = wp.from_torch(elbow_angles.flatten())
        # num_solutions = num_configs * num_angles
        num_combinations = num_targets * num_configs * num_angles

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
            xpos_wp,
            qpos_seed,
            qpos_out_wp,
            success_wp,
            num_combinations,
            num_configs,
            num_angles,
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
                    (num_targets, 0 if return_all_solutions else 1, 7),
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
            th=torch.zeros(7, dtype=torch.float32, device=self.device), end_only=False
        )
        root_tf = fk_dict[next(iter(fk_dict))]
        self.root_base_xpos = root_tf.get_matrix().cpu().numpy()

        # Initialize implementation based on device
        if self.device.type == "cuda":
            self.impl = _CUDASRSSolverImpl(cfg, self.device)
        else:
            self.impl = _CPUSRSSolverImpl(cfg, self.device)

        self._update_impl_qpos_limits()

    def _update_impl_qpos_limits(self):
        qpos_limits = torch.vstack([self.lower_qpos_limits, self.upper_qpos_limits]).T
        self.impl.qpos_limits_np = qpos_limits.cpu().numpy()
        self.impl.qpos_limits_wp = wp.array(
            self.impl.qpos_limits_np,
            dtype=wp.vec2,
            device=standardize_device_string(self.device),
        )

    def set_tcp(self, xpos: np.ndarray) -> None:
        """Set TCP and synchronize the analytical backend caches."""
        super().set_tcp(xpos)
        if hasattr(self, "impl"):
            self.impl.tcp_xpos = self.tcp_xpos.copy()
            self.impl.tcp_inv_np = np.linalg.inv(self.tcp_xpos)
            if isinstance(self.impl, _CUDASRSSolverImpl):
                self.impl.tcp_inv_wp = wp.mat44(*self.impl.tcp_inv_np.flatten())

    def set_ik_nearest_weight(
        self, ik_weight: np.ndarray, joint_ids: np.ndarray | None = None
    ) -> bool:
        """Set nearest-solution weights and synchronize backend caches."""
        success = super().set_ik_nearest_weight(ik_weight, joint_ids)
        if not success or not hasattr(self, "impl"):
            return success
        weights = torch.as_tensor(
            self.ik_nearest_weight, dtype=torch.float32, device=self.device
        )
        self.cfg.ik_nearest_weight = weights.detach().cpu().numpy().copy()
        if isinstance(self.impl, _CPUSRSSolverImpl):
            self.impl.ik_nearest_weight_tensor = weights
        else:
            self.impl.ik_nearest_weight_wp = wp.from_torch(weights.contiguous())
        return True

    def update_with_robot_limit(self, robot_qpos_limits):
        super().update_with_robot_limit(robot_qpos_limits)
        self._update_impl_qpos_limits()

    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor = None,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
