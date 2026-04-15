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
import warp as wp
import polars as pl

from itertools import product
from typing import Union, Tuple, Any, Literal, TYPE_CHECKING
from scipy.spatial.transform import Rotation

from embodichain.utils import configclass, logger
from embodichain.lab.sim.solvers import SolverCfg, BaseSolver
from embodichain.utils.warp.kinematics.opw_solver import (
    OPWparam,
    opw_fk_kernel,
    opw_ik_kernel,
    opw_ik_select_kernel,
    wp_vec6f,
)
from embodichain.utils.device_utils import standardize_device_string


if TYPE_CHECKING:
    from typing import Self


def normalize_to_pi(angle):
    angle = (angle + np.pi) % (2.0 * np.pi) - np.pi
    return angle


@configclass
class OPWSolverCfg(SolverCfg):
    """Configuration for OPW inverse kinematics controller."""

    class_type: str = "OPWSolver"

    # OPW-specific parameters
    a1 = 0.0
    a2 = -21.984
    b = 0.0
    c1 = 123.0
    c2 = 285.03
    c3 = 250.75
    c4 = 91.0
    offsets = (
        0.0,
        82.21350356417211 * np.pi / 180.0,
        -167.21710113148163 * np.pi / 180.0,
        0.0,
        0.0,
        0.0,
    )
    flip_axes = (False, False, False, False, False, False)
    has_parallelogram = False

    # Parameters for the inverse-kinematics method.
    ik_params: dict | None = None

    # safe margin for joint limits, in radians
    safe_margin: float = 5.0 * np.pi / 180.0

    def init_solver(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> "OPWSolver":
        """Initialize the solver with the configuration.

        Args:
            device (torch.device): The device to use for the solver. Defaults to CPU.
            n_sample (int): The number of environments for which the solver is initialized.
            **kwargs: Additional keyword arguments that may be used for solver initialization.

        Returns:
            OPWSolver: An initialized solver instance.
        """

        solver = OPWSolver(cfg=self, device=device, **kwargs)

        # Set the Tool Center Point (TCP) for the solver
        solver.set_tcp(self._get_tcp_as_numpy())

        return solver


class OPWSolver(BaseSolver):
    r"""OPW inverse kinematics (IK) controller.

    This controller implements OPW inverse kinematics using various methods for
    computing the inverse of the Jacobian matrix.
    """

    def __init__(self, cfg: OPWSolverCfg, device: str = "cpu", **kwargs):
        r"""Initializes the OPW kinematics solver.

            This constructor sets up the kinematics solver using OPW methods,
            allowing for efficient computation of robot kinematics based on
            the specified URDF model.

        Args:
            cfg: The configuration for the solver.
            device (str, optional): The device to use for the solver (e.g., "cpu" or "cuda"). Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the base solver.

        """
        super().__init__(cfg=cfg, device=device, **kwargs)
        # Note: the warp-based solver is currently the only active backend and
        # is initialized regardless of the selected device.
        self._init_warp_solver(cfg, **kwargs)
        self.set_tcp(np.eye(4))

    def set_tcp(self, xpos: np.ndarray):
        super().set_tcp(xpos)
        self._tcp_warp = wp.mat44f(self.tcp_xpos)
        tcp_inv = np.eye(4, dtype=float)
        tcp_inv[:3, :3] = self.tcp_xpos[:3, :3].T
        tcp_inv[:3, 3] = -tcp_inv[:3, :3].T @ self.tcp_xpos[:3, 3]
        self._tcp_inv_warp = wp.mat44f(tcp_inv)

    def _init_warp_solver(self, cfg: OPWSolverCfg, **kwargs):
        self.params = OPWparam()
        # convert unit from mm to m, increate precision
        self.params.a1 = cfg.a1 / 1000.0
        self.params.a2 = cfg.a2 / 1000.0
        self.params.b = cfg.b / 1000.0
        self.params.c1 = cfg.c1 / 1000.0
        self.params.c2 = cfg.c2 / 1000.0
        self.params.c3 = cfg.c3 / 1000.0
        self.params.c4 = cfg.c4 / 1000.0
        self.offsets = wp.array(
            cfg.offsets, dtype=float, device=standardize_device_string(self.device)
        )
        self.sign_corrections = wp.array(
            [-1.0 if flip else 1.0 for flip in cfg.flip_axes],
            dtype=float,
            device=standardize_device_string(self.device),
        )

    def get_fk(self, qpos: torch.tensor, **kwargs) -> torch.tensor:
        r"""
        Computes the forward kinematics for the end-effector link.

        Args:
            qpos (torch.Tensor): Joint positions. Can be a single configuration (dof,) or a batch (batch_size, dof).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            torch.Tensor: The homogeneous transformation matrix of the end link with TCP applied.
                        Shape is (4, 4) for single input, or (batch_size, 4, 4) for batch input.
        """
        if standardize_device_string(self.device) == "cpu":
            return super().get_fk(qpos, **kwargs)
        else:
            return self.get_fk_warp(qpos, **kwargs)

    def get_fk_warp(self, qpos: torch.tensor, **kwargs) -> torch.tensor:
        r"""
        Computes the forward kinematics for the end-effector link.

        Args:
            qpos (torch.Tensor): Joint positions. Can be a single configuration (dof,) or a batch (batch_size, dof).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            torch.Tensor: The homogeneous transformation matrix of the end link with TCP applied.
                        Shape is (4, 4) for single input, or (batch_size, 4, 4) for batch input.
        """
        if qpos.shape == (6,):
            qpos_batch = qpos[None, :]
        else:
            qpos_batch = qpos
        n_sample = qpos_batch.shape[0]
        qpos_wp = wp.from_torch(qpos_batch.reshape(-1))  # dtype=float, device="cuda")
        # qpos_wp = wp.array(qpos_batch.detach().cpu().numpy().reshape(-1), dtype=float, device=self.device)
        xpos_wp = wp.zeros(
            n_sample * 16, dtype=float, device=standardize_device_string(self.device)
        )
        wp.launch(
            kernel=opw_fk_kernel,
            dim=(n_sample),
            inputs=[
                qpos_wp,
                self._tcp_warp,
                self.params,
                self.offsets,
                self.sign_corrections,
            ],
            outputs=[xpos_wp],
            device=standardize_device_string(self.device),
        )
        xpos = wp.to_torch(xpos_wp).reshape(n_sample, 4, 4)
        return xpos

    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor = None,
        return_all_solutions: bool = False,
        **kwargs,
    ):
        """Compute target joint positions using OPW inverse kinematics.

        Args:
            target_xpos (torch.Tensor): Current end-effector pose, shape (n_sample, 4, 4).
            qpos_seed (torch.Tensor): Current joint positions, shape (n_sample, num_joints). Defaults to None.
            return_all_solutions (bool, optional): Whether to return all IK solutions or just the best one. Defaults to False.
            **kwargs: Additional keyword arguments for future extensions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - target_joints (torch.Tensor): Computed target joint positions, shape (n_sample, num_joints).
                - success (torch.Tensor): Boolean tensor indicating IK solution validity for each environment, shape (n_sample,).
        """
        return self.get_ik_warp(target_xpos, qpos_seed, return_all_solutions, **kwargs)

    def get_ik_warp(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor,
        return_all_solutions: bool = False,
        **kwargs,
    ):
        """Compute target joint positions using OPW inverse kinematics.

        Args:
            target_xpos (torch.Tensor): Current end-effector pose, shape (n_sample, 4, 4).
            qpos_seed (torch.Tensor): Current joint positions, shape (n_sample, num_joints).
            return_all_solutions (bool, optional): Whether to return all IK solutions or just the best one. Defaults to False.
            **kwargs: Additional keyword arguments for future extensions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - target_joints (torch.Tensor): Computed target joint positions, shape (n_sample, n_solution, num_joints).
                - success (torch.Tensor): Boolean tensor indicating IK solution validity for each environment, shape (n_sample,).
        """
        N_SOL = 8
        DOF = 6
        n_sample = target_xpos.shape[0]
        kernel_device = standardize_device_string(self.device)

        if target_xpos.shape == (4, 4):
            target_xpos_batch = target_xpos[None, :, :].to(kernel_device)
        else:
            target_xpos_batch = target_xpos.to(kernel_device)
        target_xpos_wp = wp.from_torch(target_xpos_batch.reshape(-1))

        all_qpos_wp = wp.zeros(
            n_sample * N_SOL * DOF,
            dtype=float,
            device=standardize_device_string(kernel_device),
        )
        all_ik_valid_wp = wp.zeros(
            n_sample * N_SOL, dtype=int, device=standardize_device_string(kernel_device)
        )

        # TODO: whether require gradient
        offsets_ = self.offsets.to(standardize_device_string(kernel_device))
        sign_corrections_ = self.sign_corrections.to(
            standardize_device_string(kernel_device)
        )
        lower_limits_ = wp_vec6f(
            self.lower_qpos_limits[0],
            self.lower_qpos_limits[1],
            self.lower_qpos_limits[2],
            self.lower_qpos_limits[3],
            self.lower_qpos_limits[4],
            self.lower_qpos_limits[5],
        )
        upper_limits_ = wp_vec6f(
            self.upper_qpos_limits[0],
            self.upper_qpos_limits[1],
            self.upper_qpos_limits[2],
            self.upper_qpos_limits[3],
            self.upper_qpos_limits[4],
            self.upper_qpos_limits[5],
        )
        wp.launch(
            kernel=opw_ik_kernel,
            dim=(n_sample),
            inputs=(
                target_xpos_wp,
                self._tcp_inv_warp,
                self.params,
                offsets_,
                sign_corrections_,
                lower_limits_,
                upper_limits_,
                self.cfg.safe_margin,
            ),
            outputs=[all_qpos_wp, all_ik_valid_wp],
            device=standardize_device_string(kernel_device),
        )

        if return_all_solutions:
            all_qpos = wp.to_torch(all_qpos_wp).reshape(n_sample, N_SOL, DOF)
            all_ik_valid = wp.to_torch(all_ik_valid_wp).reshape(n_sample, N_SOL)
            return all_ik_valid, all_qpos
        if qpos_seed is not None:
            if qpos_seed.shape == (
                n_sample,
                DOF,
            ):
                qpos_seed_ = qpos_seed.to(kernel_device)
            elif qpos_seed.shape == (DOF,):
                qpos_seed_ = (
                    qpos_seed.unsqueeze(0).repeat(n_sample, 1).to(kernel_device)
                )
            else:
                logger.log_error(
                    f"Invalid shape for qpos_seed: {qpos_seed.shape}. Expected ({n_sample}, {DOF}) or ({DOF},)."
                )
            qpos_seed_wp = wp.from_torch(qpos_seed_)
        else:
            qpos_seed = torch.zeros(
                (n_sample, DOF), dtype=torch.float32, device=kernel_device
            )
            qpos_seed_wp = wp.from_torch(qpos_seed)
        all_qpos_wp = all_qpos_wp.reshape((n_sample, N_SOL, DOF))
        all_ik_valid_wp = all_ik_valid_wp.reshape((n_sample, N_SOL))
        joint_weight = kwargs.get("joint_weight", torch.ones(size=(DOF,), dtype=float))
        joint_weight_wp = wp_vec6f(
            joint_weight[0],
            joint_weight[1],
            joint_weight[2],
            joint_weight[3],
            joint_weight[4],
            joint_weight[5],
        )
        best_ik_result_wp = wp.zeros(
            (n_sample, 6), dtype=float, device=standardize_device_string(kernel_device)
        )
        best_ik_valid_wp = wp.zeros(
            n_sample, dtype=int, device=standardize_device_string(kernel_device)
        )
        wp.launch(
            kernel=opw_ik_select_kernel,
            dim=(n_sample),
            inputs=[
                all_qpos_wp,
                all_ik_valid_wp,
                qpos_seed_wp,
                joint_weight_wp,
            ],
            outputs=[
                best_ik_result_wp,
                best_ik_valid_wp,
            ],
            device=standardize_device_string(kernel_device),
        )

        best_ik_result = (
            wp.to_torch(best_ik_result_wp).reshape(n_sample, 1, 6).to(self.device)
        )
        best_ik_valid = wp.to_torch(best_ik_valid_wp).to(self.device)
        return best_ik_valid, best_ik_result

    def _calculate_dynamic_weights(
        self, current_joints, joint_limits, base_weights=None
    ) -> np.ndarray:
        r"""Calculate dynamic joint weights based on proximity to joint limits.

        This function increases the weight of joints that are close to their limits, making the IK solver
        penalize solutions that move joints near their boundaries. The closer a joint is to its limit,
        the higher its weight will be, encouraging safer joint configurations.

        Args:
            current_joints (np.ndarray): Current joint positions, shape (6,).
            joint_limits (list or np.ndarray): List of (min, max) tuples for each joint, shape (6, 2).
            base_weights (np.ndarray, optional): Base weights for each joint, shape (6,). Defaults to ones.

        Returns:
            np.ndarray: Dynamic weights for each joint, shape (6,).
        """
        if base_weights is None:
            base_weights = np.ones(6)

        dynamic_weights = np.copy(base_weights)
        for i in range(6):
            cj = current_joints[i]
            if isinstance(cj, np.ndarray):
                if cj.size == 1:
                    cj = float(cj)
                else:
                    cj = float(cj.flat[0])
            jl_min = joint_limits[i][0]
            jl_max = joint_limits[i][1]
            range_size = jl_max - jl_min
            distance_to_min = cj - jl_min
            distance_to_max = jl_max - cj

            min_ratio = distance_to_min / range_size
            max_ratio = distance_to_max / range_size
            danger_ratio = min(float(min_ratio), float(max_ratio))
            if danger_ratio < 0.2:
                dynamic_weights[i] *= 5.0
            elif danger_ratio < 0.4:
                dynamic_weights[i] *= 2.0

        return dynamic_weights

    def _select_optimal_solution(
        self,
        current_joints,
        all_solutions,
        joint_limits=None,
        weights=None,
        prev_joints=None,
        return_all_valid=False,
    ) -> Tuple[bool, np.ndarray]:
        r"""Select the optimal IK solution based on joint limits and weighted differences.

        Args:
            current_joints (np.ndarray): Current joint positions in radians, shape=(6,)
            all_solutions (List[np.ndarray]): List of all possible IK solutions, each solution has shape=(6,)
            joint_limits (List[Tuple], optional): Joint limits list [(min1,max1),...,(min6,max6)]. Defaults to None.
            weights (np.ndarray, optional): Weight coefficients for each joint, shape=(6,). Defaults to None.
            prev_joints (np.ndarray, optional): Previous joint positions in radians, shape=(6,). Defaults to None.
            return_all_valid (bool, optional): If True, return all valid solutions instead of just the optimal one. Defaults to False.

        Returns:
            Tuple[bool, np.ndarray]: A tuple containing:
                - Success flag (True if solution found)
                - Joint angles of the optimal solution (single solution) or all valid solutions (if return_all_valid=True)
        """
        # Input validation
        if current_joints is None or all_solutions is None:
            return False, np.array([])

        # Convert inputs to numpy arrays
        current_joints = np.asarray(current_joints).reshape(-1)
        all_solutions = [(np.asarray(sol)) for sol in all_solutions]

        # Set default joint limits if none provided
        if joint_limits is None:
            joint_limits = [
                (-np.pi, np.pi),  # joint 1
                (-np.pi, np.pi),  # joint 2
                (-np.pi, np.pi),  # joint 3
                (-np.pi, np.pi),  # joint 4
                (-np.pi, np.pi),  # joint 5
                (-np.pi, np.pi),  # joint 6
            ]

        # TODO: support funciton to setting safty margin
        # SAFETY_MARGIN = np.radians(5.0)
        # joint_limits = [
        #     (-2.618 + SAFETY_MARGIN, 2.618 - SAFETY_MARGIN),  # 约(-145.5°+5°, 124.2°-5°)
        #     (0.0 + SAFETY_MARGIN, 3.14 - SAFETY_MARGIN),  # 约(0°+5°, 180°-5°)
        #     (-2.967 + SAFETY_MARGIN, 0.0 - SAFETY_MARGIN),  # 约(-170°+5°, 0°-5°)
        #     (-1.745 + SAFETY_MARGIN, 1.745 - SAFETY_MARGIN),  # 约(-100°+5°, 100°-5°)
        #     (-1.22 + SAFETY_MARGIN, 1.22 - SAFETY_MARGIN),  # 约(-70°+5°, 70°-5°)
        #     (-2.0944 + SAFETY_MARGIN, 2.0944 - SAFETY_MARGIN),  # 约(-120°+5°, 120°-5°)
        # ]

        # Handle empty solution case
        if len(all_solutions) == 0:
            logger.log_warning("No available solutions found.")
            return None, np.array([])

        # Set default weights if none provided
        if weights is None:
            weights = np.ones(6)
        else:
            weights = np.asarray(weights)

        # Initialize previous joints if not provided
        if prev_joints is None:
            prev_joints = current_joints
        else:
            prev_joints = np.asarray(prev_joints)

        # Ensure we only work with first 6 joints
        current_joints = current_joints[:6]
        prev_joints = prev_joints[:6]

        # Calculate dynamic weights considering joint limits
        dynamic_weights = self._calculate_dynamic_weights(
            current_joints, joint_limits, weights
        )

        # Initialize variables for tracking best solution and all valid solutions with scores
        best_score = float("inf")
        best_qpos = None
        all_valid_solutions = []  # List of (solution, score) tuples for sorting

        # Evaluate each IK solution
        for q in all_solutions:
            possible_arrays = []
            valid_solution = True

            # Generate possible joint values considering 2π periodicity
            for i in range(6):
                current_possible_values = []
                # Determine previous movement direction
                prev_move = current_joints[i] - prev_joints[i]

                # Prefer offsets in the same direction as previous movement
                preferred_offsets = range(0, 3) if prev_move >= 0 else range(-2, 1)
                for offset in preferred_offsets:
                    adjusted_value = q[i] + offset * (2 * np.pi)
                    if joint_limits[i][0] <= adjusted_value <= joint_limits[i][1]:
                        current_possible_values.append(adjusted_value)

                # If no values found in preferred direction, try all directions
                if not current_possible_values:
                    for offset in range(-2, 3):
                        adjusted_value = q[i] + offset * (2 * np.pi)
                        if joint_limits[i][0] <= adjusted_value <= joint_limits[i][1]:
                            current_possible_values.append(adjusted_value)

                # If still no valid values, mark solution as invalid
                if not current_possible_values:
                    valid_solution = False
                    break

                possible_arrays.append(current_possible_values)

            # Skip invalid solutions
            if not valid_solution:
                continue

            # Helper function to safely normalize weights
            def safe_normalize(weights):
                max_weight = np.max(weights)
                if max_weight > 0:
                    return weights / max_weight
                return np.zeros_like(weights)

            # Evaluate all combinations of possible joint values
            for combination in product(*possible_arrays):
                solution = np.array(combination)
                if solution.size != 6:
                    continue

                solution = solution.reshape(current_joints.shape)

                # Calculate optimization score for this solution
                # 1. Position difference penalty (weighted squared difference)
                pos_diff = np.sum((solution - current_joints) ** 2 * dynamic_weights)

                # 2. Joint limit proximity penalty
                limit_penalty = 0
                for i in range(6):
                    margin = 0.05  # 5% safety margin
                    # Calculate safe operating range
                    lower = joint_limits[i][0] + margin * (
                        joint_limits[i][1] - joint_limits[i][0]
                    )
                    upper = joint_limits[i][1] - margin * (
                        joint_limits[i][1] - joint_limits[i][0]
                    )

                    # Apply penalty if near joint limits
                    if solution[i] < lower or solution[i] > upper:
                        normalized_weights = safe_normalize(dynamic_weights)
                        limit_penalty += 5.0 * (1 - normalized_weights[i])

                # 3. Direction change penalty (for avoiding sign flips)
                direction_penalty = 0
                for i in range(6):
                    prev_move = current_joints[i] - prev_joints[i]
                    current_move = solution[i] - current_joints[i]
                    # Penalize direction reversals
                    if prev_move * current_move < 0:  # Opposite signs
                        direction_penalty += (
                            abs(current_move) * dynamic_weights[i] * 2.0
                        )

                # 4. Velocity continuity penalty (for smooth motion)
                velocity_penalty = 0
                for i in range(6):
                    prev_vel = current_joints[i] - prev_joints[i]
                    current_vel = solution[i] - current_joints[i]
                    accel = abs(current_vel - prev_vel)
                    # Penalize excessive acceleration (>30°/s²)
                    if accel > np.radians(30):
                        velocity_penalty += accel * dynamic_weights[i] * 5.0

                # Combine all penalty terms
                total_score = (
                    pos_diff + limit_penalty + direction_penalty + velocity_penalty
                )

                # Add to valid solutions list with score (for sorting when return_all_valid=True)
                all_valid_solutions.append((solution.copy(), total_score))

                # Update best solution if current one is better (for single solution return)
                if total_score < best_score:
                    best_score = total_score
                    best_qpos = solution.copy()

        # Return results based on what was requested
        if return_all_valid:
            if len(all_valid_solutions) == 0:
                return False, np.array([])

            # Sort solutions by score (ascending order - lower score is better)
            all_valid_solutions.sort(key=lambda x: x[1])

            # Extract only the solutions (remove scores)
            sorted_solutions = np.array([sol[0] for sol in all_valid_solutions])
            return True, sorted_solutions
        else:
            if best_qpos is None:
                return False, np.array([])
            return True, best_qpos
