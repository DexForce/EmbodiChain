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
from typing import Optional, Union, Tuple, Any, Literal, TYPE_CHECKING
from scipy.spatial.transform import Rotation

from embodichain.utils import configclass, logger
from embodichain.lab.sim.solvers import SolverCfg, BaseSolver
from embodichain.utils.math import (
    apply_delta_pose,
    compute_pose_error,
)


if TYPE_CHECKING:
    from typing import Self


@configclass
class DifferentialSolverCfg(SolverCfg):
    """Configuration for differential inverse kinematics controller."""

    class_type: str = "DifferentialSolver"

    pos_eps: float = 5e-4  # Tolerance for convergence for position
    rot_eps: float = 5e-4  # Tolerance for convergence for rotation
    max_iterations: int = 1000  # Maximum number of iterations for the solver

    # Constraint configuration
    is_only_position_constraint: bool = (
        False  # Whether to only consider position constraints
    )

    # Type of task-space command to control the articulation's body.
    command_type: Literal["position", "pose"] = "pose"

    # Whether to use relative mode for the controller.
    use_relative_mode: bool = False

    # Method for computing inverse of Jacobian."""
    ik_method: Literal["pinv", "svd", "trans", "dls"] = "pinv"

    # Parameters for the inverse-kinematics method.
    ik_params: Optional[dict] = None

    def __post_init__(self):
        # Default parameters for different inverse kinematics approaches
        default_ik_params = {
            "pinv": {"k_val": 1.0},
            "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
            "trans": {"k_val": 1.0},
            "dls": {"lambda_val": 0.01},
        }

        # Update parameters for IK-method if not provided
        params = self.ik_params or {}
        self.ik_params = {**default_ik_params[self.ik_method], **params}

    def init_solver(
        self, num_envs: int = 1, device: torch.device = torch.device("cpu"), **kwargs
    ) -> "DifferentialSolver":
        """Initialize the solver with the configuration.

        Args:
            device (torch.device): The device to use for the solver. Defaults to CPU.
            num_envs (int): The number of environments for which the solver is initialized.
            **kwargs: Additional keyword arguments that may be used for solver initialization.

        Returns:
            DifferentialSolver: An initialized solver instance.
        """

        solver = DifferentialSolver(
            cfg=self, num_envs=num_envs, device=device, **kwargs
        )

        # Set the Tool Center Point (TCP) for the solver
        solver.set_tcp(self._get_tcp_as_numpy())

        return solver


class DifferentialSolver(BaseSolver):
    r"""Differential inverse kinematics (IK) controller.

    This controller implements differential inverse kinematics using various methods for
    computing the inverse of the Jacobian matrix.
    """

    def __init__(
        self,
        cfg: DifferentialSolverCfg,
        num_envs: int = 1,
        device: str = "cpu",
        **kwargs,
    ):
        r"""Initializes the differential kinematics solver.

            This constructor sets up the kinematics solver using differential methods,
            allowing for efficient computation of robot kinematics based on
            the specified URDF model.

        Args:
            cfg: The configuration for the solver.
            num_envs (int): The number of environments for the solver. Defaults to 1.
            device (str, optional): The device to use for the solver (e.g., "cpu" or "cuda"). Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the base solver.

        """
        super().__init__(cfg=cfg, num_envs=num_envs, device=device, **kwargs)

        # Initialize buffers
        self.ee_pos_des = torch.zeros(num_envs, 3, device=device)
        self.ee_quat_des = torch.zeros(num_envs, 4, device=device)
        self._command = torch.zeros(num_envs, self.action_dim, device=device)

    @property
    def action_dim(self) -> int:
        """Returns the dimension of the controller's input command.

        Returns:
            int: The dimension of the input command.
        """
        if self.cfg.command_type == "position":
            return 3  # (x, y, z)
        elif self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6  # (dx, dy, dz, droll, dpitch, dyaw)
        else:
            return 7  # (x, y, z, qw, qx, qy, qz)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset the internal buffers for the specified environments.

        Args:
            env_ids (Optional[torch.Tensor]): The environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.ee_pos_des[env_ids] = 0
        self.ee_quat_des[env_ids] = torch.tensor([1.0, 0, 0, 0], device=self.device)
        self._command[env_ids] = 0

    def set_command(
        self,
        command: torch.Tensor,
        ee_pos: Optional[torch.Tensor] = None,
        ee_quat: Optional[torch.Tensor] = None,
    ) -> bool:
        """Set the target end-effector pose command.

        Args:
            command (torch.Tensor): The command tensor.
            ee_pos (Optional[torch.Tensor]): Current end-effector position (for relative mode).
            ee_quat (Optional[torch.Tensor]): Current end-effector quaternion (for relative mode).

        Returns:
            bool: True if the command was set successfully, False otherwise.
        """
        # TODO: Init solver with correct batch size
        batch_size = command.shape[0]
        if self._command.shape[0] != batch_size:
            device = command.device
            self._command = torch.zeros(batch_size, self.action_dim, device=device)
            self.ee_pos_des = torch.zeros(batch_size, 3, device=device)
            self.ee_quat_des = torch.zeros(batch_size, 4, device=device)
        self._command[:] = command

        if self.cfg.command_type == "position":
            if ee_quat is None:
                logger.log_warning(
                    "End-effector orientation cannot be None for position control"
                )
                return False

            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    logger.log_warning("Current position required for relative mode")
                    return False
                self.ee_pos_des[:] = ee_pos + self._command
                self.ee_quat_des[:] = ee_quat
            else:
                self.ee_pos_des[:] = self._command
                self.ee_quat_des[:] = ee_quat
        else:
            if self.cfg.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    logger.log_warning("Current pose required for relative mode")
                    return False
                self.ee_pos_des, self.ee_quat_des = apply_delta_pose(
                    ee_pos, ee_quat, self._command
                )
            else:
                self.ee_pos_des = self._command[:, 0:3]
                self.ee_quat_des = self._command[:, 3:7]

        return True

    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor = None,
        return_all_solutions: bool = False,
        jacobian: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute target joint positions using differential inverse kinematics.

        Args:
            target_xpos (torch.Tensor): Current end-effector position, shape (num_envs, 3).
            qpos_seed (torch.Tensor): Current joint positions, shape (num_envs, num_joints). Defaults to zeros.
            return_all_solutions (bool, optional): Whether to return all IK solutions or just the best one. Defaults to False.
            jacobian (torch.Tensor): Jacobian matrix, shape (num_envs, 6, num_joints).
            **kwargs: Additional keyword arguments for future extensions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - success (torch.Tensor): Boolean tensor indicating IK solution validity for each environment, shape (num_envs,).
                - target_joints (torch.Tensor): Computed target joint positions, shape (num_envs, num_joints).
        """
        if qpos_seed is None:
            qpos_seed = torch.zeros(self.dof, device=self.device)

        if jacobian is None:
            jacobian = self.get_jacobian(qpos_seed)
            current_xpos = self.get_fk(qpos_seed, to_matrix=True)

        # Transform target_xpos by TCP
        # Note: torch.as_tensor does not modify the input, so deepcopy is unnecessary
        tcp_xpos = torch.as_tensor(
            self.tcp_xpos, device=self.device, dtype=torch.float32
        )
        tcp_xpos_inv = torch.inverse(tcp_xpos)
        current_xpos = current_xpos @ tcp_xpos_inv
        compute_xpos = target_xpos @ tcp_xpos_inv

        # Ensure compute_xpos is a batch of matrices
        if current_xpos.dim() == 2 and current_xpos.shape == (4, 4):
            current_xpos = current_xpos.unsqueeze(0)

        # Ensure compute_xpos is a batch of matrices
        if compute_xpos.dim() == 2 and compute_xpos.shape == (4, 4):
            compute_xpos = compute_xpos.unsqueeze(0)

        compute_pose = self._matrix_to_pos_quat(compute_xpos)
        self.set_command(command=compute_pose)

        qpos = qpos_seed
        num_iter = 1 if self.cfg.max_iterations == 1 else self.cfg.max_iterations
        for i in range(num_iter):
            current_pose = self._matrix_to_pos_quat(current_xpos)
            ee_pos = current_pose[:, :3]
            ee_quat = current_pose[:, 3:]

            if self.cfg.command_type == "position":
                position_error = self.ee_pos_des - ee_pos
                jacobian_pos = jacobian[:, :3]
                delta_joint_pos = self._compute_delta_joint_pos(
                    delta_pose=position_error, jacobian=jacobian_pos
                )
            else:
                pos_error, rot_error = compute_pose_error(
                    ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des
                )
                pose_error = torch.cat((pos_error, rot_error), dim=1)
                delta_joint_pos = self._compute_delta_joint_pos(
                    delta_pose=pose_error, jacobian=jacobian
                )

            qpos = qpos + delta_joint_pos
            current_xpos = self.get_fk(qpos)

            # Ensure current_xpos and target_xpos are batches of matrices
            if current_xpos.dim() == 2 and current_xpos.shape == (4, 4):
                current_xpos = current_xpos.unsqueeze(0)

            if target_xpos.dim() == 2 and target_xpos.shape == (4, 4):
                target_xpos = target_xpos.unsqueeze(0)

            pos_converged = (
                torch.norm(current_xpos[:, :3, 3] - target_xpos[:, :3, 3], dim=1)
                < self.cfg.pos_eps
            )
            rot_converged = (
                torch.norm(current_xpos[:, :3, :3] - target_xpos[:, :3, :3], dim=(1, 2))
                < self.cfg.rot_eps
            )

            if self.cfg.is_only_position_constraint:
                if pos_converged.all():
                    break
            else:
                if (pos_converged & rot_converged).all():
                    break

        if return_all_solutions:
            logger.log_warning(
                "return_all_solutions=True is not supported in DifferentialSolver. Returning the best solution only."
            )

        if self.cfg.is_only_position_constraint:
            success = pos_converged
        else:
            success = pos_converged & rot_converged

        return success, qpos

    # Helper functions
    def _compute_delta_joint_pos(
        self, delta_pose: torch.Tensor, jacobian: torch.Tensor
    ) -> torch.Tensor:
        """Compute joint-space delta using the specified IK method.

        Args:
            delta_pose (torch.Tensor): The pose error tensor.
            jacobian (torch.Tensor): The Jacobian matrix.

        Returns:
            torch.Tensor: The joint-space delta tensor.
        """
        method = self.cfg.ik_method
        params = self.cfg.ik_params

        # compute the delta in joint-space
        if method == "pinv":  # Jacobian pseudo-inverse
            # params
            k_val = params["k_val"]
            # compute
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_joint_pos = k_val * (
                jacobian_pinv @ delta_pose.unsqueeze(-1)
            ).squeeze(-1)
        elif method == "svd":
            # params
            k_val = params["k_val"]
            min_singular_value = params["min_singular_value"]
            # compute
            # U: 6xd, S: dxd, V: d x num-joint
            U, S, Vh = torch.linalg.svd(jacobian, full_matrices=False)
            S_inv = 1.0 / S
            S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = (
                torch.transpose(Vh, 1, 2)[:, :, :6]
                @ torch.diag_embed(S_inv)
                @ torch.transpose(U, 1, 2)
            )
            delta_joint_pos = k_val * (
                jacobian_pinv @ delta_pose.unsqueeze(-1)
            ).squeeze(-1)
        elif method == "trans":
            # params
            k_val = params["k_val"]
            # compute
            jacobian_T = torch.transpose(jacobian, 1, 2)
            delta_joint_pos = params["k_val"] * (
                jacobian_T @ delta_pose.unsqueeze(-1)
            ).squeeze(-1)
        elif method == "dls":
            # params
            lambda_val = self.cfg.ik_params["lambda_val"]
            # compute
            jacobian_T = torch.transpose(jacobian, 1, 2)
            lambda_matrix = (lambda_val**2) * torch.eye(
                jacobian.shape[1], device=self.device
            )
            delta_joint_pos = (
                jacobian_T
                @ torch.linalg.solve(
                    jacobian @ jacobian_T + lambda_matrix, delta_pose.unsqueeze(-1)
                )
            ).squeeze(-1)
        else:
            raise ValueError(f"Unsupported IK method: {method}")

        return delta_joint_pos

    @staticmethod
    def _matrix_to_pos_quat(mat):
        """Convert a transformation matrix to position and quaternion.

        Args:
            mat (torch.Tensor): Transformation matrix tensor of shape (N, 4, 4).

        Returns:
            torch.Tensor: Concatenated position and quaternion tensor of shape (N, 7).
        """
        # Ensure mat is a batch of matrices
        if mat.dim() == 2 and mat.shape == (4, 4):
            mat = mat.unsqueeze(0)  # Convert (4, 4) to (1, 4, 4)
        elif mat.dim() != 3 or mat.shape[1:] != (4, 4):
            raise ValueError(
                f"Expected mat to have shape (N, 4, 4), but got {mat.shape}"
            )

        # Extract position
        pos = mat[:, :3, 3]

        # Extract rotation matrix and convert to quaternion
        rot_matrices = mat[:, :3, :3].cpu().numpy()  # Convert to NumPy for scipy
        quats = Rotation.from_matrix(rot_matrices).as_quat()  # (N, 4), [x, y, z, w]

        # Convert quaternion back to torch.Tensor and reorder to [w, x, y, z]
        quats = torch.tensor(quats, device=mat.device, dtype=mat.dtype)  # (N, 4)
        quats = quats[:, [3, 0, 1, 2]]  # Reorder to [w, x, y, z]

        # Concatenate position and quaternion
        return torch.cat([pos, quats], dim=1)
