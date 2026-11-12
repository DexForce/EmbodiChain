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

import torch
import numpy as np
import warp as wp
from ._buffers import _with_ik_buffers

from typing import TYPE_CHECKING

from embodichain.utils import configclass, logger
from embodichain.utils.math import inv_transform
from embodichain.lab.sim.motion.solvers import SolverCfg, BaseSolver
from embodichain.compute.kinematics._warp.opw import (
    OPWparam,
    opw_fk_kernel,
    opw_ik_kernel,
    opw_ik_path_select_kernel,
    opw_ik_select_kernel,
    wp_vec6f,
)
from embodichain.utils.device_utils import standardize_device_string

if TYPE_CHECKING:
    from typing import Self

__all__ = ["OPWSolver", "OPWSolverCfg"]


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
    safe_margin: float = 0.0  # 5.0 * np.pi / 180.0

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
        tcp_inv = inv_transform(self.tcp_xpos)
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
        # Retimed trajectories retain float64; the OPW kernel consumes float32.
        qpos_wp = wp.from_torch(qpos_batch.to(dtype=torch.float32).reshape(-1))
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

    @_with_ik_buffers
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
        kernel_device = standardize_device_string(self.device)

        if target_xpos.shape == (4, 4):
            target_xpos_batch = target_xpos[None, :, :].to(kernel_device)
        else:
            target_xpos_batch = target_xpos.to(kernel_device)
        n_sample = target_xpos_batch.shape[0]
        target_xpos_wp = wp.from_torch(target_xpos_batch.reshape(-1))

        all_qpos_wp = self._ik_buffers.zeros(
            "qpos", n_sample, N_SOL * DOF, torch.float32
        )
        all_ik_valid_wp = self._ik_buffers.zeros("valid", n_sample, N_SOL, torch.int32)

        # TODO: whether require gradient
        offsets_ = self.offsets.to(standardize_device_string(kernel_device))
        sign_corrections_ = self.sign_corrections.to(
            standardize_device_string(kernel_device)
        )
        # Pack both vectors in one host transfer instead of converting twelve
        # CUDA scalars separately. Reading live limits preserves setter/in-place
        # updates without an invalidation cache.
        limits = (
            torch.stack((self.lower_qpos_limits, self.upper_qpos_limits))
            .detach()
            .cpu()
            .tolist()
        )
        lower_limits_ = wp_vec6f(*limits[0])
        upper_limits_ = wp_vec6f(*limits[1])
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
            return all_ik_valid.clone(), all_qpos.clone()
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
            qpos_seed = (
                self.get_default_qpos_seed()
                .to(dtype=torch.float32, device=kernel_device)
                .unsqueeze(0)
                .repeat(n_sample, 1)
            )
            qpos_seed_wp = wp.from_torch(qpos_seed)
        all_qpos_wp = all_qpos_wp.reshape((n_sample, N_SOL, DOF))
        all_ik_valid_wp = all_ik_valid_wp.reshape((n_sample, N_SOL))
        joint_weight = kwargs.get("joint_weight")
        weights = (
            [1.0] * DOF
            if joint_weight is None
            else torch.as_tensor(joint_weight).detach().cpu().tolist()
        )
        joint_weight_wp = wp_vec6f(*weights)
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
        # Keep the public result independent from Warp-owned temporary storage
        # and from allocator reuse in a subsequent solve.
        return best_ik_valid.clone(), best_ik_result.clone()

    @property
    def supports_continuous_batch_ik(self) -> bool:
        """Return whether OPW supports batched candidates and continuous selection."""
        return True

    def _select_continuous_ik_path(
        self,
        candidate_qpos: torch.Tensor,
        candidate_valid: torch.Tensor,
        qpos_seed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select a continuous branch from precomputed OPW candidates."""
        if candidate_qpos.ndim != 4 or candidate_qpos.shape[-2:] != (8, 6):
            raise ValueError("candidate_qpos must have shape (B, N, 8, 6).")
        batch_size, sample_count = candidate_qpos.shape[:2]
        if candidate_valid.shape != (batch_size, sample_count, 8):
            raise ValueError(
                f"candidate_valid must have shape ({batch_size}, {sample_count}, 8)."
            )
        if qpos_seed.shape != (batch_size, 6):
            raise ValueError(f"qpos_seed must have shape ({batch_size}, 6).")
        kernel_device = standardize_device_string(self.device)
        lower_limits = self.lower_qpos_limits.detach().cpu().tolist()
        upper_limits = self.upper_qpos_limits.detach().cpu().tolist()
        path_qpos = wp.zeros(
            (batch_size, sample_count, 6), dtype=float, device=kernel_device
        )
        path_valid = wp.zeros(
            (batch_size, sample_count), dtype=int, device=kernel_device
        )
        joint_weight = self.ik_nearest_weight.detach().cpu().tolist()
        wp.launch(
            kernel=opw_ik_path_select_kernel,
            dim=batch_size,
            inputs=[
                wp.from_torch(candidate_qpos.to(kernel_device).contiguous()),
                wp.from_torch(
                    candidate_valid.to(
                        device=kernel_device, dtype=torch.int32
                    ).contiguous()
                ),
                wp.from_torch(
                    qpos_seed.to(device=kernel_device, dtype=torch.float32).contiguous()
                ),
                wp_vec6f(*joint_weight),
                wp_vec6f(*lower_limits),
                wp_vec6f(*upper_limits),
                self.cfg.safe_margin,
            ],
            outputs=[path_qpos, path_valid],
            device=kernel_device,
        )
        return wp.to_torch(path_valid).bool(), wp.to_torch(path_qpos)
