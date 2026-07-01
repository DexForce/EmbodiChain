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

import math
from typing import Any

import numpy as np
import torch
import warp as wp

from embodichain.data import get_data_path
from embodichain.lab.sim.solvers import BaseSolver, SolverCfg
from embodichain.utils import configclass, logger
from embodichain.utils.device_utils import standardize_device_string
from embodichain.utils.warp.kinematics.ur_solver import (
    URParam,
    ur_ik_kernel,
)


@configclass
class URSolverCfg(SolverCfg):
    class_type: str = "URSolver"
    ur_type: str = "ur10"
    end_link_name: str = "ee_link"
    root_link_name: str = "base_link"
    urdf_path: str = get_data_path("UniversalRobots/UR10/UR10.urdf")
    # DH parameters: default ur10 parameters
    d1: float = 0.1273
    a2: float = -0.612
    a3: float = -0.5723
    d4: float = 0.163941
    d5: float = 0.1157
    d6: float = 0.0922

    alpha1: float = torch.pi * 0.5
    alpha4: float = torch.pi * 0.5
    alpha5: float = -torch.pi * 0.5

    def __post_init__(self):
        super().__post_init__()
        # from https://github.com/Victorlouisdg/ur-analytic-ik/blob/main/src/ur_analytic_ik/dh_parameters.hh
        if self.ur_type == "ur3":
            self.d1 = 0.1519
            self.d4 = 0.11235
            self.d5 = 0.08535
            self.d6 = 0.0819
            self.a2 = -0.24365
            self.a3 = -0.21325
            self.urdf_path = get_data_path("UniversalRobots/UR3/UR3.urdf")
        elif self.ur_type == "ur3e":
            self.d1 = 0.15185
            self.d4 = 0.13105
            self.d5 = 0.08535
            self.d6 = 0.0921
            self.a2 = -0.24355
            self.a3 = -0.2132
            self.urdf_path = get_data_path("UniversalRobots/UR3e/UR3e.urdf")
        elif self.ur_type == "ur5":
            self.d1 = 0.089159
            self.d4 = 0.10915
            self.d5 = 0.09465
            self.d6 = 0.0823
            self.a2 = -0.425
            self.a3 = -0.39225
            self.urdf_path = get_data_path("UniversalRobots/UR5/UR5.urdf")
        elif self.ur_type == "ur5e":
            self.d1 = 0.1625
            self.d4 = 0.1333
            self.d5 = 0.0997
            self.d6 = 0.0996
            self.a2 = -0.425
            self.a3 = -0.3922
            self.urdf_path = get_data_path("UniversalRobots/UR5e/UR5e.urdf")
        elif self.ur_type == "ur10":
            self.d1 = 0.1273
            self.d4 = 0.163941
            self.d5 = 0.1157
            self.d6 = 0.0922
            self.a2 = -0.612
            self.a3 = -0.5723
            self.urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
        elif self.ur_type == "ur10e":
            self.d1 = 0.1807
            self.d4 = 0.17415
            self.d5 = 0.11985
            self.d6 = 0.11655
            self.a2 = -0.612
            self.a3 = -0.5723
            self.urdf_path = get_data_path("UniversalRobots/UR10e/UR10e.urdf")
        else:
            raise ValueError(f"Unknown UR type: {self.ur_type}")

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> "URSolverCfg":
        """Initialize the UR solver configuration from a dictionary."""
        cfg = cls(ur_type=init_dict.get("ur_type", "ur10"))
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.log_warning(f"Key '{key}' not found in {cls.__name__}.")
        return cfg

    def init_solver(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> "URSolver":
        """Initialize the solver with the configuration.

        Args:
            device (torch.device): The device to use for the solver. Defaults to CPU.
            **kwargs: Additional keyword arguments that may be used for solver initialization.

        Returns:
            URSolver: An initialized solver instance.
        """
        solver = URSolver(cfg=self, device=device, **kwargs)
        solver.set_tcp(self._get_tcp_as_numpy())
        return solver


class URSolver(BaseSolver):
    def __init__(self, cfg: URSolverCfg, device: str, **kwargs):
        super().__init__(cfg, device, **kwargs)
        self.dof = 6
        self._init_warp_solver(cfg)

    def _init_warp_solver(self, cfg: URSolverCfg):
        self._ur_params = URParam()
        self._ur_params.d1 = cfg.d1
        self._ur_params.a2 = cfg.a2
        self._ur_params.a3 = cfg.a3
        self._ur_params.d4 = cfg.d4
        self._ur_params.d5 = cfg.d5
        self._ur_params.d6 = cfg.d6

    def set_tcp(self, tcp: np.ndarray):
        super().set_tcp(tcp)
        self._tcp_inv = np.eye(4, dtype=float)
        self._tcp_inv[:3, :3] = self.tcp_xpos[:3, :3].T
        self._tcp_inv[:3, 3] = -self._tcp_inv[:3, :3] @ self.tcp_xpos[:3, 3]

    def get_ik(
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
        N_SOL = 512
        DOF = 6
        if target_xpos.shape == (4, 4):
            target_xpos_batch = target_xpos[None, :, :]
        else:
            target_xpos_batch = target_xpos
        tcp_inv = torch.tensor(self._tcp_inv, dtype=torch.float32, device=self.device)
        target_xpos_batch = target_xpos_batch @ tcp_inv[None, :, :]
        n_sample = target_xpos_batch.shape[0]

        device = self.device
        wp_device = standardize_device_string(self.device)
        # Flatten target poses to a 1-D float array for the Warp kernel.
        xpos_wp = wp.from_torch(target_xpos_batch.reshape(-1))
        all_qpos_wp = wp.zeros(n_sample * N_SOL * DOF, dtype=float, device=wp_device)
        all_ik_valid_wp = wp.zeros(n_sample * N_SOL, dtype=int, device=wp_device)
        lower_qpos_limits_wp = wp.from_torch(self.lower_qpos_limits)
        upper_qpos_limits_wp = wp.from_torch(self.upper_qpos_limits)
        wp.launch(
            kernel=ur_ik_kernel,
            dim=(n_sample,),
            inputs=[
                xpos_wp,
                self._ur_params,
                lower_qpos_limits_wp,
                upper_qpos_limits_wp,
            ],
            outputs=[all_qpos_wp, all_ik_valid_wp],
            device=wp_device,
        )

        all_solutions = (
            wp.to_torch(all_qpos_wp)
            .reshape(n_sample, N_SOL, DOF)
            .to(dtype=torch.float32, device=device)
        )
        all_solutions_validity = (
            wp.to_torch(all_ik_valid_wp)
            .reshape(n_sample, N_SOL)
            .bool()
            .to(device=device)
        )

        if return_all_solutions:
            return all_solutions_validity, all_solutions
        # Select ik qpos based on the closest distance to the seed qpos
        qpos_seed_expanded = qpos_seed.unsqueeze(1).expand(-1, N_SOL, -1)
        distances = torch.norm(all_solutions - qpos_seed_expanded, dim=-1)
        # fill invalid solutions with inf distance
        distances[~all_solutions_validity] = float("inf")
        closest_indices = torch.argmin(distances, dim=1)
        ik_qpos = all_solutions[torch.arange(n_sample), closest_indices]
        ik_validity = all_solutions_validity[torch.arange(n_sample), closest_indices]
        return ik_validity, ik_qpos

    @staticmethod
    def dh_matrix(theta_i, d_i, a_i, alpha_i):
        """
        Compute the Denavit-Hartenberg transformation matrix.

        Args:
            theta_i (float): Joint angle in radians.
            d_i (float): Link offset along the previous z-axis.
            a_i (float): Link length along the previous x-axis.
            alpha_i (float): Link twist angle in radians.

        Returns:
            torch.Tensor: A 4x4 transformation matrix representing the pose of the next link.
        """
        m = torch.zeros((4, 4), dtype=torch.float32)
        s_ai = math.sin(alpha_i)
        c_ai = math.cos(alpha_i)

        m[0, 0] = torch.cos(theta_i)
        m[0, 1] = -torch.sin(theta_i) * c_ai
        m[0, 2] = torch.sin(theta_i) * s_ai
        m[0, 3] = a_i * torch.cos(theta_i)

        m[1, 0] = torch.sin(theta_i)
        m[1, 1] = torch.cos(theta_i) * c_ai
        m[1, 2] = -torch.cos(theta_i) * s_ai
        m[1, 3] = a_i * torch.sin(theta_i)

        m[2, 0] = 0.0
        m[2, 1] = s_ai
        m[2, 2] = c_ai
        m[2, 3] = d_i

        m[3, 0] = 0.0
        m[3, 1] = 0.0
        m[3, 2] = 0.0
        m[3, 3] = 1.0

        return m


__all__ = ["URSolverCfg", "URSolver"]
