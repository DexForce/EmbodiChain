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

"""Robot runtime adapter for the shared differentiable DLS computation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, List  # List resolves inherited SolverCfg annotations.

import numpy as np
import torch

from embodichain.compute.kinematics import (
    DifferentiableIKCfg,
    DifferentiableIKResult,
    SerialChainKinematics,
    solve_dls_ik,
)
from embodichain.utils import configclass
from .base_solver import BaseSolver, SolverCfg

__all__ = ["DifferentiableIKSolverCfg", "DifferentiableIKSolver"]


@configclass
class DifferentiableIKSolverCfg(SolverCfg):
    """Robot binding plus shared DLS options; geometry is not trainable."""

    class_type: str = "DifferentiableIKSolver"
    ik: DifferentiableIKCfg = DifferentiableIKCfg()
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if isinstance(self.ik, dict):
            self.ik = DifferentiableIKCfg(**self.ik)
        self.ik.__post_init__()
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")

    def init_solver(
        self, device: torch.device | str = "cpu", **kwargs: Any
    ) -> DifferentiableIKSolver:
        """Build a graph-preserving runtime solver on the requested device."""
        return DifferentiableIKSolver(self, device, **kwargs)


class DifferentiableIKSolver(BaseSolver):
    """Single-seed IK sharing the exact computation used by training layers."""

    def __init__(
        self,
        cfg: DifferentiableIKSolverCfg,
        device: torch.device | str | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = deepcopy(cfg)
        cfg.__post_init__()
        device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = getattr(torch, cfg.dtype)
        chain = kwargs.pop("pk_serial_chain", None)
        if chain is None:
            model = SerialChainKinematics.from_urdf(
                Path(cfg.urdf_path).read_text(),
                cfg.end_link_name,
                cfg.root_link_name,
                device=device,
                dtype=dtype,
            )
        else:
            model = SerialChainKinematics(chain, device, dtype)
        if cfg.joint_names is not None and tuple(cfg.joint_names) != model.joint_names:
            raise ValueError("Configured joint order must match the serial chain")
        cfg.joint_names = list(model.joint_names)
        super().__init__(cfg, device, pk_serial_chain=model.chain, **kwargs)
        self.model = model
        self.dtype = dtype
        self.set_tcp(cfg._get_tcp_as_numpy())

    def set_tcp(self, xpos: np.ndarray) -> None:
        """Update the validated fixed TCP and its device tensor."""
        super().set_tcp(xpos)
        self._tcp_tensor = torch.as_tensor(
            self.tcp_xpos, dtype=self.dtype, device=self.device
        ).clone()

    def _evaluate(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.evaluate(qpos, self._tcp_tensor)

    def get_fk(self, qpos: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return batched TCP FK with this solver's dtype and device."""
        joints = torch.as_tensor(qpos, dtype=self.dtype, device=self.device)
        if joints.ndim == 1:
            joints = joints.unsqueeze(0)
        return self._evaluate(joints)[0]

    def get_jacobian(
        self,
        qpos: torch.Tensor,
        locations: torch.Tensor | np.ndarray | None = None,
        jac_type: str = "full",
    ) -> torch.Tensor:
        """Return TCP Jacobians, or Jacobians at a supplied flange offset.

        Args:
            qpos: Joint positions, shape ``(..., dof)``.
            locations: Optional flange-frame translations, shape ``(3,)`` or
                matching the joint batch dimensions followed by ``3``.
            jac_type: ``full``, ``trans`` or ``rot``.

        Returns:
            Root-frame Jacobian with the selected rows.
        """
        joints = torch.as_tensor(qpos, dtype=self.dtype, device=self.device)
        if joints.ndim == 1:
            joints = joints.unsqueeze(0)
        tcp = self._tcp_tensor
        if locations is not None:
            offset = torch.as_tensor(locations, dtype=self.dtype, device=self.device)
            batch = joints.shape[:-1]
            if offset.shape not in ((3,), (*batch, 3)):
                raise ValueError("locations must be (3,) or match the joint batch")
            tcp = tcp.expand(*batch, 4, 4)
            top = torch.cat((tcp[..., :3, :3], offset.expand(*batch, 3)[..., None]), -1)
            tcp = torch.cat((top, tcp[..., 3:, :]), -2)
        jacobian = self.model.evaluate(joints, tcp)[1]
        if jac_type == "trans":
            return jacobian[..., :3, :]
        if jac_type == "rot":
            return jacobian[..., 3:, :]
        if jac_type != "full":
            raise ValueError("jac_type must be full, trans or rot")
        return jacobian

    def solve(
        self, target_xpos: torch.Tensor, qpos_seed: torch.Tensor | None = None
    ) -> DifferentiableIKResult:
        """Return final joints and differentiable residuals for arbitrary batches."""
        target = torch.as_tensor(target_xpos, dtype=self.dtype, device=self.device)
        lower = self.lower_qpos_limits.to(dtype=self.dtype, device=self.device)
        upper = self.upper_qpos_limits.to(dtype=self.dtype, device=self.device)
        if qpos_seed is None:
            bounded = torch.isfinite(lower) & torch.isfinite(upper)
            seed = ((lower.where(bounded, 0) + upper.where(bounded, 0)) * 0.5).clamp(
                min=lower, max=upper
            )
        else:
            seed = torch.as_tensor(qpos_seed, dtype=self.dtype, device=self.device)
        return solve_dls_ik(target, seed, self._evaluate, lower, upper, self.cfg.ik)

    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor | None = None,
        return_all_solutions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single solution, optionally exposing a singleton candidate axis."""
        target = torch.as_tensor(target_xpos, dtype=self.dtype, device=self.device)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        result = self.solve(target, qpos_seed)
        if return_all_solutions:
            return result.success.unsqueeze(-1), result.qpos.unsqueeze(-2)
        return result.success, result.qpos
