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

"""A graph-preserving IK layer for supervised and policy learning."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from embodichain.compute.kinematics import (
    DifferentiableIKCfg,
    DifferentiableIKResult,
    solve_dls_ik,
)
from .kinematics import KinematicsLayer

__all__ = ["DifferentiableIKLayer"]


class DifferentiableIKLayer(nn.Module):
    """Compose fixed-step DLS with a differentiable kinematics module.

    Args:
        kinematics: Model owning geometry, TCP and physical limit buffers.
        cfg: Numerical settings, copied at construction.

    Note:
        There is no recurrent seed state or optimizer inside this module.
        The caller owns rollout state and backpropagation boundaries.
    """

    def __init__(
        self, kinematics: KinematicsLayer, cfg: DifferentiableIKCfg | None = None
    ) -> None:
        super().__init__()
        self.kinematics = kinematics
        self.cfg = deepcopy(cfg or DifferentiableIKCfg())
        self.cfg.__post_init__()

    def forward(
        self, target_xpos: torch.Tensor, qpos_seed: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return joint positions while retaining target and seed gradients."""
        return self.solve(target_xpos, qpos_seed).qpos

    def solve(
        self, target_xpos: torch.Tensor, qpos_seed: torch.Tensor | None = None
    ) -> DifferentiableIKResult:
        """Return final joints, validity and differentiable TCP residuals.

        Args:
            target_xpos: Root-frame TCP transforms, shape ``(..., 4, 4)``.
            qpos_seed: ``(dof,)`` or matching batch dimensions; None uses midpoints.

        Returns:
            Result retaining the target's leading batch dimensions.
        """
        seed = self.kinematics.default_seed() if qpos_seed is None else qpos_seed
        return solve_dls_ik(
            target_xpos,
            seed,
            self.kinematics.evaluate,
            self.kinematics.lower,
            self.kinematics.upper,
            self.cfg,
        )

    def get_extra_state(self) -> dict[str, Any]:
        """Save numerical settings with the module checkpoint."""
        return asdict(self.cfg)

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Restore and validate numerical settings."""
        self.cfg = DifferentiableIKCfg(**state)
