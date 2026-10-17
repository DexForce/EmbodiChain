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

"""Configuration and results for graph-preserving inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from embodichain.utils import configclass

__all__ = ["DifferentiableIKCfg", "DifferentiableIKResult"]


@configclass
class DifferentiableIKCfg:
    """Fixed-step DLS options shared by training and robot adapters.

    Damping is strictly positive. Hard limit projection is piecewise
    differentiable and has zero gradients outside the projected interval.
    Tolerances only classify the final result; they never stop iterations.
    """

    max_iterations: int = 16
    damping: float = 0.05
    step_size: float = 0.5
    pos_eps: float = 5e-4
    rot_eps: float = 5e-4
    enforce_joint_limits: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enforce_joint_limits, bool):
            raise ValueError("enforce_joint_limits must be a boolean")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        for name in ("damping", "step_size", "pos_eps", "rot_eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class DifferentiableIKResult:
    """Final iterate and root-frame TCP residuals, retaining their graphs.

    Attributes:
        qpos: Joint positions, shape ``(..., dof)``.
        success: Boolean convergence and physical-limit mask, shape ``(...)``.
        position_error: Target minus achieved translation, shape ``(..., 3)``.
        rotation_error: Shortest root-frame rotation vector, shape ``(..., 3)``.
    """

    qpos: torch.Tensor
    success: torch.Tensor
    position_error: torch.Tensor
    rotation_error: torch.Tensor
