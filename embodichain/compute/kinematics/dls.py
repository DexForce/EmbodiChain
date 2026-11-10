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

"""Pure Torch DLS updates and fixed-step differentiable IK."""

from __future__ import annotations

from collections.abc import Callable
import math

import torch

from .pose_error import pose_error
from .types import DifferentiableIKCfg, DifferentiableIKResult

__all__ = ["damped_least_squares_step", "solve_dls_ik"]


def damped_least_squares_step(
    error: torch.Tensor, jacobian: torch.Tensor, damping: float
) -> torch.Tensor:
    """Compute ``J.T @ solve(J @ J.T + damping**2 I, error)``.

    Args:
        error: Task error, shape ``(..., task_dim)``.
        jacobian: Jacobian, shape ``(..., task_dim, dof)``.
        damping: Finite positive damping coefficient.

    Returns:
        Joint increment, shape ``(..., dof)``, preserving dtype and device.
    """
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be finite and positive")
    if jacobian.ndim < 2 or error.shape != jacobian.shape[:-1]:
        raise ValueError("error must match the Jacobian batch and task dimensions")
    if error.dtype != jacobian.dtype or error.device != jacobian.device:
        raise ValueError("error and jacobian must share dtype and device")
    jt = jacobian.transpose(-1, -2)
    regularizer = damping**2 * torch.eye(
        jacobian.shape[-2], dtype=jacobian.dtype, device=jacobian.device
    )
    solution, _ = torch.linalg.solve_ex(
        jacobian @ jt + regularizer, error.unsqueeze(-1), check_errors=False
    )
    return (jt @ solution).squeeze(-1)


def solve_dls_ik(
    target_xpos: torch.Tensor,
    qpos_seed: torch.Tensor,
    kinematics: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    lower: torch.Tensor,
    upper: torch.Tensor,
    cfg: DifferentiableIKCfg,
) -> DifferentiableIKResult:
    """Unroll IK without detaching, early stopping, or shared scratch storage.

    Args:
        target_xpos: Root-frame TCP targets, shape ``(..., 4, 4)``.
        qpos_seed: Joint seed, shape ``(dof,)`` or ``(..., dof)``.
        kinematics: Maps flattened ``(N, dof)`` joints to TCP transforms and
            root-frame geometric Jacobians. Both outputs must be differentiable.
        lower: Physical lower joint limits, shape ``(dof,)``.
        upper: Physical upper joint limits, shape ``(dof,)``.
        cfg: Fixed-step solver configuration.

    Returns:
        Final iterate and residuals with the target's leading batch dimensions.
        Failure returns the final iterate, never a detached seed fallback.

    Note:
        All floating tensors must share dtype/device; inputs must be finite,
        targets must be rigid transforms, and lower limits cannot exceed upper.
        Success still checks physical limits when projection is disabled.
    """
    if (
        target_xpos.shape[-2:] != (4, 4)
        or lower.ndim != 1
        or upper.shape != lower.shape
    ):
        raise ValueError("Invalid target or limit shapes")
    batch = target_xpos.shape[:-2]
    dof = lower.numel()
    if qpos_seed.shape not in ((dof,), (*batch, dof)):
        raise ValueError("Seed must be (dof,) or match target batch dimensions")
    if target_xpos.dtype not in (torch.float32, torch.float64):
        raise ValueError("Differentiable IK supports float32 and float64")
    for value in (qpos_seed, lower, upper):
        if value.dtype != target_xpos.dtype or value.device != target_xpos.device:
            raise ValueError("All IK inputs must share dtype and device")
    target = target_xpos.reshape(-1, 4, 4)
    q = qpos_seed.expand(*batch, dof).reshape(-1, dof)
    if cfg.enforce_joint_limits:
        q = q.clamp(min=lower, max=upper)
    for _ in range(cfg.max_iterations):
        current, jacobian = kinematics(q)
        position, rotation = pose_error(current, target)
        q = q + cfg.step_size * damped_least_squares_step(
            torch.cat((position, rotation), -1), jacobian, cfg.damping
        )
        if cfg.enforce_joint_limits:
            q = q.clamp(min=lower, max=upper)
    current, _ = kinematics(q)
    position, rotation = pose_error(current, target)
    finite = (
        torch.isfinite(q).all(-1)
        & torch.isfinite(position).all(-1)
        & torch.isfinite(rotation).all(-1)
    )
    success = (
        finite
        & (position.norm(dim=-1) < cfg.pos_eps)
        & (rotation.norm(dim=-1) < cfg.rot_eps)
        & ((q >= lower) & (q <= upper)).all(-1)
    )
    return DifferentiableIKResult(
        q.reshape(*batch, dof),
        success.reshape(batch),
        position.reshape(*batch, 3),
        rotation.reshape(*batch, 3),
    )
