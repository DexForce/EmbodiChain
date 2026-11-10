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

"""Internal kinematic derivatives for sampled Cartesian paths.

These tensor-only operations use the caller's Jacobian frame and joint order.
"""

from __future__ import annotations

import torch

__all__: list[str] = []


def joint_derivatives_from_path_time_law(
    jacobians: torch.Tensor,
    path_positions: torch.Tensor,
    path_velocities: torch.Tensor,
    path_accelerations: torch.Tensor,
    cartesian_tangent: torch.Tensor,
    cartesian_curvature: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert scalar path derivatives into joint derivatives.

    For a Cartesian line ``x(s)`` the tangent is constant and
    ``d²x/ds² = 0``. The joint path derivatives therefore satisfy
    ``J q_s = x_s`` and ``J q_ss = -J_s q_s``. Double-S then supplies the
    time law through ``q_dot = q_s s_dot`` and
    ``q_ddot = q_ss s_dot² + q_s s_ddot``.

    Args:
        jacobians: Geometric Jacobians shaped ``(B, N, 6, DOF)``.
        path_positions: Scalar metric path positions shaped ``(B, N)``.
        path_velocities: Scalar path velocities shaped ``(B, N)``.
        path_accelerations: Scalar path accelerations shaped ``(B, N)``.
        cartesian_tangent: Root-frame unit twists shaped ``(B, 6)`` or
            per-sample twists shaped ``(B, N, 6)``.
        cartesian_curvature: Optional root-frame second derivatives with
            respect to arc distance, shaped ``(B, N, 6)``.

    Returns:
        Joint velocity and acceleration tensors shaped ``(B, N, DOF)``.
    """
    batch_size, sample_count, _, dof = jacobians.shape
    expected_scalar_shape = (batch_size, sample_count)
    if (
        path_positions.shape != expected_scalar_shape
        or path_velocities.shape != expected_scalar_shape
        or path_accelerations.shape != expected_scalar_shape
        or cartesian_tangent.shape
        not in ((batch_size, 6), (batch_size, sample_count, 6))
    ):
        raise ValueError("Path derivative tensors have incompatible shapes.")
    if sample_count < 3:
        raise ValueError("At least three path samples are required.")
    if bool((torch.diff(path_positions, dim=1) <= 0.0).any().item()):
        raise ValueError("Path positions must be strictly increasing.")

    if cartesian_curvature is None:
        jacobian_pinv = torch.linalg.pinv(jacobians)
    else:
        damping = 1e-4
        identity = torch.eye(6, dtype=jacobians.dtype, device=jacobians.device)
        identity = identity.expand(batch_size, sample_count, -1, -1)
        normal = (
            torch.matmul(jacobians, jacobians.transpose(-1, -2)) + damping**2 * identity
        )
        jacobian_pinv = torch.matmul(
            jacobians.transpose(-1, -2), torch.linalg.solve(normal, identity)
        )
    if cartesian_tangent.dim() == 2:
        tangent = cartesian_tangent[:, None].expand(-1, sample_count, -1)
    else:
        tangent = cartesian_tangent
    tangent = tangent[..., None]
    q_s = torch.matmul(jacobian_pinv, tangent).squeeze(-1)
    jacobian_s_rows: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        jacobian_s_rows.append(
            torch.gradient(
                jacobians[batch_index],
                spacing=(path_positions[batch_index],),
                dim=(0,),
                edge_order=2,
            )[0]
        )
    jacobian_s = torch.stack(jacobian_s_rows)
    jacobian_term = torch.matmul(jacobian_s, q_s.unsqueeze(-1))
    if cartesian_curvature is None:
        curvature = torch.zeros_like(jacobian_term)
    else:
        if cartesian_curvature.shape != (batch_size, sample_count, 6):
            raise ValueError("Cartesian curvature must have shape (B, N, 6).")
        curvature = cartesian_curvature[..., None]
    q_ss = torch.matmul(jacobian_pinv, curvature - jacobian_term).squeeze(-1)
    velocities = q_s * path_velocities[..., None]
    accelerations = (
        q_ss * path_velocities.square()[..., None] + q_s * path_accelerations[..., None]
    )
    if velocities.shape != (batch_size, sample_count, dof):
        raise RuntimeError("Unexpected joint derivative shape.")
    return velocities, accelerations


def nearest_equivalent_joint_solution(
    solution: torch.Tensor,
    seed: torch.Tensor,
    limits: torch.Tensor,
) -> torch.Tensor:
    """Select the limit-valid ``2π`` equivalent closest to the previous seed.

    Args:
        solution: Joint solutions in radians.
        seed: Reference joints with the same shape as ``solution``.
        limits: Lower/upper limits with shape ``(*solution.shape, 2)``.

    Returns:
        Nearest equivalent where valid, otherwise the original joint solution.
    """
    if solution.shape != seed.shape or limits.shape != (*solution.shape, 2):
        raise ValueError("solution, seed, and limits have incompatible shapes.")
    turns = torch.round((seed - solution) / (2.0 * torch.pi))
    nearest = solution + turns * (2.0 * torch.pi)
    valid = (nearest >= limits[..., 0]) & (nearest <= limits[..., 1])
    return torch.where(valid, nearest, solution)
