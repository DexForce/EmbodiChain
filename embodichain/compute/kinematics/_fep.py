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
"""Pure Torch numerical correction and selection for seven-axis FEP search.

The search policy follows HolisticMotion's FEPKinematics.cpp at ef044f52;
kinematic models and solver state are supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Callable
import torch

__all__ = []


def configuration(qpos: torch.Tensor) -> torch.Tensor:
    """Return shoulder/elbow/wrist signs and seventh-joint seed angle."""
    signs = torch.where(qpos[..., [1, 3, 5]] >= 0, 1.0, -1.0)
    return torch.cat((signs, qpos[..., 6:7]), dim=-1)


def wrapped_distance(qpos: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    """Return joint differences modulo a full revolution."""
    delta = qpos - seed
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def pose_error(target: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Return geometric position and rotation-vector error in the chain root."""
    from pytorch_kinematics.transforms import matrix_to_axis_angle

    rotation = target[..., :3, :3] @ actual[..., :3, :3].transpose(-1, -2)
    return torch.cat(
        (target[..., :3, 3] - actual[..., :3, 3], matrix_to_axis_angle(rotation)),
        dim=-1,
    )


def solve_seed(
    target: torch.Tensor,
    seed: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    fk: Callable[[torch.Tensor], torch.Tensor],
    jacobian: Callable[[torch.Tensor], torch.Tensor],
    *,
    max_iterations: int,
    damping: float,
    max_step: float,
    position_tolerance: float,
    rotation_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply batched bounded DLS; accept only FK-verified converged states."""
    qpos = seed.clamp(lower, upper).clone()
    eye = torch.eye(6, device=seed.device, dtype=seed.dtype)
    scales = seed.new_tensor([1.0, 0.5, 0.25, 0.125]).view(1, 4, 1)
    error = pose_error(target, fk(qpos))
    stalled = torch.zeros(len(seed), device=seed.device, dtype=torch.bool)
    for iteration in range(max_iterations + 1):
        valid = (error[:, :3].norm(dim=-1) <= position_tolerance) & (
            error[:, 3:].norm(dim=-1) <= rotation_tolerance
        )
        if iteration == max_iterations:
            break
        # One compaction, rather than repeated boolean indexing (each performs
        # a CUDA nonzero synchronization). Preserve converged rows exactly.
        indices = (~valid & ~stalled).nonzero().flatten()
        if indices.numel() == 0:
            break
        current = qpos.index_select(0, indices)
        residual = error.index_select(0, indices)
        targets = target.index_select(0, indices)
        jac = jacobian(current)
        dual, info = torch.linalg.solve_ex(
            jac @ jac.transpose(-1, -2) + damping**2 * eye,
            residual.unsqueeze(-1),
            check_errors=False,
        )
        step = (jac.transpose(-1, -2) @ dual).squeeze(-1)
        step = torch.where((info == 0)[:, None], step, 0.0)
        step *= max_step / step.abs().amax(dim=-1, keepdim=True).clamp_min(max_step)
        # Evaluate the same four backtracking scales in one kinematics batch.
        trials = (current[:, None] + scales * step[:, None]).clamp(lower, upper)
        poses = fk(trials.reshape(-1, seed.shape[-1])).reshape(-1, 4, 4, 4)
        trial_errors = pose_error(targets[:, None], poses)
        costs = trial_errors.square().sum(dim=-1)
        costs = costs.masked_fill(~torch.isfinite(costs), torch.inf)
        best_cost, best_index = costs.min(dim=1)
        best = trials.gather(
            1, best_index[:, None, None].expand(-1, 1, seed.shape[-1])
        ).squeeze(1)
        best_error = trial_errors.gather(
            1, best_index[:, None, None].expand(-1, 1, 6)
        ).squeeze(1)
        improved = best_cost < residual.square().sum(dim=-1)
        # With no accepted update the next deterministic iteration would
        # repeat identical work. Stop that row without marking it successful.
        stalled.index_copy_(0, indices, ~improved)
        qpos.index_copy_(0, indices, torch.where(improved[:, None], best, current))
        # Reuse the selected FK error on the next iteration. It was evaluated
        # at precisely these joint positions; no extra FK pass is needed.
        error.index_copy_(
            0, indices, torch.where(improved[:, None], best_error, residual)
        )
    valid &= torch.isfinite(qpos).all(dim=-1)
    valid &= ((qpos >= lower) & (qpos <= upper)).all(dim=-1)
    return valid, qpos
