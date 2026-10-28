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

"""Bounded, inference-only pose-component IK from a caller-owned FK map."""

from __future__ import annotations

from collections.abc import Callable
import math

import torch

from .dls import damped_least_squares_step
from .pose_error import pose_error

__all__: list[str] = []


def _solve_masked_ik(
    target: torch.Tensor,
    seed: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    forward: Callable[[torch.Tensor], torch.Tensor],
    modality: str,
    *,
    iterations: int = 96,
    restarts: int = 4,
    position_tolerance: float = 0.005,
    rotation_tolerance: float = 0.05,
    damping: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve active components without reading the unused target payload.

    ``forward`` maps ``(B, N, D)`` joints to ``(B, N, 4, 4)`` TCP poses.
    Finite-difference probes need O(B * restarts * D**2) storage. Seed, midpoint
    and fixed independent uniform restarts make this deterministic without
    accessing reference solutions or consuming the caller's random stream.
    """
    if modality not in {"position", "orientation", "pose"}:
        raise ValueError("Masked IK requires position, orientation, or pose.")
    if iterations < 1 or restarts < 1:
        raise ValueError("Masked IK iterations and restarts must be positive.")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (position_tolerance, rotation_tolerance, damping)
    ):
        raise ValueError(
            "Masked IK tolerances and damping must be finite and positive."
        )
    batch, dof = seed.shape
    clean_target = (
        torch.eye(4, device=seed.device, dtype=seed.dtype).expand(batch, 4, 4).clone()
    )
    if modality in {"position", "pose"}:
        clean_target[:, :3, 3] = target[:, :3, 3]
    if modality in {"orientation", "pose"}:
        clean_target[:, :3, :3] = target[:, :3, :3]
    if not bool(torch.isfinite(clean_target).all()):
        raise ValueError("Active masked IK target components must be finite.")
    lower, upper = lower.to(seed).expand_as(seed), upper.to(seed).expand_as(seed)
    generator = torch.Generator(device="cpu").manual_seed(0)
    random = torch.rand(batch, max(restarts - 2, 0), dof, generator=generator).to(seed)
    random = lower[:, None] + random * (upper - lower)[:, None]
    qpos = torch.cat((seed[:, None], ((lower + upper) * 0.5)[:, None], random), dim=1)[
        :, :restarts
    ].clone()
    epsilon = 1.0e-3
    probe_offsets = epsilon * torch.eye(dof, device=seed.device, dtype=seed.dtype)
    active = (
        slice(0, 3)
        if modality == "position"
        else (slice(3, 6) if modality == "orientation" else slice(0, 6))
    )
    for iteration in range(iterations + 1):
        probes = qpos[:, :, None].expand(-1, -1, dof + 1, -1).clone()
        probes[:, :, 1:] += probe_offsets
        poses = forward(probes.reshape(batch, restarts * (dof + 1), dof)).reshape(
            batch, restarts, dof + 1, 4, 4
        )
        current = poses[:, :, 0]
        position, rotation = pose_error(current, clean_target[:, None])
        solved = torch.isfinite(qpos).all(-1)
        if modality in {"position", "pose"}:
            solved &= position.norm(dim=-1) <= position_tolerance
        if modality in {"orientation", "pose"}:
            solved &= rotation.norm(dim=-1) <= rotation_tolerance
        if iteration == iterations or (
            iteration % 4 == 0 and bool(solved.any(-1).all())
        ):
            break
        linear, angular = pose_error(current[:, :, None], poses[:, :, 1:])
        jacobian = (
            torch.cat((linear / 0.3, angular), dim=-1).transpose(-1, -2) / epsilon
        )
        error = torch.cat((position / 0.3, rotation), dim=-1)
        delta = damped_least_squares_step(
            error[..., active], jacobian[..., active, :], damping
        )
        qpos = torch.where(
            solved[..., None],
            qpos,
            (qpos + delta.clamp(-0.25, 0.25)).clamp(lower[:, None], upper[:, None]),
        )
    distance = (qpos - seed[:, None]).square().sum(-1)
    selected = torch.where(solved, distance, torch.inf).argmin(-1)
    rows = torch.arange(batch, device=seed.device)
    success = solved[rows, selected]
    return success, torch.where(success[:, None], qpos[rows, selected], seed)
