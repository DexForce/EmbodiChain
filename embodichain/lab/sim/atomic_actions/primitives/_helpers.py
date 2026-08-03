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

"""Shared internal helpers for concrete atomic actions."""

from __future__ import annotations

import math

import torch

from embodichain.utils import logger

from ..core import WorldState


def resolve_object_target(
    target: torch.Tensor,
    *,
    n_envs: int,
    device: torch.device,
    name: str = "object_target_pose",
) -> torch.Tensor:
    """Broadcast an object target pose to ``(n_envs, 4, 4)`` or validate it."""
    target = target.to(device=device, dtype=torch.float32)
    if target.shape == (4, 4):
        target = target.unsqueeze(0).repeat(n_envs, 1, 1)
    if target.shape != (n_envs, 4, 4):
        logger.log_error(
            f"{name} must be (4, 4) or ({n_envs}, 4, 4), but got {target.shape}",
            ValueError,
        )
    return target


def arm_qpos_from_state(state: WorldState, arm_joint_ids: list[int]) -> torch.Tensor:
    """Extract the arm slice of the full-DoF ``last_qpos`` carried in state."""
    return state.last_qpos[:, arm_joint_ids]


def upright_yaw_pose_variants(
    target_pose: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Return object poses with evenly sampled world-Z yaw rotations."""
    if sample_count < 1:
        logger.log_error("upright_yaw_samples must be positive.", ValueError)
    if target_pose.ndim != 3 or target_pose.shape[-2:] != (4, 4):
        logger.log_error("target_pose must have shape (N, 4, 4).", ValueError)

    signed_steps = [0]
    for step in range(1, (sample_count + 1) // 2):
        signed_steps.extend((step, -step))
    if sample_count % 2 == 0:
        signed_steps.append(sample_count // 2)
    angles = torch.tensor(
        signed_steps,
        dtype=target_pose.dtype,
        device=target_pose.device,
    ) * (2.0 * math.pi / sample_count)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    yaw = torch.zeros(
        (sample_count, 3, 3),
        dtype=target_pose.dtype,
        device=target_pose.device,
    )
    yaw[:, 0, 0] = cos
    yaw[:, 0, 1] = -sin
    yaw[:, 1, 0] = sin
    yaw[:, 1, 1] = cos
    yaw[:, 2, 2] = 1.0

    variants = target_pose[:, None].repeat(1, sample_count, 1, 1)
    variants[:, :, :3, :3] = torch.matmul(yaw[None], target_pose[:, None, :3, :3])
    return variants


__all__ = [
    "arm_qpos_from_state",
    "resolve_object_target",
    "upright_yaw_pose_variants",
]
