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

"""Time resampling of floating-root motion references."""

from __future__ import annotations
import math
import torch
from embodichain.utils.math import (
    matrix_from_quat,
    quat_from_matrix,
    axis_angle_from_quat,
)

__all__ = ["resample_floating_trajectory"]


def resample_floating_trajectory(
    qpos: torch.Tensor, source_fps: float, target_fps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample one root/joint clip and estimate world root-origin velocities.

    Args:
        qpos: ``(frames, 7 + joints)`` positions with root XYZ and XYZW quaternion.
        source_fps: Uniform input sample frequency.
        target_fps: Uniform output frequency. The final sample is the last target
            grid point within the input duration; no extra endpoint is appended.

    Returns:
        Resampled qpos and ``[world linear, world angular, joint velocity]`` qvel.
        Translation and joints use linear interpolation; orientation uses shortest
        arc SLERP. Velocities use centered differences and one-sided endpoints.
    """
    if any(not math.isfinite(f) or f <= 0 for f in (source_fps, target_fps)):
        raise ValueError("Motion sample frequencies must be finite and positive")
    if (
        qpos.ndim != 2
        or qpos.shape[0] < 2
        or qpos.shape[1] < 8
        or not qpos.is_floating_point()
        or not bool(torch.isfinite(qpos).all())
    ):
        raise ValueError(
            "Motion positions require finite (frames>=2, 7+joints) samples"
        )
    if bool((qpos[:, 3:7].norm(dim=-1) < 1e-8).any()):
        raise ValueError("Root quaternions must be nonzero")
    dt = 1.0 / target_fps
    intervals = (qpos.shape[0] - 1) * target_fps / source_fps
    # Include an endpoint on the target grid despite one-ULP division rounding.
    count = math.floor(math.nextafter(intervals, math.inf)) + 1
    if count < 2:
        raise ValueError("Target grid must contain at least two frames")
    t = torch.arange(count, device=qpos.device, dtype=qpos.dtype) * dt * source_fps
    lower = t.floor().long().clamp(max=qpos.shape[0] - 2)
    alpha = (t - lower.to(t.dtype)).clamp(0, 1).unsqueeze(-1)
    start, end = qpos[lower], qpos[lower + 1]
    linear = (1 - alpha) * start + alpha * end
    qa, qb = start[:, 3:7], end[:, 3:7]
    dot = (qa * qb).sum(-1, keepdim=True)
    qb = torch.where(dot < 0, -qb, qb)
    dot = dot.abs().clamp(max=1.0)
    small = dot > 1.0 - 1e-7
    angle = torch.acos(torch.where(small, torch.zeros_like(dot), dot))
    denominator = angle.sin()
    wa = torch.where(small, 1 - alpha, ((1 - alpha) * angle).sin() / denominator)
    wb = torch.where(small, alpha, (alpha * angle).sin() / denominator)
    quat = torch.nn.functional.normalize(wa * qa + wb * qb, dim=-1)
    sampled = torch.cat((linear[:, :3], quat, linear[:, 7:]), -1)
    translation_joints = torch.cat((sampled[:, :3], sampled[:, 7:]), -1)
    velocity = torch.gradient(translation_joints, spacing=dt, dim=0)[0]
    rotation = matrix_from_quat(quat)
    before = torch.cat((rotation[:1], rotation[:-2], rotation[-2:-1]), 0)
    after = torch.cat((rotation[1:2], rotation[2:], rotation[-1:]), 0)
    span = sampled.new_full((count, 1), 2 * dt)
    span[0] = dt
    span[-1] = dt
    angular = (
        axis_angle_from_quat(quat_from_matrix(after @ before.transpose(-1, -2))) / span
    )
    return sampled, torch.cat((velocity[:, :3], angular, velocity[:, 3:]), -1)
