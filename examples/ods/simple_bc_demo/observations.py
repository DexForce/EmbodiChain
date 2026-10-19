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

"""Measured state features and frozen training-only normalization."""

from __future__ import annotations

from typing import Any
import torch
from torch.nn import functional as F


def policy_observations(obs: Any, mode: str = "geometry") -> torch.Tensor:
    """Pack current measured state only; never read demonstration annotations."""
    if mode != "geometry":
        raise ValueError("This demo uses the measured geometry observation contract")
    from embodichain.utils.math import matrix_from_quat

    tcp = obs["tcp_pose"]
    cube = obs["cube_pose"]
    quaternion = F.normalize(cube[..., 3:], dim=-1)
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1
    quaternion = torch.where(
        quaternion.norm(dim=-1, keepdim=True) == 0, identity, quaternion
    )
    rotation = matrix_from_quat(quaternion)
    return torch.cat(
        (
            obs["robot"]["qpos"],
            obs["robot"]["qvel"],
            tcp[..., :3, 3],
            tcp[..., :3, :2].transpose(-1, -2).flatten(-2),
            cube[..., :3],
            rotation[..., :3, :2].transpose(-1, -2).flatten(-2),
            obs["goal_position"],
            cube[..., :3] - tcp[..., :3, 3],
            obs["goal_position"] - cube[..., :3],
            obs["cube_velocity"],
        ),
        dim=-1,
    ).float()


def _geometry_statistics(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    floors = values.new_tensor(
        [0.1] * 7
        + [0.02] * 2
        + [0.5] * 7
        + [0.02] * 2
        + [0.05] * 3
        + [1.0] * 6
        + [0.05] * 3
        + [1.0] * 6
        + [0.1] * 3
        + [0.05] * 6
        + [0.1] * 3
        + [0.5] * 3
    )
    mean, std = values.mean(0), values.std(0, correction=0).clamp_min(floors)
    mean[21:27], std[21:27] = 0, 1
    mean[30:36], std[30:36] = 0, 1
    return mean, std
