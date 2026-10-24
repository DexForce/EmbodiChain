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

"""Task-owned PickPlace target semantics for the pinned PR828 BC recipe.

Geometry, scaling and normalization follow PR828 commit 51f1f0c. Helpers in
this experiment module are deliberately not part of the generic BC model API.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from embodichain.data_pipeline.datasets.bc_hdf5 import BCWindowDataset
from embodichain.utils.math import (
    axis_angle_from_quat,
    axis_angle_to_rotation_matrix,
    quat_from_matrix,
)

CAMERAS = ("cam_high", "cam_wrist")
TARGET_SEMANTICS = "arena_tcp_position_rotation6d_parallel_gripper@1"


def rotation_from_6d(value: torch.Tensor) -> torch.Tensor:
    first = value[..., :3]
    default = torch.zeros_like(first)
    default[..., 0] = 1
    first = F.normalize(
        torch.where(first.norm(dim=-1, keepdim=True) < 1e-6, default, first), dim=-1
    )
    second = value[..., 3:] - (first * value[..., 3:]).sum(-1, keepdim=True) * first
    alternate = F.one_hot(first.abs().argmin(-1), 3).to(value)
    alternate = alternate - (alternate * first).sum(-1, keepdim=True) * first
    second = F.normalize(
        torch.where(second.norm(dim=-1, keepdim=True) < 1e-6, alternate, second), dim=-1
    )
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), -1)


def pose_targets(geometry: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Recover absolute commands using their aligned pre-action TCP poses."""
    if (
        geometry.shape[:-1] != actions.shape[:-1]
        or geometry.shape[-1] != 51
        or actions.shape[-1] != 7
    ):
        raise ValueError(
            "PickPlace target encoding requires aligned geometry51/actions7"
        )
    rotation = axis_angle_to_rotation_matrix(
        actions[..., 3:6] * 0.1
    ) @ rotation_from_6d(geometry[..., 21:27])
    return torch.cat(
        (
            geometry[..., 18:21] + actions[..., :3] * 0.02,
            rotation[..., :, :2].transpose(-1, -2).flatten(-2),
            actions[..., 6:],
        ),
        -1,
    )


def encode_targets(fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return pose_targets(fields["geometry"], fields["actions"])


def decode_request(target: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
    rotation_error = rotation_from_6d(target[..., 3:9]) @ rotation_from_6d(
        geometry[..., 21:27]
    ).transpose(-1, -2)
    return torch.cat(
        (
            (target[..., :3] - geometry[..., 18:21]) / 0.02,
            axis_angle_from_quat(quat_from_matrix(rotation_error)) / 0.1,
            target[..., 9:],
        ),
        -1,
    ).clamp(-1, 1)


def source(manifest_path: str | Path, *, max_open_files: int = 32) -> BCWindowDataset:
    """Bind the fixed corpus to the pinned camera and action contracts."""
    manifest = json.loads(Path(manifest_path).read_text())
    if (
        tuple(manifest["cameras"]) != CAMERAS
        or manifest.get("camera_profile_version") != "pick-place-v2"
        or manifest.get("action_semantics") != "policy_request"
        or manifest.get("action_contracts")
        != ["eef_pose.relative@1", "gripper.parallel@1"]
        or manifest.get("step_dt") != 1 / 30
    ):
        raise ValueError(
            "Offline corpus differs from the pinned PickPlace input/action recipe"
        )
    return BCWindowDataset(
        manifest_path,
        history_length=2,
        prediction_horizon=32,
        observation_key="geometry",
        observation_slice=slice(0, 27),
        target_encoder=encode_targets,
        target_fields=("geometry", "actions"),
        target_encoder_id=TARGET_SEMANTICS + ":relative-scales-0.02m-0.1rad@1",
        max_open_files=max_open_files,
    )


def frozen_statistics(
    dataset: BCWindowDataset, episodes: int = 32
) -> dict[str, torch.Tensor]:
    """Compute the PR828 floors and means from the first training episodes."""
    if len(dataset.episodes) < episodes:
        raise ValueError("Normalization requires the declared initial episode panel")
    geometry, targets = [], []
    for ordinal in range(episodes):
        numeric = dataset.read_episode(ordinal, ("geometry", "actions"))
        measured = numeric["geometry"].float()
        command = numeric["actions"].float()
        geometry.append(measured)
        targets.append(pose_targets(measured, command))
    measured, target = torch.cat(geometry), torch.cat(targets)
    floors = measured.new_tensor(
        [0.1] * 7 + [0.02] * 2 + [0.5] * 7 + [0.02] * 2 + [0.05] * 3 + [1.0] * 6
    )
    mean = measured[:, :27].mean(0)
    std = measured[:, :27].std(0, correction=0).clamp_min(floors)
    mean[21:27], std[21:27] = 0, 1
    target_mean = target.mean(0)
    target_std = target.std(0, correction=0).clamp_min(0.05)
    target_mean[3:], target_std[3:] = 0, 1
    return {
        "observation_mean": mean,
        "observation_std": std,
        "target_mean": target_mean,
        "target_std": target_std,
    }
