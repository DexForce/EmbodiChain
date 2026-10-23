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

"""Independent ordered matching of heterogeneous semantic constraints."""

from __future__ import annotations

import math
from collections.abc import Sequence
import torch

from .trajectory import _pose_error_matrices


def match_constraints(
    qpos: torch.Tensor,
    poses: torch.Tensor,
    joint_goals: torch.Tensor,
    pose_goals: torch.Tensor,
    modalities: Sequence[str],
    *,
    position_threshold_m: float,
    rotation_threshold_rad: float,
    joint_threshold_rad: float,
) -> dict[str, object]:
    """Match each token at one strictly later native sample using only its payload.

    Pose requires simultaneous position and orientation. Joint uses maximum absolute per-joint error in radians.
    No orientation is implicitly added to Position, or position to Orientation.
    """
    thresholds = (position_threshold_m, rotation_threshold_rad, joint_threshold_rad)
    if any(not math.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("Constraint thresholds must be finite and positive.")
    count = len(modalities)
    if not count or pose_goals.shape[0] != count or joint_goals.shape[0] != count:
        raise ValueError("Constraint modalities must match nonempty goals.")
    if any(
        value not in {"joint", "position", "orientation", "pose"}
        for value in modalities
    ):
        raise ValueError("Unknown constraint modality.")
    pos_error, rot_error = _pose_error_matrices(pose_goals.to(poses), poses)
    joint_delta = joint_goals.to(qpos)[:, None] - qpos[None]
    joint_error = joint_delta.abs().amax(dim=-1)
    joint_l2_error = torch.linalg.vector_norm(joint_delta, dim=-1)
    arrivals, errors = [], []
    next_sample = 0
    for index, modality in enumerate(modalities):
        valid = torch.ones(qpos.shape[0], dtype=torch.bool, device=qpos.device)
        components = {}
        if modality in {"position", "pose"}:
            valid &= pos_error[index] <= position_threshold_m
            components["position_m"] = pos_error[index]
        if modality in {"orientation", "pose"}:
            valid &= rot_error[index] <= rotation_threshold_rad
            components["rotation_rad"] = rot_error[index]
        if modality == "joint":
            valid &= joint_error[index] <= joint_threshold_rad
            components["joint_max_rad"] = joint_error[index]
            components["joint_l2_rad"] = joint_l2_error[index]
        indices = torch.nonzero(valid[next_sample:], as_tuple=False).flatten()
        if not indices.numel():
            break
        sample = next_sample + int(indices[0])
        arrivals.append(sample)
        errors.append(
            {
                "modality": modality,
                **{name: float(value[sample]) for name, value in components.items()},
            }
        )
        next_sample = sample + 1
    return {
        "ordered_waypoints_reached": len(arrivals) == count,
        "completed_waypoint_ratio": len(arrivals) / count,
        "arrival_indices": arrivals,
        "arrival_errors": errors,
        "joint_error_norm": "linf",
        "sample_grid": "native",
    }
