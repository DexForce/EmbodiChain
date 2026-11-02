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

"""GenSim-owned center proposals, not source-qualified grasp certificates."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    AntipodalGraspPoseGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler import AntipodalSampler
from embodichain.utils.math import get_pc_center_box
from embodichain.utils.nms import pose_nms

__all__: list[str] = []


def _row(
    pairs: torch.Tensor,
    vertices: torch.Tensor,
    pose: torch.Tensor,
    direction: torch.Tensor,
    algorithm: AntipodalGraspPoseGeneratorCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    origin = pairs[:, 0] @ pose[:3, :3].T + pose[:3, 3]
    hit = pairs[:, 1] @ pose[:3, :3].T + pose[:3, 3]
    closing = F.normalize(hit - origin, dim=-1)
    angle = torch.acos((closing * direction).sum(-1).clamp(-1, 1)).abs()
    valid = (angle - torch.pi / 2).abs() <= algorithm.approach_deviation_angle
    if not valid.any():
        return torch.eye(4, device=pairs.device)[None], pairs.new_full((1,), torch.inf)
    centers = ((origin + hit) / 2)[valid]
    widths = torch.norm(origin[valid] - hit[valid], dim=-1)
    closing = closing[valid]
    directions = [direction]
    for _ in range(algorithm.approach_direction_samples - 1):
        directions.append(
            AntipodalSampler._random_rotate_unit_vectors(
                direction.unsqueeze(0), algorithm.approach_deviation_angle
            )[0]
        )
    poses = []
    for approach in directions:
        y = F.normalize(
            torch.cross(approach[None].repeat(len(closing), 1), closing, dim=-1), dim=-1
        )
        z = F.normalize(torch.cross(closing, y, dim=-1), dim=-1)
        candidate = torch.eye(4, device=pairs.device, dtype=torch.float32)[None].repeat(
            len(closing), 1, 1
        )
        candidate[:, :3, 0], candidate[:, :3, 1], candidate[:, :3, 2] = closing, y, z
        candidate[:, :3, 3] = centers
        poses.append(candidate)
    poses, keep = pose_nms(torch.vstack(poses), angle_th=math.pi / 36, dist_th=0.005)
    centers = centers.repeat(len(directions), 1)[keep]
    widths = widths.repeat(len(directions))[keep]
    mesh = vertices @ pose[:3, :3].T + pose[:3, 3]
    center = (
        get_pc_center_box(mesh) if algorithm.center_mode == "bounds" else mesh.mean(0)
    )
    axis = F.normalize(poses[:, :3, 0], dim=-1)
    angle = torch.acos((axis * direction).sum(-1).clamp(-1, 1)).abs()
    angle_cost = (angle - 0.5 * torch.pi).abs() / (0.5 * torch.pi)
    distance = torch.norm(centers - center, dim=-1)
    center_cost = distance / distance.max()
    length_cost = 1 - widths / widths.max()
    costs = 0.25 * angle_cost + 0.25 * length_cost + 0.5 * center_cost
    # The owning Pick applies its configured cap after final-TCP qualification.
    return poses, costs


def native_center_proposals(
    generator: AntipodalGraspPoseGenerator,
    *,
    mesh_vertices: torch.Tensor,
    mesh_triangles: torch.Tensor,
    obj_poses: torch.Tensor,
    approach_direction: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Reuse shared contact pairs while GenSim owns pose proposals and qualification.

    This center-only path cannot override ground or authored end-region policies.
    The caller qualifies final transformed TCP poses before applying its cap.
    """
    if generator.collision_cfg.filter_ground_collision:
        raise ValueError("Native center proposals cannot bypass ground filtering.")
    pairs = generator.prepare_mesh(
        mesh_vertices=mesh_vertices, mesh_triangles=mesh_triangles
    )
    if (
        obj_poses.ndim != 3
        or obj_poses.shape[1:] != (4, 4)
        or not len(obj_poses)
        or not torch.isfinite(obj_poses).all()
    ):
        raise ValueError(
            "Native proposals require finite nonempty (B, 4, 4) object poses."
        )
    poses = obj_poses.to(device=pairs.device, dtype=torch.float32)
    directions = approach_direction.to(device=pairs.device, dtype=torch.float32)
    if directions.shape == (3,):
        directions = directions[None].expand(len(poses), -1)
    if directions.shape != (len(poses), 3) or not torch.isfinite(directions).all():
        raise ValueError("Native proposals require finite (3,) or (B, 3) directions.")
    lengths = torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    if (lengths <= 1e-6).any():
        raise ValueError("Native proposal directions must be nonzero.")
    directions = directions / lengths
    algorithm = generator.algorithm_cfg
    return [
        _row(pairs, mesh_vertices, pose, direction, algorithm)
        for pose, direction in zip(poses, directions, strict=True)
    ]
