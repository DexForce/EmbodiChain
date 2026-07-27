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

"""Small tensor and geometry helpers shared by pose and grasp modules."""

from __future__ import annotations

import torch

from embodichain.utils.math import pose_inv

__all__ = [
    "_ensure_batched_pose_tensor",
    "_normalize_vector",
    "_orthogonalized_axis",
    "_transform_local_point",
    "_current_arm_positions",
    "_pose_from_axes",
    "_world_pose_to_object_pose",
    "_world_bounds_from_local_vertices",
    "_world_vertices_from_local_vertices",
    "_object_world_vertices",
    "_object_mesh_vertices",
]


def _ensure_batched_pose_tensor(pose, device) -> torch.Tensor:
    """Ensure a pose tensor has shape (n_envs, 4, 4)."""
    pose = torch.as_tensor(pose, dtype=torch.float32, device=device)
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    if pose.ndim != 3 or pose.shape[-2:] != (4, 4):
        raise ValueError(
            "Batched pose target must have shape (4, 4) or (n_envs, 4, 4), "
            f"got {tuple(pose.shape)}."
        )
    return pose.clone()


def _normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(vector)
    if float(norm) < 1e-6:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _orthogonalized_axis(axis: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    axis = axis - torch.dot(axis, reference) * reference
    if float(torch.linalg.norm(axis)) < 1e-6:
        fallback = torch.tensor([1.0, 0.0, 0.0], device=reference.device)
        if float(torch.abs(torch.dot(fallback, reference))) > 0.9:
            fallback = torch.tensor([0.0, 1.0, 0.0], device=reference.device)
        axis = fallback - torch.dot(fallback, reference) * reference
    return _normalize_vector(axis)


def _transform_local_point(object_pose: torch.Tensor, local_point: torch.Tensor):
    homogeneous = torch.cat(
        [
            local_point.to(device=object_pose.device, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32, device=object_pose.device),
        ]
    )
    return (object_pose @ homogeneous)[:3]


def _current_arm_positions(
    env, device, env_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if env is None or not hasattr(env, "get_current_xpos_agent"):
        return None
    try:
        left_pose, right_pose = env.get_current_xpos_agent()
        left_pose = _ensure_batched_pose_tensor(left_pose, device)
        right_pose = _ensure_batched_pose_tensor(right_pose, device)
    except Exception:
        return None
    return left_pose[env_id, :3, 3], right_pose[env_id, :3, 3]


def _pose_from_axes(
    *,
    position: torch.Tensor,
    x_axis: torch.Tensor,
    y_axis: torch.Tensor,
    z_axis: torch.Tensor,
) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32, device=position.device)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = position
    return pose


def _world_pose_to_object_pose(
    object_pose: torch.Tensor,
    world_pose: torch.Tensor,
) -> torch.Tensor:
    return pose_inv(object_pose) @ world_pose


def _world_bounds_from_local_vertices(
    object_pose: torch.Tensor,
    vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_vertices = _world_vertices_from_local_vertices(object_pose, vertices)
    return world_vertices.min(dim=0).values, world_vertices.max(dim=0).values


def _world_vertices_from_local_vertices(
    object_pose: torch.Tensor,
    vertices: torch.Tensor,
) -> torch.Tensor:
    return (object_pose[:3, :3] @ vertices.T).T + object_pose[:3, 3]


def _object_world_vertices(obj, device, env_id: int = 0) -> torch.Tensor:
    vertices = _object_mesh_vertices(obj, device, env_id=env_id)
    pose = _ensure_batched_pose_tensor(obj.get_local_pose(to_matrix=True), device)
    return (pose[env_id, :3, :3] @ vertices.T).T + pose[env_id, :3, 3]


def _object_mesh_vertices(obj, device, env_id: int = 0) -> torch.Tensor:
    vertices = obj.get_vertices(env_ids=[env_id], scale=True)
    if isinstance(vertices, (list, tuple)):
        vertices = vertices[0]
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    if vertices.ndim == 3 and vertices.shape[0] == 1:
        vertices = vertices.squeeze(0)
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
        raise ValueError("Object mesh vertices must have shape (N, 3).")
    return vertices
