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

from __future__ import annotations

import numpy as np
import open3d as o3d
import torch

from embodichain.utils import configclass
from embodichain.utils.math import transform_points_mat
from .collision_checker import ConvexCollisionChecker
from .fixed_geometry import FixedGripperCollisionGeometry

__all__ = ["GripperCollisionCfg", "GripperCollisionChecker"]


@configclass
class GripperCollisionCfg:
    """Target-object decomposition settings, independent of gripper geometry."""

    max_decomposition_hulls: int = 16
    """Maximum number of convex hulls used to represent the target object."""


class GripperCollisionChecker:
    """Check actual gripper surfaces at their declared pre-grasp joint state."""

    def __init__(
        self,
        object_mesh_verts: torch.Tensor,
        object_mesh_faces: torch.Tensor,
        *,
        collision_geometry: FixedGripperCollisionGeometry,
        cfg: GripperCollisionCfg | None = None,
    ) -> None:
        if not isinstance(collision_geometry, FixedGripperCollisionGeometry):
            raise TypeError("collision_geometry must be explicit fixed mesh geometry.")
        self.cfg = GripperCollisionCfg() if cfg is None else cfg
        self._checker = ConvexCollisionChecker(
            base_mesh_verts=object_mesh_verts,
            base_mesh_faces=object_mesh_faces,
            max_decomposition_hulls=self.cfg.max_decomposition_hulls,
        )
        self.obj_mesh_verts = object_mesh_verts
        self.device = object_mesh_verts.device
        self.surface_points = collision_geometry.points.to(self.device).clone()

    def get_ground_height(self, obj_pose: torch.Tensor) -> float:
        """Return the inferred support plane at the object's lowest vertex."""
        vertices = self.obj_mesh_verts.to(dtype=obj_pose.dtype)
        world = vertices @ obj_pose[:3, :3].T + obj_pose[:3, 3]
        return world[:, 2].min().item()

    def query(
        self,
        obj_pose: torch.Tensor,
        grasp_poses: torch.Tensor,
        *,
        collision_threshold: float = 0.0,
        is_filter_ground_collision: bool = True,
        is_visual: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce finite mesh-surface queries at canonical grasp poses.

        Args:
            obj_pose: World pose of the target mesh, shape ``(4, 4)``.
            grasp_poses: World canonical contact poses, shape ``(B, 4, 4)``.
                The supplied geometry already includes the contact/TCP mapping.
            collision_threshold: Signed-distance threshold in metres.
            is_filter_ground_collision: Include the inferred support plane.
            is_visual: Display each pose's object-relative collision samples.

        Returns:
            Per-pose collision mask and minimum signed distance, both ``(B,)``.
            Finite surface samples do not certify continuous mesh intersection,
            swept motion, closing contacts, or full robot collision freedom.
        """
        return self._query_chunks(
            obj_pose,
            grasp_poses,
            collision_threshold=collision_threshold,
            is_filter_ground_collision=is_filter_ground_collision,
            is_visual=is_visual,
        )

    def _query_chunks(
        self,
        obj_pose: torch.Tensor,
        grasp_poses: torch.Tensor,
        *,
        collision_threshold: float,
        is_filter_ground_collision: bool,
        is_visual: bool,
        candidate_chunk_size: int = 128,
        point_chunk_size: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce all samples with bounded candidate/point/hull intermediates."""
        relative = torch.linalg.inv(obj_pose).unsqueeze(0) @ grasp_poses
        count = len(grasp_poses)
        collided = torch.zeros(count, dtype=torch.bool, device=self.device)
        distance = torch.full(
            (count,), float("inf"), device=self.device, dtype=grasp_poses.dtype
        )
        ground = (
            self.get_ground_height(obj_pose) if is_filter_ground_collision else None
        )
        # Visualization holds at most one pose's surface and colors.
        batch_size = 1 if is_visual else candidate_chunk_size
        samples = self.surface_points.to(dtype=grasp_poses.dtype)
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            display_points, display_flags = [], []
            for points in samples.split(point_chunk_size):
                local = transform_points_mat(points, relative[start:end])
                flags, distances = self._checker.query_batch_points(
                    local, collision_threshold=collision_threshold
                )
                if ground is not None:
                    world = transform_points_mat(points, grasp_poses[start:end])
                    ground_distance = world[:, :, 2] - ground
                    flags = flags | (ground_distance < collision_threshold)
                    distances = torch.minimum(distances, ground_distance)
                collided[start:end] |= flags.any(dim=1)
                distance[start:end] = torch.minimum(
                    distance[start:end], distances.min(dim=1).values
                )
                if is_visual:
                    display_points.append(local[0].cpu().numpy())
                    display_flags.append(flags[0].cpu().numpy())
            if is_visual:
                cloud = o3d.geometry.PointCloud()
                displayed = np.concatenate(display_points)
                flags_np = np.concatenate(display_flags)
                cloud.points = o3d.utility.Vector3dVector(displayed)
                colors = np.zeros_like(displayed)
                colors[flags_np, 0] = 1
                colors[~flags_np, 1] = 1
                cloud.colors = o3d.utility.Vector3dVector(colors)
                o3d.visualization.draw_geometries([self._checker.mesh, cloud])
        return collided, distance
