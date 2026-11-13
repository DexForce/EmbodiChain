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
    """Check actual gripper surfaces at their declared pre-grasp joint state.

    When the supplied geometry also owns closure surfaces, ``query`` can check
    support clearance at the blocked aperture of each candidate instead of the
    open command alone.
    """

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
        apertures = collision_geometry.closure_apertures
        closure = collision_geometry.closure_points
        self.closure_apertures = (
            None
            if apertures is None
            else apertures.to(device=self.device, dtype=object_mesh_verts.dtype)
        )
        self.closure_points = (
            None
            if closure is None
            else closure.to(device=self.device, dtype=object_mesh_verts.dtype)
        )

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
        filter_closure_support: bool = False,
        is_visual: bool = False,
        apertures: torch.Tensor | None = None,
        closure_grasp_poses: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce finite mesh-surface queries at canonical grasp poses.

        Args:
            obj_pose: World pose of the target mesh, shape ``(4, 4)``.
            grasp_poses: World canonical contact poses, shape ``(B, 4, 4)``.
                The supplied geometry already includes the contact/TCP mapping.
            collision_threshold: Signed-distance threshold in metres.
            is_filter_ground_collision: Check the open surfaces against the
                inferred support plane.
            filter_closure_support: Check pad surfaces against the inferred
                support plane at each candidate's blocked aperture. Requires
                closure samples and apertures.
            is_visual: Display each pose's object-relative collision samples.
            apertures: Optional blocked aperture per candidate in metres. When
                the geometry owns closure surfaces, support clearance is also
                qualified at the closing state that aperture reaches. States
                round down to the next sampled aperture, so a candidate is never
                credited with more clearance than its physical closure allows.
            closure_grasp_poses: Optional world canonical poses, shape
                ``(B, 4, 4)``, used only by the closure support check, for a
                caller that closes the hand at a different pose than the one
                it qualifies here. ``None`` checks closure at ``grasp_poses``.
                Ignored unless ``filter_closure_support`` is enabled.

        Returns:
            Per-pose collision mask and minimum signed distance, both ``(B,)``.
            Finite surface samples do not certify continuous mesh intersection,
            swept motion, closing contacts, or full robot collision freedom.
        """
        if apertures is not None and apertures.shape != (len(grasp_poses),):
            raise ValueError("Closure apertures must contain one value per candidate.")
        if (
            closure_grasp_poses is not None
            and closure_grasp_poses.shape != grasp_poses.shape
        ):
            raise ValueError("Closure grasp poses must align with grasp_poses.")
        if filter_closure_support and (
            apertures is None or self.closure_points is None
        ):
            raise ValueError(
                "Closure support filtering requires sampled apertures and closure surfaces."
            )
        return self._query_chunks(
            obj_pose,
            grasp_poses,
            collision_threshold=collision_threshold,
            is_filter_ground_collision=is_filter_ground_collision,
            filter_closure_support=filter_closure_support,
            is_visual=is_visual,
            apertures=apertures,
            closure_grasp_poses=closure_grasp_poses,
        )

    def _query_chunks(
        self,
        obj_pose: torch.Tensor,
        grasp_poses: torch.Tensor,
        *,
        collision_threshold: float,
        is_filter_ground_collision: bool,
        is_visual: bool,
        filter_closure_support: bool = False,
        apertures: torch.Tensor | None = None,
        closure_grasp_poses: torch.Tensor | None = None,
        candidate_chunk_size: int = 128,
        point_chunk_size: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce all samples with bounded candidate/point/hull intermediates."""
        closure_poses = (
            grasp_poses
            if closure_grasp_poses is None
            else closure_grasp_poses.to(grasp_poses)
        )
        relative = torch.linalg.inv(obj_pose).unsqueeze(0) @ grasp_poses
        count = len(grasp_poses)
        collided = torch.zeros(count, dtype=torch.bool, device=self.device)
        distance = torch.full(
            (count,), float("inf"), device=self.device, dtype=grasp_poses.dtype
        )
        ground = (
            self.get_ground_height(obj_pose)
            if (is_filter_ground_collision or filter_closure_support)
            else None
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
                if ground is not None and is_filter_ground_collision:
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
            if filter_closure_support and ground is not None and apertures is not None:
                closure_distance = self._closure_support_distance(
                    closure_poses[start:end], apertures[start:end], ground
                )
                if closure_distance is not None:
                    collided[start:end] |= closure_distance < collision_threshold
                    distance[start:end] = torch.minimum(
                        distance[start:end], closure_distance
                    )
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

    def _closure_support_distance(
        self,
        grasp_poses: torch.Tensor,
        apertures: torch.Tensor,
        ground: float,
    ) -> torch.Tensor | None:
        """Return each candidate's support clearance at its blocked aperture.

        The queried closing state is the sampled aperture at or below the
        blocked aperture, so a candidate is never credited with more clearance
        than its physical closure reaches.
        """
        if self.closure_points is None or self.closure_apertures is None:
            return None
        blocked = apertures.to(
            device=self.device, dtype=self.closure_apertures.dtype
        ).flatten()
        index = torch.searchsorted(self.closure_apertures, blocked, right=True) - 1
        index = index.clamp(min=0)
        points = self.closure_points[index].to(dtype=grasp_poses.dtype)
        height = (
            torch.einsum("bpk,bk->bp", points, grasp_poses[:, 2, :3])
            + grasp_poses[:, 2, 3:4]
        )
        return height.min(dim=1).values - ground
