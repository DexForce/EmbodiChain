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

from contextlib import contextmanager, nullcontext
from collections.abc import Iterator, Callable

import numpy as np
import open3d as o3d
import torch

from embodichain.utils import configclass
from embodichain.utils.math import transform_points_mat
from .collision_checker import ConvexCollisionChecker
from .fixed_geometry import FixedGripperCollisionGeometry

__all__ = ["GripperCollisionCfg", "GripperCollisionChecker"]


_NATIVE_POINT_QUERY = ConvexCollisionChecker.query_batch_points


@contextmanager
def _cpu_query_device(
    checker: ConvexCollisionChecker,
) -> Iterator[tuple[torch.device, Callable]]:
    """Keep a query's existing CUDA computation resident with bounded scratch.

    CPU transforms, every surface sample and the signed-distance kernel are
    unchanged. Inputs are copied afresh per query; no mutable geometry is cached
    across calls. Scratch is released on its owning stream before scope exit.
    """
    import warp as wp
    from embodichain.compute.geometry._warp.convex_query import (
        convex_signed_distance_kernel,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    planes = checker.plane_equations["plane_equations"].to(device).clone()
    counts = checker.plane_equations["plane_equation_counts"].to(device).clone()
    workspace = None

    def query(
        points: torch.Tensor, *, collision_threshold: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal workspace
        gpu_points = points.to(device)
        shape = (*points.shape[:2], planes.shape[0])
        if workspace is None or workspace.shape != shape:
            workspace = None
            workspace = wp.empty(shape, dtype=wp.float32, device=str(device))
        workspace.fill_(-float("inf"))
        wp.launch(
            convex_signed_distance_kernel,
            dim=shape,
            inputs=(
                wp.from_torch(gpu_points),
                wp.from_torch(planes),
                wp.from_torch(counts),
            ),
            outputs=(workspace,),
            device=str(device),
        )
        distance = wp.to_torch(workspace).min(dim=-1).values
        return distance <= collision_threshold, distance

    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device))):
        try:
            yield device, query
        finally:
            workspace = None


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

        CPU inputs retain CPU transforms and the existing CUDA classification.
        Native multi-chunk queries keep geometry and reductions on that device
        for one call; every original sample and chunk bound is preserved.

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
        ground = (
            self.get_ground_height(obj_pose) if is_filter_ground_collision else None
        )
        # Visualization holds at most one pose's surface and colors.
        batch_size = 1 if is_visual else candidate_chunk_size
        samples = self.surface_points.to(dtype=grasp_poses.dtype)
        native_cpu = (
            count > 0
            and (
                count > candidate_chunk_size
                or len(self.surface_points) > point_chunk_size
            )
            and self.device.type == "cpu"
            and not is_visual
            and type(self._checker) is ConvexCollisionChecker
            and getattr(self._checker.query_batch_points, "__func__", None)
            is _NATIVE_POINT_QUERY
        )
        scope = (
            _cpu_query_device(self._checker)
            if native_cpu
            else nullcontext((self.device, self._checker.query_batch_points))
        )
        with scope as (query_device, point_query):
            collided = torch.zeros(count, dtype=torch.bool, device=query_device)
            distance = torch.full(
                (count,), float("inf"), device=query_device, dtype=grasp_poses.dtype
            )
            for start in range(0, count, batch_size):
                end = min(start + batch_size, count)
                display_points, display_flags = [], []
                for points in samples.split(point_chunk_size):
                    local = transform_points_mat(points, relative[start:end])
                    flags, distances = point_query(
                        local, collision_threshold=collision_threshold
                    )
                    if ground is not None:
                        world = transform_points_mat(points, grasp_poses[start:end])
                        ground_distance = (world[:, :, 2] - ground).to(query_device)
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
            return collided.to(self.device), distance.to(self.device)
