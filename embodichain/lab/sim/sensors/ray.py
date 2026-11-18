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

"""Static collision ray queries and renderer-independent terrain scans."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING, Literal

import torch

from embodichain.compute.geometry.raycast import TriangleRayMesh, cast_rays
from embodichain.utils import configclass

from .base_sensor import SensorCfg
from .state import _ReadOnlySensor

if TYPE_CHECKING:
    from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc
    from embodichain.lab.sim.sim_manager import SimulationManager
    from .context import SensorContext

__all__ = ["RayScene", "TerrainScan", "TerrainScanCfg"]


class RayScene:
    """Topology-scoped static collision query with one geometry set per batch."""

    def __init__(self, meshes: Sequence[tuple[TriangleRayMesh, torch.Tensor]]) -> None:
        self._meshes = tuple(meshes)

    def cast(
        self, origins: torch.Tensor, directions: torch.Tensor, max_distance: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return arena-frame nearest distances and collision shape indices.

        Args:
            origins: Arena-frame ray origins, shape (B, R, 3).
            directions: Arena-frame ray directions, shape (B, R, 3).
            max_distance: Maximum distance in metres.

        Returns:
            Distances and shape indices. Misses are infinity and -1.
        """
        return cast_rays(origins, directions, self._meshes, max_distance)


def _collision_mesh(shape: CollisionShapeDesc, device: torch.device) -> TriangleRayMesh:
    if shape.vertices is not None and shape.triangles is not None:
        return TriangleRayMesh(shape.vertices.to(device), shape.triangles.to(device))
    kind = shape.shape_type.name
    if kind in ("CUBE", "BOX") and shape.half_extents is not None:
        vertices = torch.tensor(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            device=device,
        ) * shape.half_extents.to(device)
        faces = torch.tensor(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            device=device,
        )
        return TriangleRayMesh(vertices, faces)

    if kind in ("SPHERE", "CAPSULE") and shape.radius is not None:
        try:
            import trimesh
        except ImportError as error:
            raise NotImplementedError(
                f"{kind} ray geometry requires the optional geometry extra."
            ) from error
        if kind == "SPHERE":
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=shape.radius)
        elif shape.half_height is not None:
            mesh = trimesh.creation.capsule(
                radius=shape.radius, height=shape.half_height * 2
            )
        else:
            raise ValueError(f"Collision capsule {shape.name!r} has no half-height.")
        return TriangleRayMesh(
            torch.tensor(mesh.vertices, device=device),
            torch.tensor(mesh.faces, device=device),
        )
    raise NotImplementedError(
        f"No ray geometry adapter for collision shape {shape.name!r} ({kind})."
    )


def _build_collision_scene(
    manager: SimulationManager, targets: tuple[str, ...]
) -> RayScene:
    device = manager.device
    meshes = []
    for uid in targets:
        if uid == "default_plane":
            plane = manager._spawn_default_plane_descriptor.collisions[0]
            half_length = float(plane.size[0]) * 0.5
            vertices = torch.tensor(
                [
                    [-half_length, -half_length, 0],
                    [half_length, -half_length, 0],
                    [half_length, half_length, 0],
                    [-half_length, half_length, 0],
                ],
                device=device,
            )
            faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device)
            meshes.append(
                (TriangleRayMesh(vertices, faces), torch.eye(4, device=device)[None])
            )
            continue
        asset = manager._rigid_objects.get(uid)
        if asset is None:
            raise ValueError(f"Unknown rigid collision ray target {uid!r}.")
        if asset.cfg.body_type != "static":
            raise ValueError(f"Static ray target {uid!r} must have body_type='static'.")
        pose = asset.get_local_pose(to_matrix=True).to(
            device=device, dtype=torch.float32
        )
        for shape in asset.get_collision_shapes():
            meshes.append(
                (
                    _collision_mesh(shape, device),
                    pose @ shape.local_pose.to(device=device, dtype=torch.float32),
                )
            )
    return RayScene(meshes)


@configclass
class TerrainScanCfg(SensorCfg):
    """Sample terrain with a local XY grid and downward -Z rays."""

    sensor_type: str = "TerrainScan"
    sample_phase: Literal["physics"] = "physics"
    mount: SensorCfg.OffsetCfg = SensorCfg.OffsetCfg()
    target_uids: tuple[str, ...] = ("default_plane",)
    size: tuple[float, float] = (1.0, 1.0)
    resolution: float = 0.25
    max_distance: float = 20.0

    def get_data_types(self) -> list[str]:
        """Return distance, hit position, shape identity and validity fields."""
        return ["distance", "position", "shape_id", "hit"]


class TerrainScan(_ReadOnlySensor):
    """Measure the first terrain hit for each probe in a regular grid.

    Args:
        config: Scan grid, mount, collision targets and sample schedule.
        context: Manager-owned ray provider and time services.
    """

    def __init__(self, config: TerrainScanCfg, *, context: SensorContext) -> None:
        super().__init__(config, context=context)

    def _local_rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not math.isfinite(self.cfg.resolution)
            or self.cfg.resolution <= 0
            or any(not math.isfinite(value) or value < 0 for value in self.cfg.size)
        ):
            raise ValueError("Scan size must be nonnegative and resolution positive.")
        axes = [
            torch.linspace(
                -length / 2,
                length / 2,
                max(1, math.ceil(length / self.cfg.resolution) + 1),
                device=self.device,
            )
            for length in self.cfg.size
        ]
        x, y = torch.meshgrid(*axes, indexing="ij")
        starts = torch.stack(
            (x.flatten(), y.flatten(), torch.zeros_like(x.flatten())), dim=-1
        )
        directions = torch.zeros_like(starts)
        directions[:, 2] = -1
        return starts, directions

    def on_bind(self) -> None:
        if self.context.ray_factory is None:
            raise NotImplementedError("A collision ray provider is required.")
        if not math.isfinite(self.cfg.max_distance) or self.cfg.max_distance <= 0:
            raise ValueError("max_distance must be positive and finite.")
        self._ray_scene = self.context.ray_factory(tuple(self.cfg.target_uids))
        self._starts, self._directions = self._local_rays()
        shape = (self.num_instances, len(self._starts))
        self._data_buffer["distance"] = torch.full(shape, torch.inf, device=self.device)
        self._data_buffer["position"] = torch.full(
            (*shape, 3), torch.nan, device=self.device
        )
        self._data_buffer["shape_id"] = torch.full(
            shape, -1, dtype=torch.int32, device=self.device
        )
        self._data_buffer["hit"] = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )

    def on_unbind(self) -> None:
        self._ray_scene = None

    def update(self, **kwargs: object) -> None:
        """Cast the mounted local probes and publish the closest collision hits."""
        pose = self.get_mount_pose()
        rotation_t = pose[:, :3, :3].transpose(-2, -1)
        origins = self._starts @ rotation_t + pose[:, None, :3, 3]
        directions = self._directions @ rotation_t
        distance, indices = self._ray_scene.cast(
            origins, directions, self.cfg.max_distance
        )
        hit = indices >= 0
        position = origins + directions * torch.where(hit, distance, 0)[..., None]
        self._data_buffer["distance"].copy_(distance)
        self._data_buffer["position"].copy_(
            torch.where(hit[..., None], position, torch.nan)
        )
        self._data_buffer["shape_id"].copy_(indices)
        self._data_buffer["hit"].copy_(hit)

    def get_data_descriptors(self) -> dict[str, dict[str, object]]:
        """Describe range and arena-frame position fields."""
        fields = super().get_data_descriptors()
        fields["distance"].update(unit="m", frame="ray")
        fields["position"].update(unit="m", frame="arena")
        return fields
