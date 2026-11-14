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

from collections.abc import Callable, Sequence
import math
import weakref
from typing import TYPE_CHECKING, Literal

import torch

from embodichain.compute.geometry.raycast import TriangleRayMesh, cast_rays
from embodichain.utils import configclass

from .base_sensor import SensorCfg
from .state import _ReadOnlySensor

if TYPE_CHECKING:
    from dexsim.spawn import GeometryDesc
    from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc
    from embodichain.lab.sim.sim_manager import SimulationManager
    from .context import SensorContext

__all__ = [
    "RayScene",
    "TerrainScan",
    "TerrainScanCfg",
    "RayLidar",
    "RayLidarCfg",
    "RayDepthCamera",
    "RayDepthCameraCfg",
]


class RayScene:
    """Topology-scoped mesh queries with independently posed batch rows.

    Args:
        meshes: Local BVHs paired with fixed or callable transforms.
    """

    def __init__(
        self,
        meshes: Sequence[
            tuple[
                TriangleRayMesh,
                torch.Tensor
                | Callable[[], torch.Tensor]
                | tuple[Callable[[], torch.Tensor], torch.Tensor],
            ]
        ],
        *,
        topology_is_current: Callable[[], bool] | None = None,
    ) -> None:
        self._meshes = tuple(meshes)
        self._topology_is_current = topology_is_current

    def cast(
        self, origins: torch.Tensor, directions: torch.Tensor, max_distance: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return arena-frame nearest distances and mesh indices.

        Args:
            origins: Arena-frame ray origins, shape (B, R, 3).
            directions: Arena-frame ray directions, shape (B, R, 3).
            max_distance: Maximum distance in metres.

        Returns:
            Distances and shape indices. Misses are infinity and -1.
        """
        meshes = self._posed_meshes()
        return cast_rays(origins, directions, meshes, max_distance)

    def _check_topology(self) -> None:
        if self._topology_is_current is not None and not self._topology_is_current():
            raise RuntimeError(
                "Ray scene topology is stale. Reacquire the provider after prepare()."
            )

    def _posed_meshes(self) -> list[tuple[TriangleRayMesh, torch.Tensor]]:
        self._check_topology()
        poses = {}
        meshes = []
        for mesh, pose in self._meshes:
            if isinstance(pose, tuple):
                owner, offset = pose
                if owner not in poses:
                    poses[owner] = owner()
                pose = poses[owner] @ offset
            elif callable(pose):
                owner = pose
                if owner not in poses:
                    poses[owner] = owner()
                pose = poses[owner]
            meshes.append((mesh, pose))
        return meshes


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


def _authored_mesh(
    geometry: GeometryDesc, device: torch.device, scale: torch.Tensor
) -> TriangleRayMesh:
    from dexsim.spawn.adapters.common import generated_mesh

    if geometry.usd is not None or geometry.file_path:
        raise NotImplementedError(
            "Authored ray geometry currently requires a procedural or array-backed mesh."
        )
    vertices, faces, _, _ = generated_mesh(geometry)
    return TriangleRayMesh(
        torch.as_tensor(vertices, device=device) * scale,
        torch.as_tensor(faces, device=device),
    )


def _build_ray_scene(
    manager: SimulationManager,
    targets: tuple[str, ...],
    source: Literal["collision", "visual"],
) -> RayScene:
    if source not in ("collision", "visual"):
        raise ValueError(f"Unknown ray geometry source {source!r}.")
    device = manager.device
    meshes = []
    for uid in targets:
        if uid == "default_plane":
            descriptor = manager._spawn_scene.rigid_geometry_descriptor(uid)
            geometry = (
                descriptor.collisions[0]
                if source == "collision"
                else descriptor.renders[0]
            )
            meshes.append(
                (
                    _authored_mesh(geometry, device, torch.ones(3, device=device)),
                    torch.eye(4, device=device)[None],
                )
            )
            continue
        asset = manager._rigid_objects.get(uid)
        if source == "visual":
            visual_owner = (
                asset if asset is not None else manager.get_visual_object(uid)
            )
            if visual_owner is None:
                raise ValueError(f"Unknown rigid or visual object ray target {uid!r}.")
            descriptor = manager._spawn_scene.rigid_geometry_descriptor(uid)
            if descriptor.env_variants or not descriptor.renders:
                raise NotImplementedError(
                    f"Visual ray target {uid!r} requires shared authored triangle geometry."
                )
            pose_reader = _RayPose(visual_owner.get_local_pose, device)
            pose = (
                pose_reader() if asset is not None and asset.is_static else pose_reader
            )
            scale = torch.as_tensor(descriptor.body_scale, device=device)
            meshes.extend(
                (_authored_mesh(render, device, scale), pose)
                for render in descriptor.renders
            )
            continue
        if asset is not None:
            if manager._spawn_scene.rigid_geometry_descriptor(uid).env_variants:
                raise NotImplementedError(
                    f"Collision ray target {uid!r} requires shared cooked geometry."
                )
            pose_reader = _RayPose(asset.get_local_pose, device)
            static_pose = pose_reader() if asset.is_static else None
            shapes = asset.get_collision_shapes()
        else:
            asset_uid, sep, link = uid.partition("/")
            articulation = manager._robots.get(asset_uid) or manager._articulations.get(
                asset_uid
            )
            if not sep or articulation is None:
                raise ValueError(
                    f"Unknown rigid object or articulation link ray target {uid!r}."
                )
            pose_reader = _RayPose(
                lambda to_matrix, owner=articulation, name=link: owner.get_link_pose(
                    name, to_matrix=to_matrix
                ),
                device,
            )
            static_pose = None
            shapes = articulation.get_link_collision_shapes(link)
        for shape in shapes:
            local = shape.local_pose.to(device=device, dtype=torch.float32)
            pose = (
                static_pose @ local if static_pose is not None else (pose_reader, local)
            )
            meshes.append((_collision_mesh(shape, device), pose))
    manager_ref = weakref.ref(manager)
    topology = int(manager.spawn_result.topology_revision)

    def topology_is_current() -> bool:
        owner = manager_ref()
        if owner is None or getattr(owner, "_is_pending_kill", False):
            return False
        result = owner.spawn_result
        return (
            result is not None
            and not result.needs_rebuild
            and int(result.topology_revision) == topology
            and not owner._spawn_scene.builder.has_pending_changes
        )

    return RayScene(meshes, topology_is_current=topology_is_current)


class _RayPose:
    def __init__(self, fetch, device):
        self._fetch = fetch
        self._device = device

    def __call__(self):
        return self._fetch(to_matrix=True).to(device=self._device, dtype=torch.float32)


@configclass
class TerrainScanCfg(SensorCfg):
    """Sample terrain with a local XY grid and downward -Z rays."""

    sensor_type: str = "TerrainScan"
    sample_phase: Literal["physics"] = "physics"
    mount: SensorCfg.OffsetCfg = SensorCfg.OffsetCfg()
    target_uids: tuple[str, ...] = ("default_plane",)
    geometry_source: Literal["collision", "visual"] = "collision"
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
            raise NotImplementedError("A ray scene provider is required.")
        if not math.isfinite(self.cfg.max_distance) or self.cfg.max_distance <= 0:
            raise ValueError("max_distance must be positive and finite.")
        self._ray_scene = self.context.ray_factory(
            tuple(self.cfg.target_uids), self.cfg.geometry_source
        )
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
        """Cast the mounted local probes and publish the closest mesh hits."""
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


@configclass
class RayLidarCfg(TerrainScanCfg):
    """Geometric first-return LiDAR with explicit elevation channels."""

    sensor_type: str = "RayLidar"
    vertical_angles: tuple[float, ...] = (0.0,)
    horizontal_samples: int = 360
    horizontal_fov: float = 360.0


class RayLidar(TerrainScan):
    """Acquire a geometric snapshot, ordered by channel then azimuth."""

    def _local_rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if (
            type(cfg.horizontal_samples) is not int
            or cfg.horizontal_samples <= 0
            or not 0 < cfg.horizontal_fov <= 360
            or not cfg.vertical_angles
            or any(not -90 <= angle <= 90 for angle in cfg.vertical_angles)
        ):
            raise ValueError(
                "LiDAR scan angles and sample counts are outside their supported range."
            )
        az = (
            torch.arange(
                cfg.horizontal_samples, device=self.device, dtype=torch.float32
            )
            / cfg.horizontal_samples
            - 0.5
        ) * math.radians(cfg.horizontal_fov)
        el = torch.tensor(
            cfg.vertical_angles, device=self.device, dtype=torch.float32
        ).deg2rad()
        el, az = torch.meshgrid(el, az, indexing="ij")
        directions = torch.stack(
            (el.cos() * az.cos(), el.cos() * az.sin(), el.sin()), dim=-1
        ).reshape(-1, 3)
        return torch.zeros_like(directions), directions


@configclass
class RayDepthCameraCfg(TerrainScanCfg):
    """Renderer-independent pinhole sensor using optical +Z forward and +Y down."""

    sensor_type: str = "RayDepthCamera"
    width: int = 64
    height: int = 48
    fx: float = 60.0
    fy: float = 60.0
    cx: float | None = None
    cy: float | None = None

    def get_data_types(self) -> list[str]:
        """Include a (height, width) optical-axis depth image."""
        return [*super().get_data_types(), "depth"]


class RayDepthCamera(TerrainScan):
    """Publish radial range plus optical-axis pinhole depth."""

    def _local_rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if (
            type(cfg.width) is not int
            or type(cfg.height) is not int
            or min(cfg.width, cfg.height) < 1
        ):
            raise ValueError("Depth image dimensions must be positive integers.")
        if any(not math.isfinite(x) or x <= 0 for x in (cfg.fx, cfg.fy)):
            raise ValueError("Focal lengths must be positive and finite.")
        cx = (cfg.width - 1) * 0.5 if cfg.cx is None else cfg.cx
        cy = (cfg.height - 1) * 0.5 if cfg.cy is None else cfg.cy
        if not math.isfinite(cx) or not math.isfinite(cy):
            raise ValueError("Principal point must be finite.")
        y, x = torch.meshgrid(
            torch.arange(cfg.height, device=self.device, dtype=torch.float32),
            torch.arange(cfg.width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        directions = torch.stack(
            ((x - cx) / cfg.fx, (y - cy) / cfg.fy, torch.ones_like(x)), dim=-1
        ).reshape(-1, 3)
        directions = torch.nn.functional.normalize(directions, dim=-1)
        return torch.zeros_like(directions), directions

    def on_bind(self) -> None:
        super().on_bind()
        self._data_buffer["depth"] = torch.full(
            (self.num_instances, self.cfg.height, self.cfg.width),
            torch.inf,
            device=self.device,
        )

    def update(self, **kwargs: object) -> None:
        """Acquire a depth image and retain individual radial ranges."""
        super().update(**kwargs)
        depth = self._data_buffer["distance"] * self._directions[:, 2]
        self._data_buffer["depth"].copy_(depth.reshape_as(self._data_buffer["depth"]))

    def get_data_descriptors(self) -> dict[str, dict[str, object]]:
        """Return depth in metres along the optical axis."""
        fields = super().get_data_descriptors()
        fields["depth"].update(unit="m", frame="optical")
        return fields
