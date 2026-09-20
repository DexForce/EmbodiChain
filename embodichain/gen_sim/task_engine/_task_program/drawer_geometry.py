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

"""Link-local placement hints, without cavity qualification or fit rejection."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from .articulation_binding import PrismaticBinding, discover_prismatic_parts, _stage

__all__: list[str] = []

RELEASE_GAP = 0.003


def collision_meshes(config: dict[str, Any]) -> dict[tuple[str, str], trimesh.Trimesh]:
    """Read enabled collision meshes in their native owning link frames."""
    from pxr import UsdGeom, UsdPhysics

    stage = _stage(config["fpath"])
    cache = UsdGeom.XformCache()
    scale = np.asarray(config.get("body_scale", [1, 1, 1]), dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Drawer geometry requires a finite positive scale.")
    if not np.allclose(scale, scale[0]):
        raise ValueError("Drawer geometry requires uniform scale.")
    result = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        owner = prim
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        if not owner:
            raise ValueError("Drawer collision mesh has no rigid-body owner.")
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not len(vertices)
            or not np.isfinite(vertices).all()
            or not len(indices)
            or (counts < 3).any()
            or counts.sum() != len(indices)
            or indices.min() < 0
            or indices.max() >= len(vertices)
        ):
            raise ValueError("Drawer collision mesh has invalid topology.")
        pose = np.asarray(cache.GetLocalToWorldTransform(owner)).T
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6):
            raise ValueError("Drawer link frames must be unscaled rigid transforms.")
        local = np.linalg.inv(pose) @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        vertices = (vertices @ local[:3, :3].T + local[:3, 3]) * scale
        faces, offset = [], 0
        for count in counts:
            polygon = indices[offset : offset + count]
            faces.extend(
                (polygon[0], polygon[i], polygon[i + 1]) for i in range(1, count - 1)
            )
            offset += count
        result[(owner.GetName(), str(prim.GetPath()))] = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=True
        )
    if not result:
        raise ValueError("Drawer has no collision geometry.")
    return result


def moving_links(config: dict[str, Any], binding: PrismaticBinding) -> set[str]:
    """Include fixed descendants by physical topology, not USD path ancestry."""
    from pxr import UsdPhysics

    stage = _stage(config["fpath"])
    candidates = discover_prismatic_parts(config)
    candidate = next(p for p in candidates if p.joint == binding.joint)
    paths = {candidate.link_path}
    edges = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            continue
        parent, child = (
            joint.GetBody0Rel().GetTargets(),
            joint.GetBody1Rel().GetTargets(),
        )
        if len(parent) == len(child) == 1:
            edges.append((str(parent[0]), str(child[0])))
    while True:
        updated = paths | {child for parent, child in edges if parent in paths}
        if updated == paths:
            return {stage.GetPrimAtPath(path).GetName() for path in paths}
        paths = updated


def drawer_region(
    config: dict[str, Any], binding: PrismaticBinding
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the selected part in the same link frame E6 uses for its handle."""
    from ..scene.articulation_geometry import read_articulation_geometry

    poses = read_articulation_geometry(config["fpath"]).link_poses
    selected = moving_links(config, binding)
    meshes = []
    for (link, path), mesh in collision_meshes(config).items():
        if link not in selected:
            continue
        relative = np.linalg.inv(poses[binding.link]) @ poses[link]
        relative[:3, 3] *= binding.scale
        mesh.apply_transform(relative)
        meshes.append((path, mesh))
    structural = [mesh for path, mesh in meshes if path != binding.handle_path]
    return placement_region(structural or [mesh for _, mesh in meshes])


def placement_region(meshes: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a floor and center; missing walls or obstructions do not reject a task."""
    vertices = np.concatenate([mesh.vertices for mesh in meshes])
    low, high = vertices.min(0), vertices.max(0)
    components = [part for mesh in meshes for part in mesh.split(only_watertight=False)]
    floors = [
        mesh for mesh in components if mesh.extents[2] < min(mesh.extents[:2]) / 3
    ]
    if floors:
        floor = max(floors, key=lambda mesh: float(np.prod(mesh.extents[:2])))
        low[:2], high[:2] = floor.bounds[:, :2]
        low[2] = floor.bounds[1, 2]
        high[2] = max(high[2], low[2])
    return low, high


def placement_pose(
    vertices: np.ndarray,
    rotation: np.ndarray,
    link_pose: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Aim over the floor center, even when the object exceeds the drawer bounds."""
    if not all(
        np.isfinite(v).all() for v in (vertices, rotation, link_pose, low, high)
    ):
        raise ValueError(
            "Drawer placement requires finite observed geometry and poses."
        )
    local_rotation = link_pose[:3, :3].T @ rotation
    points = vertices @ local_rotation.T
    minimum, maximum = points.min(0), points.max(0)
    local_position = (low + high - minimum - maximum) / 2
    local_position[2] = low[2] - minimum[2] + RELEASE_GAP
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = link_pose[:3, :3] @ local_position + link_pose[:3, 3]
    return result
