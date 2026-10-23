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

"""Link-local drawer placement hints and measured containment reports."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from .articulation_binding import PrismaticBinding, discover_prismatic_parts, _stage

__all__: list[str] = []

RELEASE_GAP = 0.003
DRAWER_PLACEMENT_MARGIN = 0.005
DRAWER_PLACEMENT_ROTATIONS = (0.0, np.pi / 2.0)
DRAWER_PLACEMENT_MAX_CANDIDATES = 3


def _rotation_matrix(
    angle: float, axis: np.ndarray | tuple[float, float, float]
) -> np.ndarray:
    """Return one world-frame axis rotation for release-pose candidates."""
    axis_array = np.asarray(axis, dtype=float)
    axis_array /= np.linalg.norm(axis_array)
    x, y, z = axis_array
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [
                cosine + x * x * (1 - cosine),
                x * y * (1 - cosine) - z * sine,
                x * z * (1 - cosine) + y * sine,
            ],
            [
                y * x * (1 - cosine) + z * sine,
                cosine + y * y * (1 - cosine),
                y * z * (1 - cosine) - x * sine,
            ],
            [
                z * x * (1 - cosine) - y * sine,
                z * y * (1 - cosine) + x * sine,
                cosine + z * z * (1 - cosine),
            ],
        ]
    )


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


def placement_pose_candidates(
    vertices: np.ndarray,
    rotation: np.ndarray,
    link_pose: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    margin: float = DRAWER_PLACEMENT_MARGIN,
    preserve_vertical: bool = False,
) -> tuple[np.ndarray, ...]:
    """Generate a bounded set of link-local floor targets for one payload.

    The returned poses are object poses, not end-effector poses.  Candidates
    are eroded by the payload footprint and vertical extent before being
    emitted.  This is a geometric preflight only; callers still must run IK,
    velocity, rim-clearance, and terminal containment checks.
    """
    vertices = np.asarray(vertices, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    link_pose = np.asarray(link_pose, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or rotation.shape != (3, 3)
        or link_pose.shape != (4, 4)
        or low.shape != (3,)
        or high.shape != (3,)
        or not all(
            np.isfinite(value).all() for value in (rotation, link_pose, low, high)
        )
        or (low > high).any()
        or not np.isfinite(margin)
        or margin < 0.0
    ):
        raise ValueError("Drawer placement candidate inputs are invalid.")
    if not np.allclose(link_pose[:3, :3].T @ link_pose[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("Drawer link pose must be a rigid transform.")

    link_rotation = link_pose[:3, :3]
    candidates: list[np.ndarray] = []
    extents = np.ptp(vertices, axis=0)
    isotropic = float(extents.max()) <= max(float(extents.min()) * 1.25, 1e-6)
    if isotropic and not preserve_vertical:
        rotation_variants = (
            rotation,
            _rotation_matrix(np.pi / 2.0, link_rotation[:, 0]) @ rotation,
            _rotation_matrix(np.pi / 2.0, link_rotation[:, 1]) @ rotation,
        )
    else:
        rotation_variants = tuple(
            _rotation_matrix(angle, link_rotation[:, 2]) @ rotation
            for angle in (
                (*DRAWER_PLACEMENT_ROTATIONS, -np.pi / 2.0, np.pi)
                if preserve_vertical
                else DRAWER_PLACEMENT_ROTATIONS
            )
        )
    for variant_index, world_rotation in enumerate(rotation_variants):
        local_rotation = link_rotation.T @ world_rotation
        points = vertices @ local_rotation.T
        minimum, maximum = points.min(0), points.max(0)
        safe_low = low + margin
        safe_high = high - margin
        center_low = safe_low[:2] - minimum[:2]
        center_high = safe_high[:2] - maximum[:2]
        # Keep a finite center even when the footprint exceeds the drawer.
        # The runtime must attempt the placement and let the physical result
        # decide whether the object actually fits.
        center = (center_low + center_high) / 2.0
        xy_candidates = (center,)
        if not isotropic and variant_index == 0:
            xy_candidates += (center_low * 0.25 + center_high * 0.75,)
        z = low[2] + margin - minimum[2] + RELEASE_GAP
        for xy in xy_candidates:
            pose = np.eye(4)
            pose[:3, :3] = world_rotation
            local_position = np.array([xy[0], xy[1], z])
            pose[:3, 3] = link_rotation @ local_position + link_pose[:3, 3]
            if not any(
                np.allclose(pose, existing, atol=1e-8) for existing in candidates
            ):
                candidates.append(pose)
                if len(candidates) >= (
                    5 if preserve_vertical else DRAWER_PLACEMENT_MAX_CANDIDATES
                ):
                    return tuple(candidates)
    return tuple(candidates)


def drawer_containment_report(
    vertices: np.ndarray,
    object_pose: np.ndarray,
    link_pose: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    margin: float = DRAWER_PLACEMENT_MARGIN,
) -> dict[str, object]:
    """Check the complete payload mesh against a live drawer-link envelope."""
    vertices = np.asarray(vertices, dtype=float)
    object_pose = np.asarray(object_pose, dtype=float)
    link_pose = np.asarray(link_pose, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    relative = np.linalg.inv(link_pose) @ object_pose
    local = vertices @ relative[:3, :3].T + relative[:3, 3]
    safe_low = low + margin
    # Contact with the measured floor is allowed within the floor tolerance;
    # lateral bounds remain eroded to keep the payload off the walls.
    safe_low[2] = low[2] - margin
    safe_high = high - margin
    minimum, maximum = local.min(0), local.max(0)
    checks = {
        "finite": bool(np.isfinite(local).all()),
        "lower_bounds": bool((minimum >= safe_low).all()),
        "upper_bounds": bool((maximum <= safe_high).all()),
        "floor_gap": float(minimum[2] - low[2]),
    }
    accepted = bool(
        checks["finite"]
        and checks["lower_bounds"]
        and checks["upper_bounds"]
        and checks["floor_gap"] >= -margin
    )
    return {
        "accepted": accepted,
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "safe_lower": safe_low.tolist(),
        "safe_upper": safe_high.tolist(),
        "checks": checks,
        "margin": float(margin),
    }
