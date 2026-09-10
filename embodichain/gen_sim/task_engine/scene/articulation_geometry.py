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

"""Read-only geometry in the base-link frame used by runtime articulation reset."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

__all__: list[str] = []


@dataclass(frozen=True)
class ArticulationGeometry:
    root_link: str
    vertices: np.ndarray
    link_poses: dict[str, np.ndarray]

    def world_vertices(self, config: Mapping[str, Any]) -> np.ndarray:
        scale = _vector(config.get("body_scale", [1, 1, 1]), "body_scale")
        if (scale <= 0).any():
            raise ValueError("Articulation body_scale must be positive.")
        rotation = Rotation.from_euler(
            "XYZ", _vector(config.get("init_rot", [0, 0, 0]), "init_rot"), degrees=True
        )
        return rotation.apply(self.vertices * scale) + _vector(
            config.get("init_pos", [0, 0, 0]), "init_pos"
        )


def _vector(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"Articulation {name} must contain three finite values.")
    return result


def read_articulation_geometry(path: str | Path) -> ArticulationGeometry:
    """Measure collision meshes relative to the unique native root rigid body.

    Runtime reset replaces the imported base pose. Measuring the USD default
    prim's world AABB instead would retain a translation that reset discards.
    Unsupported units/axes fail explicitly rather than assuming glTF conversion.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open articulation geometry: {path}")
    if (
        UsdGeom.GetStageUpAxis(stage) != "Z"
        or UsdGeom.GetStageMetersPerUnit(stage) != 1.0
    ):
        raise ValueError("Articulation geometry requires Z-up, metre-authored USD.")
    root = stage.GetDefaultPrim()
    if not root or not root.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise ValueError("Articulation geometry requires an explicit default root.")
    if any(
        not layer.anonymous and layer != stage.GetRootLayer()
        for layer in stage.GetUsedLayers()
    ):
        raise ValueError("Articulation geometry requires a self-contained USD layer.")
    prims = list(Usd.PrimRange(root))
    bodies = {p.GetPath(): p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    parents = {}
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            continue
        first, second = (
            joint.GetBody0Rel().GetTargets(),
            joint.GetBody1Rel().GetTargets(),
        )
        if not first and len(second) == 1 and prim.IsA(UsdPhysics.FixedJoint):
            continue
        if (
            len(first) != 1
            or len(second) != 1
            or first[0] not in bodies
            or second[0] not in bodies
        ):
            raise ValueError(
                "Articulation geometry requires exact internal joint bodies."
            )
        if second[0] in parents:
            raise ValueError("Articulation geometry has multiple parents for one body.")
        parents[second[0]] = first[0]
    roots = set(bodies) - set(parents)
    if len(roots) != 1:
        raise ValueError("Articulation geometry requires one connected root body.")
    base_path = next(iter(roots))
    for body in bodies:
        seen = set()
        current = body
        while current != base_path:
            if current in seen or current not in parents:
                raise ValueError(
                    "Articulation geometry contains disconnected or cyclic bodies."
                )
            seen.add(current)
            current = parents[current]
    cache = UsdGeom.XformCache()
    base = np.asarray(cache.GetLocalToWorldTransform(bodies[base_path])).T
    if not np.isfinite(base).all() or not np.allclose(
        base[:3, :3].T @ base[:3, :3], np.eye(3), atol=1e-6
    ):
        raise ValueError(
            "Articulation geometry requires a finite, unscaled root-body frame."
        )
    inverse = np.linalg.inv(base)
    if len({body.GetName() for body in bodies.values()}) != len(bodies):
        raise ValueError("Articulation geometry requires unique native body names.")
    link_poses = {
        body.GetName(): inverse @ np.asarray(cache.GetLocalToWorldTransform(body)).T
        for body in bodies.values()
    }
    vertices = []
    for prim in prims:
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        if not any(prim.GetPath().HasPrefix(body) for body in bodies):
            raise ValueError("Articulation collision mesh has no rigid-body owner.")
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise ValueError("Articulation collision mesh has invalid vertices.")
        transform = inverse @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        vertices.append(points @ transform[:3, :3].T + transform[:3, 3])
    if not vertices:
        raise ValueError("Articulation geometry has no measurable collision meshes.")
    combined = np.concatenate(vertices)
    if not np.isfinite(combined).all():
        raise ValueError(
            "Articulation transforms produced nonfinite collision geometry."
        )
    combined.setflags(write=False)
    return ArticulationGeometry(bodies[base_path].GetName(), combined, link_poses)


def fit_articulation_to_proxy(
    config: Mapping[str, Any], proxy_vertices: np.ndarray, *, table_top_z: float
) -> dict[str, Any]:
    """Explicitly fit a copy to a Z-up local proxy envelope and place its bottom.

    This repair is not an automatic loading fallback. A caller must publish the
    returned configuration as a new scene revision and retain source provenance.
    Uniform fitting preserves the mechanism and never exceeds the proxy envelope.
    A nearly level fixed base is aligned with the horizontal support plane;
    larger tilts require an explicit orientation decision instead of guessing.
    """
    if config.get("fix_base") is not True or not np.isfinite(table_top_z):
        raise ValueError("Proxy fitting requires a fixed base and measured tabletop.")
    proxy = np.asarray(proxy_vertices, dtype=float)
    if (
        proxy.ndim != 2
        or proxy.shape[1] != 3
        or not len(proxy)
        or not np.isfinite(proxy).all()
    ):
        raise ValueError("Proxy fitting requires finite three-dimensional vertices.")
    geometry = read_articulation_geometry(config["fpath"])
    rotation = Rotation.from_euler(
        "XYZ", _vector(config.get("init_rot", [0, 0, 0]), "init_rot"), degrees=True
    )
    matrix = rotation.as_matrix()
    if matrix[2, 2] < np.cos(np.deg2rad(1.0)):
        raise ValueError(
            "Tabletop proxy fitting requires a base within one degree of level."
        )
    yaw = float(np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0])))
    rotation = Rotation.from_euler("XYZ", [0.0, 0.0, yaw], degrees=True)
    actual, proxy = rotation.apply(geometry.vertices), rotation.apply(proxy)
    actual_extent, proxy_extent = np.ptp(actual, axis=0), np.ptp(proxy, axis=0)
    if (
        not np.isfinite(actual_extent).all()
        or not np.isfinite(proxy_extent).all()
        or (actual_extent <= 0).any()
        or (proxy_extent <= 0).any()
    ):
        raise ValueError("Proxy fitting requires non-degenerate geometry extents.")
    scale = float(np.min(proxy_extent / actual_extent))
    actual *= scale
    position = _vector(config.get("init_pos", [0, 0, 0]), "init_pos").copy()
    position[:2] += (
        (proxy.min(0) + proxy.max(0)) / 2 - (actual.min(0) + actual.max(0)) / 2
    )[:2]
    position[2] = float(table_top_z) + 0.001 - float(actual[:, 2].min())
    repaired = deepcopy(dict(config))
    repaired["body_scale"] = [scale] * 3
    repaired["init_pos"] = position.tolist()
    repaired["init_rot"] = [0.0, 0.0, yaw]
    return repaired
