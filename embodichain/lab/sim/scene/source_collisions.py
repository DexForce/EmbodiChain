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
"""Source-qualified articulation collision inputs with stable owner identities."""

from __future__ import annotations
from typing import Any
import numpy as np
import trimesh
from embodichain.lab.sim.spawn.source_geometry import read_usd_source

__all__ = ["read_source_collision_meshes"]


def read_source_collision_meshes(
    config: dict[str, Any],
) -> dict[tuple[str, str], trimesh.Trimesh]:
    """Read enabled source collision meshes in native owning-link frames.

    Args:
        config: Existing self-contained metre-authored USD source and positive
            uniform body scale. No native object is constructed or queried.

    Returns:
        Insertion-ordered owner/path mesh mapping retaining source polygon-fan
        triangulation and Trimesh processing. These are authored input meshes,
        not cooked native shapes or a collision-free certificate.
    """
    from pxr import UsdGeom, UsdPhysics

    source = read_usd_source(config["fpath"])
    source.require_geometry()
    stage = source.stage
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
