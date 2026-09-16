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

"""VISACD acceleration and original-solid collision checks for assembly meshes."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

import fcl
import manifold3d
import numpy as np
import trimesh

__all__ = ["decompose", "HullCollision", "ExactCollision"]


def decompose(
    mesh: trimesh.Trimesh, cache: Path, concavity: float, max_parts: int
) -> list[trimesh.Trimesh]:
    """Cache VISACD convex parts using geometry, settings and engine version.

    Args:
        mesh: Solid mesh in local coordinates, in meters.
        cache: Directory for reusable numeric hull archives.
        concavity: VISACD approximation threshold.
        max_parts: Maximum number of convex hulls.

    Returns:
        Convex meshes in the input frame. Empty decompositions raise RuntimeError.
    """
    settings = {
        "schema": 1,
        "concavity": concavity,
        "max_parts": max_parts,
        "merge_vert_ratio": 1e-5,
        "dexsim": version("dexsim_engine"),
    }
    key = hashlib.sha256(
        mesh.vertices.tobytes()
        + mesh.faces.tobytes()
        + json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"visacd_{key}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            return [
                trimesh.Trimesh(archive[f"v{i}"], archive[f"f{i}"], process=False)
                for i in range(int(archive["count"]))
            ]

    import open3d as o3d
    from dexsim.kit.meshproc import convex_decomposition_visacd

    source = o3d.t.geometry.TriangleMesh()
    source.vertex.positions = o3d.core.Tensor(mesh.vertices, o3d.core.Dtype.Float32)
    source.triangle.indices = o3d.core.Tensor(mesh.faces, o3d.core.Dtype.Int32)
    success, parts = convex_decomposition_visacd(
        source,
        merge_vert_ratio=1e-5,
        concavity=concavity,
        max_convex_hull_num=max_parts,
        is_visual=False,
        show_progress=False,
    )
    if not success or not parts:
        raise RuntimeError(
            "VISACD returned no hulls; check the mesh or use validation.collision='exact'"
        )
    hulls = [
        trimesh.Trimesh(
            part.vertex.positions.numpy(), part.triangle.indices.numpy(), process=False
        ).convex_hull
        for part in parts
    ]
    arrays = {"count": np.array(len(hulls))}
    for i, hull in enumerate(hulls):
        arrays[f"v{i}"] = hull.vertices
        arrays[f"f{i}"] = hull.faces
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache, suffix=".npz", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hulls


def _convex_object(mesh: trimesh.Trimesh) -> fcl.CollisionObject:
    faces = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).ravel()
    return fcl.CollisionObject(fcl.Convex(mesh.vertices, len(mesh.faces), faces))


class HullCollision:
    """Query solid convex parts, pruning pairs by their axis-aligned bounds."""

    def __init__(
        self, base: list[trimesh.Trimesh], assemble: list[trimesh.Trimesh]
    ) -> None:
        self.base_objects = [_convex_object(mesh) for mesh in base]
        self.assemble_objects = [_convex_object(mesh) for mesh in assemble]
        self.base_bounds = np.array([mesh.bounds for mesh in base])
        self.assemble_vertices = [mesh.vertices for mesh in assemble]

    def collides(self, pose: np.ndarray) -> bool:
        """Check convex-solid overlap for a validated assemble-to-base pose.

        Args:
            pose: Relative SE(3) matrix.

        Returns:
            Whether any pair of solid hulls touches or overlaps.
        """
        transform = fcl.Transform(pose[:3, :3], pose[:3, 3])
        for moving, vertices in zip(self.assemble_objects, self.assemble_vertices):
            world = vertices @ pose[:3, :3].T + pose[:3, 3]
            possible = np.all(self.base_bounds[:, 1] >= world.min(axis=0), axis=1)
            possible &= np.all(self.base_bounds[:, 0] <= world.max(axis=0), axis=1)
            moving.setTransform(transform)
            for i in np.flatnonzero(possible):
                if fcl.collide(
                    self.base_objects[i],
                    moving,
                    fcl.CollisionRequest(),
                    fcl.CollisionResult(),
                ):
                    return True
        return False


def _triangle_object(mesh: trimesh.Trimesh) -> fcl.CollisionObject:
    model = fcl.BVHModel()
    model.beginModel(len(mesh.faces), len(mesh.vertices))
    model.addSubModel(mesh.vertices, mesh.faces)
    model.endModel()
    return fcl.CollisionObject(model)


def _solid(mesh: trimesh.Trimesh) -> manifold3d.Manifold:
    solid = manifold3d.Manifold(
        manifold3d.Mesh64(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.uint64),
        )
    )
    if solid.status() != manifold3d.Error.NoError or solid.is_empty():
        raise ValueError(f"Mesh cannot form a Manifold solid: {solid.status()}")
    return solid


class ExactCollision:
    """Retain original triangles and solids so cavities and containment count."""

    def __init__(self, base: trimesh.Trimesh, assemble: trimesh.Trimesh) -> None:
        self.base = _triangle_object(base)
        self.assemble = _triangle_object(assemble)
        self.base_solid = _solid(base)
        self.assemble_solid = _solid(assemble)

    def surface_collision(self, pose: np.ndarray) -> bool:
        """Check original triangle boundaries for intersection.

        Args:
            pose: Validated assemble-to-base SE(3) matrix.

        Returns:
            Whether the surfaces touch or intersect; containment is separate.
        """
        self.assemble.setTransform(fcl.Transform(pose[:3, :3], pose[:3, 3]))
        return bool(
            fcl.collide(
                self.base, self.assemble, fcl.CollisionRequest(), fcl.CollisionResult()
            )
        )

    def distance(self, pose: np.ndarray) -> float:
        """Measure separation between the original triangle boundaries.

        Args:
            pose: Validated assemble-to-base SE(3) matrix.

        Returns:
            Nonnegative boundary separation in meters.
        """
        self.assemble.setTransform(fcl.Transform(pose[:3, :3], pose[:3, 3]))
        return max(0.0, float(fcl.distance(self.base, self.assemble)))

    def intersection_volume(self, pose: np.ndarray) -> float:
        """Measure original-solid overlap, including complete containment.

        Args:
            pose: Validated assemble-to-base SE(3) matrix.

        Returns:
            Intersection volume in cubic meters.
        """
        overlap = self.base_solid ^ self.assemble_solid.transform(pose[:3, :4])
        if overlap.status() != manifold3d.Error.NoError:
            raise RuntimeError(f"Solid intersection failed: {overlap.status()}")
        return max(0.0, float(overlap.volume()))
