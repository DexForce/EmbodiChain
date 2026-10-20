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

"""Interpret retained Default/USD collision materialization inputs, not poses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

__all__: list[str] = []


def _mesh(vertices: Any, faces: Any) -> trimesh.Trimesh:
    points = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 4
        or not np.isfinite(points).all()
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or not len(triangles)
        or not np.issubdtype(triangles.dtype, np.integer)
        or triangles.min() < 0
        or triangles.max() >= len(points)
    ):
        raise ValueError("scene_collision_descriptor_invalid_topology")
    return trimesh.Trimesh(vertices=points, faces=triangles, process=False)


def descriptor_collision_meshes(
    simulation: Any, native: Any, simulation_uid: str
) -> dict[tuple[str, str], trimesh.Trimesh] | None:
    """Use the exact SDK convex-part preprocessing in owning-link frames.

    None preserves the previous adapter for deployments outside this route.
    An available Default Scene must resolve completely or fail closed. Retained
    descriptors are not cooked native geometry; native shape/offset equivalence
    requires separate qualification. No live pose or physics state is cached.

    Args:
        simulation: Current simulation with a finalized spawn Scene.
        native: Bound articulation wrapper, used for source and scale identity.
        simulation_uid: Exact native articulation identity in the binding.

    Returns:
        Convex parts keyed by native link and source/part identity, or None when
        this deployment retains the previous geometry adapter.
    """
    if getattr(simulation, "physics_backend", None) != "default":
        return None
    scene = getattr(simulation, "spawn_result", None)
    if scene is None:
        raise ValueError("scene_collision_descriptor_scene_unavailable")
    from dexsim.scene import SpawnedArticulation
    from dexsim.spawn.adapters.common import generated_mesh
    from dexsim.spawn.descs import CollisionApproximation, GeometryType
    from dexsim.kit.meshproc.convex_cache import (
        convex_decomposition_to_obj,
        load_obj_as_convex_parts,
    )

    # The SDK also retains a global arena, not an extra episode environment.
    if scene.backend != "dexsim" or simulation.num_envs != 1:
        raise ValueError("scene_collision_descriptor_backend_or_batch_unavailable")
    matches = [
        (path, handle)
        for path in scene.get_paths()
        if isinstance(handle := scene.get(path), SpawnedArticulation)
        and handle.get_name() == simulation_uid
        and handle.arena_name in scene.arenas.names
    ]
    if len(matches) != 1:
        raise ValueError(
            f"scene_collision_descriptor_identity_unavailable:{simulation_uid}"
        )
    path, handle = matches[0]
    desc = handle.desc
    if desc.urdf_path is not None and Path(native.cfg.fpath).suffix.lower() == ".urdf":
        return None
    if (
        desc.usd is None
        or desc.urdf_path is not None
        or not desc.usd.file_path
        or Path(desc.usd.file_path).resolve() != Path(native.cfg.fpath).resolve()
        or len({link.name for link in desc.links}) != len(desc.links)
        or {link.name for link in desc.links} != set(handle.get_link_names())
    ):
        raise ValueError("scene_collision_descriptor_source_or_links_unavailable")
    scale = np.asarray(desc.body_scale, dtype=np.float32)
    if (
        scale.shape != (3,)
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.allclose(scale, scale[0])
        or not np.allclose(scale, native.cfg.body_scale)
    ):
        raise ValueError("scene_collision_descriptor_scale_unavailable")
    result = {}
    for link in desc.links:
        colliders = scene.get_link_desc(path, link.name).collisions
        for index, geom in enumerate(colliders):
            # Default materializes every retained collider; flags apply per link.
            if (
                geom.geometry_type != GeometryType.MESH
                or geom.vertices is None
                or geom.triangles is None
                or geom.file_path is not None
                or geom.approximation
                not in {
                    CollisionApproximation.CONVEX_HULL,
                    CollisionApproximation.CONVEX_DECOMPOSITION,
                }
            ):
                raise ValueError(
                    "scene_collision_descriptor_representation_unavailable"
                )
            _mesh(geom.vertices, geom.triangles)
            transform = np.asarray(geom.local_transform)
            if (
                transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], (0, 0, 0, 1))
            ):
                raise ValueError("scene_collision_descriptor_transform_unavailable")
            vertices, triangles, _, _ = generated_mesh(geom)
            _mesh(vertices, triangles)
            if geom.approximation == CollisionApproximation.CONVEX_DECOMPOSITION:
                cache = convex_decomposition_to_obj(
                    vertices,
                    triangles,
                    max_convex_hulls=geom.decomp_max_hulls,
                    max_vert_per_convex_hull=geom.decomp_vertex_limit,
                    algorithm=geom.decomp_algorithm,
                )
                parts = load_obj_as_convex_parts(cache)
            else:
                parts = [(vertices, triangles)]
            if not parts:
                raise ValueError("scene_collision_descriptor_parts_unavailable")
            source = geom.usd.prim_path if geom.usd is not None else geom.segment_name
            for part, (points, faces) in enumerate(parts):
                # Native ConvexMeshGeometry cooks each part independently.
                hull = _mesh(points, faces).convex_hull
                hull.apply_scale(scale)
                result[(link.name, f"{source}:{index}:{part}")] = hull
    if not result:
        raise ValueError("scene_collision_descriptor_colliders_unavailable")
    return result
