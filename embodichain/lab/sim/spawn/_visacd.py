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

"""Explicit VisACD geometry for Spawn versions without algorithm routing."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from dexsim.spawn import CollisionApproximation, CollisionDesc

__all__: list[str] = []


def explicit_visacd_collisions(collision: CollisionDesc) -> list[CollisionDesc]:
    """Materialize independent convex shapes without changing render geometry.

    File geometry uses the same native loader as Spawn, including its format
    transforms. Object body scale is intentionally left to Spawn. No persistent
    cache is used, so old CoACD geometry cannot satisfy a VisACD request.
    """
    import dexsim
    import open3d as o3d
    from dexsim.kit.meshproc.convex_decomposition import convex_decomposition_visacd
    from dexsim.spawn.adapters.common import collision_mesh_arrays

    arena = dexsim.default_world().get_env() if collision.file_path else None
    vertices, triangles = collision_mesh_arrays(arena, collision)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    requested_parts = collision.decomp_max_hulls
    # Some native builds exceed num_parts. Re-decompose the complete mesh with
    # a smaller budget rather than dropping geometry from an oversized result.
    for _ in range(3):
        success, parts = convex_decomposition_visacd(
            o3d.t.geometry.TriangleMesh.from_legacy(mesh),
            max_convex_hull_num=requested_parts,
            concavity=0.04,
            merge_vert_ratio=0.003,
            show_progress=False,
        )
        if not success:
            raise RuntimeError(
                "Requested VisACD decomposition failed; no fallback used."
            )
        if not isinstance(parts, (list, tuple)) or not parts:
            raise RuntimeError("VisACD returned no convex parts.")
        if len(parts) <= collision.decomp_max_hulls:
            break
        requested_parts -= len(parts) - collision.decomp_max_hulls
        if requested_parts < 1:
            break
    if len(parts) > collision.decomp_max_hulls:
        raise RuntimeError(
            f"VisACD returned {len(parts)} parts, exceeding "
            f"decomp_max_hulls={collision.decomp_max_hulls}."
        )
    result = []
    for index, part in enumerate(parts):
        native = part.to_legacy()
        points = np.asarray(native.vertices, dtype=np.float32)
        faces = np.asarray(native.triangles, dtype=np.int32)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) < 4
            or not np.isfinite(points).all()
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or not len(faces)
            or faces.min() < 0
            or faces.max() >= len(points)
        ):
            raise RuntimeError(f"VisACD part {index} contains invalid mesh geometry.")
        # Remove internal surface vertices before applying the declared budget.
        hull, _ = native.compute_convex_hull()
        if len(hull.vertices) > collision.decomp_vertex_limit:
            hull = hull.simplify_quadric_decimation(
                target_number_of_triangles=collision.decomp_vertex_limit
            )
            hull, _ = hull.compute_convex_hull()
        points = np.asarray(hull.vertices, dtype=np.float32)
        faces = np.asarray(hull.triangles, dtype=np.int32)
        if len(points) > collision.decomp_vertex_limit:
            raise RuntimeError(
                f"VisACD part {index} convex hull has {len(points)} vertices, "
                f"exceeding decomp_vertex_limit={collision.decomp_vertex_limit}."
            )
        result.append(
            replace(
                collision,
                segment_name=f"{collision.segment_name}_visacd_{index}",
                file_path=None,
                vertices=points.copy(),
                triangles=faces.copy(),
                normals=None,
                uv_coords=None,
                colors=None,
                local_transform=np.eye(4, dtype=np.float32),
                approximation=CollisionApproximation.CONVEX_HULL,
                render_source_index=None,
            )
        )
    return result
