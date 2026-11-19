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

"""Contact-supported XY translations of rigid objects in bounded layout proposals."""

from __future__ import annotations

import numpy as np
import shapely
import trimesh
from scipy.spatial import QhullError

from ..geometry import SupportSurfaceConfig

__all__: list[str] = []

# Accommodate small authored mesh gaps before the physical settling step.
_CONTACT_HEIGHT_TOLERANCE_M = 0.003


def _horizontal_patch(triangles, bottom, *, upward):
    """Project near-contact faces at the authored base height, retaining holes."""
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    norm = np.linalg.norm(cross, axis=1)
    z = cross[:, 2] if upward else -cross[:, 2]
    mask = (
        (norm > 1e-14)
        & (z >= SupportSurfaceConfig().normal_z_min * norm)
        & (
            np.abs(triangles[:, :, 2] - bottom).max(axis=1)
            <= _CONTACT_HEIGHT_TOLERANCE_M
        )
    )
    return shapely.union_all(shapely.polygons(triangles[mask, :, :2]))


def _root_support_offsets(source, nodes, table, bounds, movable_roots):
    """Certify bounded displacements from a measured contact patch on the named support.

    Roots translate at fixed height and orientation. Eroding the support by the
    displacement radius leaves contact points that stay supported at *every*
    admitted offset. Their convex hull must strictly surround the projected
    uniform-density centroid (convex envelope for open meshes). Body extents
    still govern obstacle clearance.
    """
    if not bounds:
        return {}
    parents = {node["parent"] for node in nodes}
    result = {}
    support_triangles = None
    for index, node in enumerate(nodes):
        uid = node["uid"]
        if (
            node["parent"] >= 0
            or index in parents
            or (uid in movable_roots and uid not in bounds)
        ):
            continue
        pose = np.asarray(source.poses[uid])
        corners = np.asarray(node["corners"]) @ pose[:3, :3].T + pose[:3, 3]
        if np.all(corners[:, :2].min(0) >= table[0]) and np.all(
            corners[:, :2].max(0) <= table[1]
        ):
            continue
        if support_triangles is None:
            support = source.support_uid
            transform = np.asarray(source.poses[support])
            vertices = source.vertices[support] @ transform[:3, :3].T + transform[:3, 3]
            support_triangles = vertices[source.faces[support]]
        vertices = source.vertices[uid] @ pose[:3, :3].T + pose[:3, 3]
        bottom = vertices[:, 2].min()
        footprint = _horizontal_patch(vertices[source.faces[uid]], bottom, upward=False)
        support_patch = _horizontal_patch(support_triangles, bottom, upward=True)
        if footprint.is_empty or support_patch.is_empty:
            continue
        mesh = trimesh.Trimesh(vertices, source.faces[uid])
        # Open surfaces do not define a physical volume. Their convex envelope
        # supplies a reproducible centroid estimate for proposal generation.
        try:
            volume = mesh if mesh.is_volume else mesh.convex_hull
        except QhullError:
            continue
        center = shapely.Point(volume.center_mass[:2])

        def stable(radius):
            # GEOS approximates circular arcs with chords. Enlarge the erosion
            # radius so even a chord around a hole clears the requested disk.
            interior = (
                support_patch.buffer(-radius / np.cos(np.pi / 64), quad_segs=16)
                if radius
                else support_patch
            )
            contact = interior.intersection(footprint)
            return contact.area > 0 and contact.convex_hull.contains(center)

        if not stable(0):
            continue
        offsets = np.asarray(bounds.get(uid, [[0, 0], [0, 0]]), dtype=float)
        radius = np.linalg.norm(np.max(np.abs(offsets), axis=0))
        if not stable(radius):
            low, high = 0.0, radius
            for _ in range(16):
                mid = (low + high) / 2
                if stable(mid):
                    low = mid
                else:
                    high = mid
            # Preserve the requested aspect ratio, including one-dimensional ranges.
            half = np.max(np.abs(offsets), axis=0) * (low / radius)
            offsets = np.array(
                [np.maximum(offsets[0], -half), np.minimum(offsets[1], half)]
            )
            if np.any(offsets[1] < offsets[0]):
                continue
        result[uid] = offsets.tolist()
    return result
