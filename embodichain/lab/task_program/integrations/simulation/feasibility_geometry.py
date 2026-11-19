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

"""Physical collision extraction for the joint feasibility simulation adapter."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from embodichain.lab.sim.scene.feasibility import (
    CollisionBody,
    UnsupportedFeasibilityError,
)

__all__ = ["collision_bodies_from_simulation"]


def collision_bodies_from_simulation(
    env: Any, *, env_id: int = 0
) -> tuple[CollisionBody, ...]:
    """Read materialized robot-link and rigid-body physical collision shapes.

    Args:
        env: Prepared Gym host with full-link Robot kinematics.
        env_id: Physical arena whose collision resources are inspected.
    Returns:
        Link-local physical shapes with IDs ``robot_uid/link/index`` and
        ``object_uid/index``. Render geometry is never substituted.

    Unsupported physical geometry APIs/scales and shapes fail explicitly.
    The caller owns permanent structural and phase-specific contact permissions.
    """
    return _physical_collision_bodies(env, env_id=env_id)


def _planar_convex_parts(points, faces, origin, axes):
    """Extrude the actual planar footprint by one micrometre for Newton."""
    import shapely
    import trimesh

    xy = (points - origin) @ axes[:2].T
    triangles = shapely.polygons(xy[faces])
    surface = shapely.union_all(triangles[shapely.area(triangles) > 0])
    polygons = shapely.get_parts(surface)
    convex = [polygon.equals(polygon.convex_hull) for polygon in polygons]
    if all(convex):
        footprints = [polygon.convex_hull for polygon in polygons]
    elif hasattr(
        shapely, "constrained_delaunay_triangles"
    ) and shapely.geos_version >= (3, 10, 0):
        footprints = []
        for polygon, is_convex in zip(polygons, convex):
            if is_convex:
                footprints.append(polygon.convex_hull)
            else:
                footprints.extend(
                    shapely.get_parts(shapely.constrained_delaunay_triangles(polygon))
                )
    else:
        # Older GEOS versions still have the exact source triangulation.
        footprints = triangles[shapely.area(triangles) > 0]
    offset = axes[-1] * 5e-7
    parts = []
    for polygon in footprints:
        ring = np.asarray(polygon.exterior.coords)[:-1] @ axes[:2] + origin
        prism = trimesh.convex.convex_hull(np.vstack((ring - offset, ring + offset)))
        parts.append((prism.vertices, prism.faces))
    return parts


@lru_cache(maxsize=8)
def _convex_parts(vertices: bytes, triangles: bytes) -> tuple:
    """Cook scaled collider-local surfaces once across layouts and operators."""
    import trimesh

    points = np.frombuffer(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.frombuffer(triangles, dtype=np.int32).reshape(-1, 3)
    edges = points[faces[:, 1:]] - points[faces[:, :1]]
    faces = faces[np.any(np.cross(edges[:, 0], edges[:, 1]) != 0, axis=1)]
    if not len(faces):
        raise ValueError("Collision mesh has no surface area")
    used, remapped = np.unique(faces, return_inverse=True)
    if len(used) != len(points):
        points = points[used]
        faces = remapped.reshape(-1, 3).astype(np.int32)
    origin = points.mean(axis=0)
    centered = points - origin
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    if singular[1] <= singular[0] * 1e-12:
        raise ValueError("Collision mesh has no surface area")
    if np.max(np.abs(centered @ axes[-1])) <= 1e-9:
        parts = _planar_convex_parts(points, faces, origin, axes)
    else:
        mesh = trimesh.Trimesh(points, faces)
        if mesh.is_convex and mesh.is_volume:
            hull = mesh.convex_hull
            parts = [(hull.vertices, hull.faces)]
        else:
            # Use the engine's CPU V-HACD directly on materialized geometry:
            # a second asset import or vertex-welding pass can erase thin walls.
            from dexsim.cuda.pybind.meshproc import ProcMesh, vhacd

            cooked = vhacd(
                ProcMesh(points, faces), max_conv_hull=64, max_vert_conv_hull=64
            )
            if not cooked.is_success:
                raise ValueError("V-HACD failed to decompose the collision mesh")
            parts = [(part.vertices, part.triangles) for part in cooked.convex_list]
    if not parts:
        raise ValueError("Convex decomposition produced no parts")
    return tuple((tuple(map(tuple, v)), tuple(map(tuple, f))) for v, f in parts)


def _physical_collision_bodies(
    env: Any, *, env_id: int, layout_mesh_parts: dict[str, int] | None = None
) -> tuple[CollisionBody, ...]:
    """Optionally decompose physical triangle meshes for isolated layout screening."""
    robot = env.robot
    if env.sim.get_articulation_uid_list():
        raise UnsupportedFeasibilityError(
            "Non-robot articulations require an explicit complete collision adapter"
        )
    if not np.allclose(robot.cfg.body_scale, (1.0, 1.0, 1.0)):
        raise UnsupportedFeasibilityError(
            "Scaled robot collision adaptation requires an explicit geometry provider"
        )

    def convert(shape, collision_id, frame_uid, *, is_robot=False):
        kind = shape.shape_type.name.lower()
        kwargs = {}
        local_pose = shape.local_pose.cpu().numpy()
        if kind == "box":
            dimensions = tuple((shape.half_extents.cpu().numpy() * 2).tolist())
        elif kind == "sphere":
            dimensions = (float(shape.radius),)
        elif kind in ("capsule", "cylinder"):
            dimensions = (float(shape.radius), 2 * float(shape.half_height))
        elif kind == "convex" and (
            shape.vertices is not None
            and shape.triangles is not None
            and shape.vertices.numel()
            and shape.triangles.numel()
        ):
            dimensions = ()
            kwargs = {
                "vertices": tuple(map(tuple, shape.vertices.cpu().numpy())),
                "triangles": tuple(
                    tuple(int(i) for i in f)
                    for f in shape.triangles.cpu().numpy().reshape(-1, 3)
                ),
            }
        elif kind == "mesh" and layout_mesh_parts is not None:
            if shape.vertices is None or shape.triangles is None:
                raise UnsupportedFeasibilityError(
                    f"Missing physical mesh for {collision_id}"
                )
            points = shape.vertices.cpu().numpy()
            faces = shape.triangles.cpu().numpy()
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or len(points) < 3
                or not np.isfinite(points).all()
                or faces.dtype.kind not in "iu"
                or faces.ndim not in (1, 2)
                or (faces.ndim == 2 and faces.shape[1] != 3)
                or not faces.size
                or faces.size % 3
                or faces.min() < 0
                or faces.max() >= len(points)
            ):
                raise UnsupportedFeasibilityError(
                    f"Invalid physical mesh for {collision_id}"
                )
            try:
                parts = _convex_parts(
                    points.astype(np.float64, copy=False).tobytes(),
                    faces.astype(np.int32, copy=False).tobytes(),
                )
            except (RuntimeError, ValueError) as error:
                raise UnsupportedFeasibilityError(
                    f"Cannot decompose {collision_id}: {error}"
                ) from error
            layout_mesh_parts[collision_id] = len(parts)
            return tuple(
                CollisionBody(
                    f"{collision_id}/hull/{i}",
                    frame_uid,
                    "convex",
                    local_pose=local_pose.tolist(),
                    vertices=v,
                    triangles=f,
                    robot=is_robot,
                    source_collision_id=collision_id,
                )
                for i, (v, f) in enumerate(parts)
            )
        else:
            raise UnsupportedFeasibilityError(
                f"Native physical shape {collision_id}: {kind} is unsupported"
            )
        return (
            CollisionBody(
                collision_id,
                frame_uid,
                kind,
                dimensions,
                local_pose.tolist(),
                robot=is_robot,
                **kwargs,
            ),
        )

    bodies = []
    for link_name in robot.link_names:
        try:
            shapes = robot.get_link_collision_shapes(
                link_name, **({"env_id": env_id} if env_id else {})
            )
        except RuntimeError as error:
            raise UnsupportedFeasibilityError(str(error)) from error
        frame = f"{robot.uid}/{link_name}"
        for index, shape in enumerate(shapes):
            bodies.extend(convert(shape, f"{frame}/{index}", frame, is_robot=True))
    if not bodies:
        raise UnsupportedFeasibilityError(
            "Robot has no accessible physical collision geometry"
        )
    for uid in env.sim.get_rigid_object_uid_list():
        obj = env.sim.get_rigid_object(uid)
        for index, shape in enumerate(
            obj.get_collision_shapes(**({"env_id": env_id} if env_id else {}))
        ):
            bodies.extend(convert(shape, f"{uid}/{index}", uid))
    if getattr(env.sim, "_default_plane", None) is not None:
        bodies.append(
            CollisionBody(
                "default_plane", "default_plane", "halfspace", (0.0, 0.0, 1.0, 0.0)
            )
        )
    return tuple(bodies)
