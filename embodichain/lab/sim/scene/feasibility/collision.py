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

"""Independent physical-geometry collision checks with lazy Python-FCL loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
import math
from numbers import Integral

import numpy as np
import torch
from embodichain.compute.geometry.solid import points_in_closed_mesh

from .._validation import _identifier, _rigid_pose
from ._values import finite
from .contracts import FeasibilityCheck

__all__ = ["CollisionBody", "FCLCollisionChecker", "UnsupportedFeasibilityError"]


class UnsupportedFeasibilityError(RuntimeError):
    """Required geometry, task adaptation or runtime capability is unavailable."""


@dataclass(frozen=True)
class CollisionBody:
    """One physical shape in an explicitly named link/entity frame.

    Args:
        collision_id: Stable physical ID for phase-specific contact permissions.
        frame_uid: Pose-provider frame identity; several shapes can share it.
        kind: box, sphere, capsule, cylinder, or triangle mesh.
        dimensions: Full box extents, radius, or radius/full cylinder height.
        local_pose: Shape-to-frame transform.
        vertices: Mesh-local positions; scale must already be applied.
        triangles: Mesh triangle indices.
        robot: Whether this shape belongs to a robot.
    """

    collision_id: str
    frame_uid: str
    kind: str
    dimensions: tuple[float, ...] = ()
    local_pose: tuple = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    vertices: tuple[tuple[float, float, float], ...] = ()
    triangles: tuple[tuple[int, int, int], ...] = ()
    robot: bool = False

    def __post_init__(self) -> None:
        _identifier(self.collision_id, "collision_id")
        _identifier(self.frame_uid, "frame_uid")
        object.__setattr__(self, "local_pose", _rigid_pose(self.local_pose))
        size = finite(self.dimensions, "shape dimensions")
        expected = {
            "box": 3,
            "sphere": 1,
            "capsule": 2,
            "cylinder": 2,
            "halfspace": 4,
            "mesh": 0,
        }
        if self.kind not in expected:
            raise UnsupportedFeasibilityError(
                f"Unsupported physical shape {self.kind!r}"
            )
        if (
            len(size) != expected[self.kind]
            or (self.kind != "halfspace" and any(v <= 0 for v in size))
            or (
                self.kind == "halfspace"
                and not np.isclose(np.linalg.norm(size[:3]), 1.0)
            )
        ):
            raise ValueError("invalid physical shape dimensions")
        object.__setattr__(self, "dimensions", size)
        vertices = tuple(finite(v, "vertex") for v in self.vertices)
        raw_triangles = tuple(tuple(v) for v in self.triangles)
        if any(
            isinstance(i, bool) or not isinstance(i, Integral)
            for t in raw_triangles
            for i in t
        ):
            raise ValueError("triangle indices must be integers")
        triangles = tuple(tuple(int(i) for i in t) for t in raw_triangles)
        if self.kind == "mesh" and (not vertices or not triangles):
            raise ValueError("mesh collision requires vertices and triangles")
        if any(len(v) != 3 for v in vertices) or any(
            len(t) != 3
            or any(type(i) is not int or not 0 <= i < len(vertices) for i in t)
            for t in triangles
        ):
            raise ValueError("invalid collision mesh")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)
        if type(self.robot) is not bool:
            raise TypeError("robot flag must be boolean")


class FCLCollisionChecker:
    """Check robot/self/inter-arm/payload collisions against complete geometry.

    Args:
        bodies: Physical shapes, including fixed mounting/environment geometry.
        ignored_pairs: Explicit permanent structural/contact exemptions. Same
            rigid-frame shapes are automatically excluded; adjacent links are not.
        clearance_m: Required non-contact margin, in metres.

    Queries use FCL primitives/convex/BVH geometry and closed-mesh solid-angle
    containment witnesses. Broad-phase box separation supplies conservative
    distance lower bounds. Open/inconsistently oriented meshes cannot pass a
    required containment check. Motion checks remain discrete.
    """

    def __init__(
        self,
        bodies: Sequence[CollisionBody],
        *,
        ignored_pairs: Sequence[tuple[str, str]] = (),
        clearance_m: float = 0.0,
    ) -> None:
        try:
            self._fcl = importlib.import_module("fcl")
        except ImportError as error:
            raise UnsupportedFeasibilityError(
                "Install embodichain[feasibility] for Python-FCL checks"
            ) from error
        self.bodies = tuple(bodies)
        if not self.bodies or len({b.collision_id for b in self.bodies}) != len(
            self.bodies
        ):
            raise ValueError("collision bodies must have unique IDs and be nonempty")
        ids = {b.collision_id for b in self.bodies}
        self._aliases = {}
        for body in self.bodies:
            self._aliases.setdefault(body.frame_uid, set()).add(body.collision_id)
        if any(
            token in ids and targets != {token}
            for token, targets in self._aliases.items()
        ):
            raise ValueError(
                "Physical collision IDs and frame aliases must be unambiguous"
            )
        for collision_id in ids:
            self._aliases.setdefault(collision_id, {collision_id})
        self.ignored_pairs = frozenset(self._expand_pairs(ignored_pairs))
        if not math.isfinite(clearance_m) or clearance_m < 0:
            raise ValueError("collision margin must be finite and nonnegative")
        self.clearance_m = clearance_m
        self.last_pair = None
        self._objects, self._corners = [], []
        self._meshes, self._witnesses = [], []
        for body in self.bodies:
            fcl = self._fcl
            if body.kind == "halfspace":
                geometry = fcl.Halfspace(
                    np.array(body.dimensions[:3]), body.dimensions[3]
                )
                self._corners.append(None)
                self._meshes.append(None)
                self._witnesses.append(np.zeros((1, 3)))
                self._objects.append(fcl.CollisionObject(geometry))
                continue
            if body.kind == "mesh":
                import trimesh

                mesh = trimesh.Trimesh(
                    vertices=np.array(body.vertices),
                    faces=np.array(body.triangles),
                    process=True,
                )
                closed = bool(mesh.is_watertight and mesh.is_winding_consistent)
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                if closed and mesh.is_convex:
                    packed = np.column_stack(
                        (np.full(len(faces), 3, dtype=np.int64), faces)
                    ).reshape(-1)
                    geometry = fcl.Convex(vertices, len(faces), packed)
                else:
                    geometry = fcl.BVHModel()
                    geometry.beginModel(len(vertices), len(faces))
                    geometry.addSubModel(vertices, faces.astype(np.int32))
                    geometry.endModel()
                components = trimesh.graph.connected_components(
                    mesh.face_adjacency, nodes=np.arange(len(faces))
                )
                witnesses = np.array(
                    [vertices[faces[int(component[0]), 0]] for component in components]
                )
                self._witnesses.append(witnesses)
                self._meshes.append(
                    (
                        closed,
                        torch.tensor(vertices, dtype=torch.float64),
                        torch.tensor(faces, dtype=torch.int64),
                    )
                )
                lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            else:
                factory = {
                    "box": fcl.Box,
                    "sphere": fcl.Sphere,
                    "capsule": fcl.Capsule,
                    "cylinder": fcl.Cylinder,
                }[body.kind]
                geometry = factory(*body.dimensions)
                if body.kind == "box":
                    half = np.array(body.dimensions) / 2
                elif body.kind == "sphere":
                    half = np.full(3, body.dimensions[0])
                else:
                    radius, height = body.dimensions
                    half = np.array(
                        [
                            radius,
                            radius,
                            height / 2 + (radius if body.kind == "capsule" else 0),
                        ]
                    )
                lo, hi = -half, half
                self._meshes.append(None)
                self._witnesses.append(np.zeros((1, 3)))
            corners = np.array(
                [
                    [x, y, z]
                    for x in (lo[0], hi[0])
                    for y in (lo[1], hi[1])
                    for z in (lo[2], hi[2])
                ]
            )
            self._corners.append(corners)
            self._objects.append(fcl.CollisionObject(geometry))

    def _contains(self, index: int, world_points: np.ndarray, pose: np.ndarray) -> bool:
        body = self.bodies[index]
        points = (world_points - pose[:3, 3]) @ pose[:3, :3]
        if body.kind == "mesh":
            closed, vertices, triangles = self._meshes[index]
            if not closed:
                raise UnsupportedFeasibilityError(
                    f"Solid containment is unavailable for open/inconsistently oriented mesh {body.collision_id}"
                )
            return bool(
                points_in_closed_mesh(
                    torch.tensor(points, dtype=torch.float64), vertices, triangles
                ).any()
            )
        if body.kind == "box":
            return bool(
                np.any(np.all(np.abs(points) < np.array(body.dimensions) / 2, axis=1))
            )
        if body.kind == "sphere":
            return bool(np.any(np.linalg.norm(points, axis=1) < body.dimensions[0]))
        radius, height = body.dimensions
        if body.kind == "capsule":
            delta = points.copy()
            delta[:, 2] -= np.clip(points[:, 2], -height / 2, height / 2)
            return bool(np.any(np.linalg.norm(delta, axis=1) < radius))
        return bool(
            np.any(
                (np.linalg.norm(points[:, :2], axis=1) < radius)
                & (np.abs(points[:, 2]) < height / 2)
            )
        )

    def _expand_pairs(self, pairs: Sequence[tuple[str, str]]) -> set[frozenset[str]]:
        aliases = self._aliases
        output = set()
        for pair in pairs:
            if (
                len(pair) != 2
                or pair[0] == pair[1]
                or any(p not in aliases for p in pair)
            ):
                raise ValueError(
                    "contact permissions require two distinct known physical IDs or frames"
                )
            output.update(
                frozenset((a, b))
                for a in aliases[pair[0]]
                for b in aliases[pair[1]]
                if a != b
            )
        return output

    def check(
        self,
        frame_poses: Mapping[str, object],
        *,
        allowed_contacts: Sequence[tuple[str, str]] = (),
        moving_objects: Sequence[str] = (),
        all_pairs: bool = False,
    ) -> tuple[FeasibilityCheck, float]:
        """Validate one synchronized configuration with phase-local permissions.

        Args:
            frame_poses: Current arena transforms for every declared shape frame.
            allowed_contacts: Current-stage explicit physical ID pairs.
            moving_objects: Held/transported object frame UIDs.
            all_pairs: Include non-robot scene overlap checks when true.
        Returns:
            Collision evidence and a nonnegative conservative clearance estimate.
        """
        self.last_pair = None
        allowed = self.ignored_pairs | self._expand_pairs(allowed_contacts)
        bounds, world_poses = [], []
        for body, obj, corners in zip(
            self.bodies, self._objects, self._corners, strict=True
        ):
            if body.frame_uid not in frame_poses:
                raise UnsupportedFeasibilityError(
                    f"Missing physical collision frame {body.frame_uid}"
                )
            pose = (
                np.asarray(_rigid_pose(frame_poses[body.frame_uid])) @ body.local_pose
            )
            world_poses.append(pose)
            obj.setTransform(self._fcl.Transform(pose[:3, :3], pose[:3, 3]))
            if corners is None:
                bounds.append((np.full(3, -np.inf), np.full(3, np.inf)))
            else:
                transformed = corners @ pose[:3, :3].T + pose[:3, 3]
                bounds.append((transformed.min(axis=0), transformed.max(axis=0)))
        clearance = float("inf")
        for i, first in enumerate(self.bodies):
            for j in range(i + 1, len(self.bodies)):
                second = self.bodies[j]
                if (
                    first.frame_uid == second.frame_uid
                    or frozenset((first.collision_id, second.collision_id)) in allowed
                ):
                    continue
                if not (
                    all_pairs
                    or first.robot
                    or second.robot
                    or first.frame_uid in moving_objects
                    or second.frame_uid in moving_objects
                ):
                    continue
                if first.kind == "halfspace" or second.kind == "halfspace":
                    plane_index, shape_index = (
                        (i, j) if first.kind == "halfspace" else (j, i)
                    )
                    plane, body = self.bodies[plane_index], self.bodies[shape_index]
                    if body.kind == "halfspace":
                        raise UnsupportedFeasibilityError(
                            "Pairs of physical halfspaces require an explicit provider"
                        )
                    pp, sp = world_poses[plane_index], world_poses[shape_index]
                    normal = pp[:3, :3] @ np.array(plane.dimensions[:3])
                    offset = plane.dimensions[3] + normal @ pp[:3, 3]
                    local_normal = sp[:3, :3].T @ normal
                    if body.kind == "mesh":
                        support = float((np.array(body.vertices) @ local_normal).min())
                    elif body.kind == "sphere":
                        support = -body.dimensions[0]
                    elif body.kind == "capsule":
                        support = (
                            -body.dimensions[0]
                            - abs(local_normal[2]) * body.dimensions[1] / 2
                        )
                    elif body.kind == "cylinder":
                        support = (
                            -body.dimensions[0]
                            * float(np.linalg.norm(local_normal[:2]))
                            - abs(local_normal[2]) * body.dimensions[1] / 2
                        )
                    else:
                        support = float(
                            (self._corners[shape_index] @ local_normal).min()
                        )
                    distance = support + float(normal @ sp[:3, 3]) - offset
                    clearance = min(clearance, max(0.0, distance))
                    if distance < self.clearance_m - 1e-8:
                        self.last_pair = (first.collision_id, second.collision_id)
                        return FeasibilityCheck(
                            "collision",
                            "failed",
                            f"{first.collision_id} / {second.collision_id}: forbidden collision or insufficient clearance",
                        ), max(0.0, distance)
                    continue
                lo1, hi1 = bounds[i]
                lo2, hi2 = bounds[j]
                gap = float(
                    np.linalg.norm(np.maximum(0.0, np.maximum(lo1 - hi2, lo2 - hi1)))
                )
                if gap > self.clearance_m:
                    clearance = min(clearance, gap)
                    continue
                hit = self._fcl.collide(
                    self._objects[i],
                    self._objects[j],
                    self._fcl.CollisionRequest(),
                    self._fcl.CollisionResult(),
                )
                if not hit and (first.kind == "mesh" or second.kind == "mesh"):
                    for container, source in ((i, j), (j, i)):
                        witnesses = (
                            self._witnesses[source] @ world_poses[source][:3, :3].T
                            + world_poses[source][:3, 3]
                        )
                        if self._contains(container, witnesses, world_poses[container]):
                            hit = 1
                            break
                distance = (
                    0.0
                    if hit
                    else max(
                        0.0,
                        float(self._fcl.distance(self._objects[i], self._objects[j])),
                    )
                )
                clearance = min(clearance, distance)
                if hit or distance < self.clearance_m:
                    self.last_pair = (first.collision_id, second.collision_id)
                    return (
                        FeasibilityCheck(
                            "collision",
                            "failed",
                            f"{first.collision_id} / {second.collision_id}: forbidden collision or insufficient clearance",
                        ),
                        distance,
                    )
        return FeasibilityCheck(
            "collision", "passed", "Discrete physical-geometry check"
        ), (0.0 if math.isinf(clearance) else clearance)
