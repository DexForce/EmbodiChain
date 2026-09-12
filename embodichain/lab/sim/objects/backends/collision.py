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
"""Backend compatibility helpers for rigid-object collision geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from dexsim.engine import (
    BoxGeometry,
    CapsuleGeometry,
    ConvexMeshGeometry,
    PlaneGeometry,
    SDFGeometry,
    SphereGeometry,
    TriangleMeshGeometry,
)
from dexsim.spawn.descs import GeometryType
from dexsim.types import RigidBodyShape

if TYPE_CHECKING:
    from ..rigid_object import CollisionShapeDesc

__all__ = ["collision_shapes_from_entity"]


def collision_shapes_from_entity(
    entity: object,
    *,
    object_uid: str,
) -> list[CollisionShapeDesc]:
    """Read one entity's physical collision shapes across DexSim backends.

    DexSim versions before the SpawnedRigidBody facade exposed a native
    ``get_physical_body`` method. Newer facades retain collision descriptors
    and expose Newton's collision mesh through the private binding instead.
    This adapter keeps those backend-specific details out of ``RigidObject``.

    Args:
        entity: Materialized DexSim rigid-body handle.
        object_uid: Logical EmbodiChain object ID used in diagnostics.

    Returns:
        Planner-independent collision-shape snapshots.

    Raises:
        RuntimeError: If no supported physical collision representation exists.
    """
    from ..rigid_object import CollisionShapeDesc

    get_physical_body = getattr(entity, "get_physical_body", None)
    physical_body = get_physical_body() if callable(get_physical_body) else None
    if physical_body is None:
        native = entity.native() if callable(getattr(entity, "native", None)) else None
        native_get_physical_body = getattr(native, "get_physical_body", None)
        physical_body = (
            native_get_physical_body() if callable(native_get_physical_body) else None
        )
    if physical_body is not None:
        return _from_physical_body(physical_body, object_uid=object_uid)

    descriptor_shapes = _from_descriptor(entity)
    if descriptor_shapes is not None:
        return descriptor_shapes

    # Newton's runtime binding exposes a complete local collision mesh even
    # when it does not expose the legacy physical-body shape API. This is a
    # fallback for mesh/convex/SDF shapes and future geometry kinds.
    binding = getattr(entity, "_physics_binding", None)
    get_collision_mesh = getattr(binding, "get_collision_mesh", None)
    if callable(get_collision_mesh):
        collision_mesh = get_collision_mesh()
        if collision_mesh is not None:
            vertices, triangles = collision_mesh
            return [
                CollisionShapeDesc(
                    name="collision_mesh",
                    shape_type=RigidBodyShape.MESH,
                    local_pose=torch.eye(4, dtype=torch.float32),
                    vertices=torch.as_tensor(vertices, dtype=torch.float32)
                    .reshape(-1, 3)
                    .clone(),
                    triangles=torch.as_tensor(triangles, dtype=torch.int32)
                    .reshape(-1, 3)
                    .clone(),
                )
            ]

    raise RuntimeError(
        f"RigidObject {object_uid!r} has no accessible physical collision "
        "descriptor."
    )


def _from_physical_body(
    physical_body: object,
    *,
    object_uid: str,
) -> list[CollisionShapeDesc]:
    """Extract collision descriptors from the legacy native body API."""
    from ..rigid_object import CollisionShapeDesc

    shape_count = int(physical_body.get_shape_count())
    shapes: list[CollisionShapeDesc] = []
    for shape_idx in range(shape_count):
        try:
            geometry = physical_body.get_shape_geometry(shape_idx)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"DexSim could not expose collision shape {shape_idx} for "
                f"RigidObject {object_uid!r}. SDF/custom shapes require a "
                "geometry descriptor or canonical collision mesh."
            ) from exc
        if geometry is None:
            raise RuntimeError(
                f"DexSim returned no geometry for collision shape {shape_idx} "
                f"of RigidObject {object_uid!r}."
            )

        shape_name = physical_body.get_shape_name(shape_idx) or f"shape_{shape_idx}"
        local_pose = torch.tensor(geometry.local_pose, dtype=torch.float32)
        if local_pose.shape != (4, 4):
            raise RuntimeError(
                f"DexSim collision shape {shape_idx} of {object_uid!r} returned "
                f"local_pose shape {tuple(local_pose.shape)}, expected (4, 4)."
            )
        desc = CollisionShapeDesc(
            name=str(shape_name),
            shape_type=_collision_shape_type(geometry),
            local_pose=local_pose.clone(),
        )
        if isinstance(geometry, BoxGeometry):
            desc.half_extents = torch.tensor(geometry.half_extents, dtype=torch.float32)
        elif isinstance(geometry, SphereGeometry):
            desc.radius = float(geometry.radius)
        elif isinstance(geometry, CapsuleGeometry):
            desc.radius = float(geometry.radius)
            desc.half_height = float(geometry.half_height)
        elif isinstance(
            geometry, (ConvexMeshGeometry, TriangleMeshGeometry, SDFGeometry)
        ):
            vertices = torch.tensor(geometry.vertices, dtype=torch.float32).reshape(
                -1, 3
            )
            triangles = torch.tensor(geometry.triangles, dtype=torch.int32).reshape(
                -1, 3
            )
            scale = getattr(geometry, "scale", None)
            if scale is not None:
                vertices = vertices * torch.tensor(scale, dtype=torch.float32).reshape(
                    1, 3
                )
            desc.vertices = vertices
            desc.triangles = triangles
        shapes.append(desc)
    return shapes


def _from_descriptor(entity: object) -> list[CollisionShapeDesc] | None:
    """Translate retained Spawn descriptors when native shape access is absent."""
    from ..rigid_object import CollisionShapeDesc

    object_desc = getattr(entity, "object_desc", None)
    collisions = getattr(object_desc, "collisions", None)
    if not collisions:
        return None

    body_scale = torch.as_tensor(
        getattr(object_desc, "body_scale", (1.0, 1.0, 1.0)),
        dtype=torch.float32,
    ).reshape(3)
    shapes: list[CollisionShapeDesc] = []
    for shape_idx, collision in enumerate(collisions):
        if getattr(collision, "enable_collision", True) is False:
            continue
        geometry_type = getattr(collision, "geometry_type", None)
        geometry_type = getattr(geometry_type, "value", geometry_type)
        local_pose = torch.as_tensor(
            getattr(collision, "local_transform", torch.eye(4)),
            dtype=torch.float32,
        ).reshape(4, 4)
        name = str(getattr(collision, "segment_name", None) or f"shape_{shape_idx}")

        if geometry_type == GeometryType.CUBE.value:
            size = torch.as_tensor(collision.size, dtype=torch.float32).reshape(3)
            shapes.append(
                CollisionShapeDesc(
                    name=name,
                    shape_type=RigidBodyShape.BOX,
                    local_pose=local_pose.clone(),
                    half_extents=(0.5 * size * body_scale).clone(),
                )
            )
            continue
        if geometry_type == GeometryType.SPHERE.value:
            shapes.append(
                CollisionShapeDesc(
                    name=name,
                    shape_type=RigidBodyShape.SPHERE,
                    local_pose=local_pose.clone(),
                    radius=float(collision.radius) * float(body_scale[0]),
                )
            )
            continue
        if geometry_type == GeometryType.CAPSULE.value:
            scale = float(body_scale[0])
            shapes.append(
                CollisionShapeDesc(
                    name=name,
                    shape_type=RigidBodyShape.CAPSULE,
                    local_pose=local_pose.clone(),
                    radius=float(collision.radius) * scale,
                    half_height=float(collision.half_height) * scale,
                )
            )
            continue
        if geometry_type == GeometryType.PLANE.value:
            shapes.append(
                CollisionShapeDesc(
                    name=name,
                    shape_type=RigidBodyShape.PLANE,
                    local_pose=local_pose.clone(),
                )
            )
            continue

        vertices = getattr(collision, "vertices", None)
        triangles = getattr(collision, "triangles", None)
        if vertices is not None and triangles is not None:
            shapes.append(
                CollisionShapeDesc(
                    name=name,
                    shape_type=RigidBodyShape.MESH,
                    local_pose=local_pose.clone(),
                    vertices=(
                        torch.as_tensor(vertices, dtype=torch.float32)
                        .reshape(-1, 3)
                        .mul(body_scale.reshape(1, 3))
                        .clone()
                    ),
                    triangles=torch.as_tensor(triangles, dtype=torch.int32)
                    .reshape(-1, 3)
                    .clone(),
                )
            )
            continue
        return None
    return shapes


def _collision_shape_type(geometry: object) -> RigidBodyShape:
    """Map a concrete DexSim geometry descriptor to its shape enum."""
    if isinstance(geometry, BoxGeometry):
        return RigidBodyShape.BOX
    if isinstance(geometry, PlaneGeometry):
        return RigidBodyShape.PLANE
    if isinstance(geometry, SphereGeometry):
        return RigidBodyShape.SPHERE
    if isinstance(geometry, CapsuleGeometry):
        return RigidBodyShape.CAPSULE
    if isinstance(geometry, ConvexMeshGeometry):
        return RigidBodyShape.CONVEX
    if isinstance(geometry, TriangleMeshGeometry):
        return RigidBodyShape.MESH
    if isinstance(geometry, SDFGeometry):
        return RigidBodyShape.SDF
    raise RuntimeError(
        f"Unsupported DexSim collision geometry descriptor "
        f"{type(geometry).__name__}."
    )
