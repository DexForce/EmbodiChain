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

import math
from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import trimesh

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

__all__ = [
    "collision_shapes_from_entity",
    "collision_shapes_from_articulation_link",
    "PhysicalCollisionGeometry",
    "capture_physical_collision_geometry",
]


@dataclass(frozen=True, slots=True)
class PhysicalCollisionGeometry:
    """Owned cooked convex/box geometry and original body offsets.

    Args:
        meshes: Already-scaled body-local collision meshes with shape poses applied.
        rest_offset: Native rest offset in metres.
        contact_offset: Native contact offset when explicitly requested.
        path: Caller-supplied physical source address for diagnostics.
    """

    meshes: tuple[trimesh.Trimesh, ...]
    rest_offset: float
    contact_offset: float | None
    path: str


def capture_physical_collision_geometry(
    physical_body: object,
    *,
    source_path: str,
    include_contact_offsets: bool = False,
) -> PhysicalCollisionGeometry:
    """Read exact cooked convex/box geometry without an authored-input fallback.

    Args:
        physical_body: Native physical-body port exposing enabled collision shapes.
        source_path: Physical address resolved by the host; not inferred here.
        include_contact_offsets: Read contact offsets in addition to rest offsets.

    Returns:
        Independent float64 geometry and offsets. Missing or collision-disabled
        bodies yield empty geometry, which does not establish clearance.

    Raises:
        ValueError: Shape type, topology, scale or local frame is unsupported.
    """
    meshes = _native_meshes(physical_body)
    return PhysicalCollisionGeometry(
        meshes,
        float(physical_body.get_rest_offset()) if meshes else 0.0,
        (
            float(physical_body.get_contact_offset())
            if meshes and include_contact_offsets
            else None
        ),
        source_path,
    )


def _native_meshes(body: object) -> tuple[trimesh.Trimesh, ...]:
    """Convert the shared native descriptor snapshots without changing precision."""
    if body is None or not body.is_collision_enable():
        return ()
    shapes = _from_physical_body(
        body,
        object_uid="native query",
        dtype=torch.float64,
        convex_only=True,
    )
    result = []
    for shape in shapes:
        if shape.shape_type == RigidBodyShape.BOX:
            half = shape.half_extents.numpy()
            if half.shape != (3,) or not np.isfinite(half).all() or (half <= 0).any():
                raise ValueError("Native box half extents must be finite and positive.")
            mesh = trimesh.creation.box(2 * half)
        else:
            vertices, faces = shape.vertices.numpy(), shape.triangles.numpy()
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or not len(vertices)
                or faces.ndim != 2
                or faces.shape[1] != 3
                or not len(faces)
                or (faces < 0).any()
                or (faces >= len(vertices)).any()
            ):
                raise ValueError("Native mesh topology is unavailable.")
            mesh = trimesh.Trimesh(vertices.copy(), faces.copy(), process=False)
        if not np.isfinite(mesh.vertices).all():
            raise ValueError("Native collision geometry is nonfinite.")
        pose = shape.local_pose.numpy()
        if (
            not np.isfinite(pose).all()
            or not np.allclose(pose[3], (0, 0, 0, 1))
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=2e-5)
        ):
            raise ValueError("Native collision geometry requires a finite rigid pose.")
        mesh.apply_transform(pose)
        result.append(mesh)
    return tuple(result)


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
    collisions = getattr(getattr(entity, "object_desc", None), "collisions", None)
    if collisions and any(
        getattr(kind, "value", kind) == GeometryType.CYLINDER.value
        for item in collisions
        for kind in (getattr(item, "geometry_type", None),)
    ):
        # The legacy native API has no cylinder geometry descriptor. Prefer
        # the per-piece Spawn descriptors, which also preserve compound gaps.
        descriptor_shapes = _from_descriptor(entity)
        if descriptor_shapes is not None:
            return descriptor_shapes
    if physical_body is not None:
        return _from_physical_body(physical_body, object_uid=object_uid)

    descriptor_shapes = _from_descriptor(entity)
    if descriptor_shapes is not None:
        return descriptor_shapes

    if collisions and len(collisions) > 1:
        raise RuntimeError(
            f"RigidObject {object_uid!r} has collision pieces that cannot be "
            "queried independently; refusing a merged collision mesh."
        )

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


def collision_shapes_from_articulation_link(
    entity: object, link_name: str, *, object_uid: str
) -> list[CollisionShapeDesc]:
    """Read materialized physical shapes of one named articulation link.

    Args:
        entity: Spawn-bound articulation or native backend handle.
        link_name: Asset-defined physical link name.
        object_uid: Logical articulation ID used in diagnostics.
    Returns:
        Link-local, already-scaled collision descriptors, including native
        convex/decomposed shapes. A link without collision geometry returns empty.
    Raises:
        RuntimeError: When the backend cannot expose physical collision geometry.
    """
    from ..rigid_object import CollisionShapeDesc

    native = entity.native() if callable(getattr(entity, "native", None)) else entity
    binding = getattr(entity, "_physics_binding", None)
    physical_ports = (
        binding,
        native if native is not entity else None,
        entity if binding is None else None,
    )
    for port in physical_ports:
        getter = getattr(port, "get_physical_body", None)
        if callable(getter):
            body = getter(link_name)
            return (
                []
                if body is None
                else _from_physical_body(body, object_uid=f"{object_uid}/{link_name}")
            )
    native_shapes = _from_newton_link(binding, link_name)
    if native_shapes is not None:
        return native_shapes
    for port in physical_ports:
        getter = getattr(port, "get_collision_mesh", None)
        if callable(getter):
            mesh = getter(link_name)
            if mesh is None:
                return []
            vertices, triangles = mesh
            return [
                CollisionShapeDesc(
                    name="collision_mesh",
                    shape_type=RigidBodyShape.MESH,
                    local_pose=torch.eye(4),
                    vertices=torch.as_tensor(vertices, dtype=torch.float32)
                    .reshape(-1, 3)
                    .clone(),
                    triangles=torch.as_tensor(triangles, dtype=torch.int32)
                    .reshape(-1, 3)
                    .clone(),
                )
            ]
    raise RuntimeError(
        f"Articulation {object_uid!r} cannot expose physical collision geometry for {link_name!r}"
    )


def _from_newton_link(
    binding: object, link_name: str
) -> list[CollisionShapeDesc] | None:
    """Preserve each cooked Newton shape and its actual convex/triangle type.

    Newton's interaction mesh accessor merges shapes, losing convex decomposition.
    This backend adapter reads the prepared model instead; no render mesh or
    inferred convex hull is substituted for the actual physical resource.
    """
    runtime = getattr(binding, "_runtime", None)
    model = getattr(runtime, "model", None)
    links = getattr(binding, "_links", None)
    if model is None or links is None:
        return None
    import newton
    from embodichain.utils.math import matrix_from_quat
    from ..rigid_object import CollisionShapeDesc

    link = next((link for link in links if link.name == link_name), None)
    if link is None:
        return []

    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(
            value.numpy() if hasattr(value, "numpy") else value, dtype=dtype
        ).clone()

    kinds = tensor(model.shape_type, torch.int64)
    flags = tensor(model.shape_flags, torch.int64)
    scales = tensor(model.shape_scale)
    transforms = tensor(model.shape_transform)
    result = []
    mapping = {
        newton.GeoType.CONVEX_MESH: RigidBodyShape.CONVEX,
        newton.GeoType.MESH: RigidBodyShape.MESH,
        newton.GeoType.BOX: RigidBodyShape.BOX,
        newton.GeoType.SPHERE: RigidBodyShape.SPHERE,
        newton.GeoType.CAPSULE: RigidBodyShape.CAPSULE,
    }
    for index, shape_id in enumerate(link.shape_ids):
        if not int(flags[shape_id]) & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        kind = newton.GeoType(int(kinds[shape_id]))
        if kind not in mapping:
            raise RuntimeError(
                f"Unsupported Newton physical collision kind {kind.name}"
            )
        transform = transforms[shape_id]
        pose = torch.eye(4)
        pose[:3, :3] = matrix_from_quat(transform[3:].reshape(1, 4))[0]
        pose[:3, 3] = transform[:3]
        shape = CollisionShapeDesc(f"shape_{index}", mapping[kind], pose)
        scale = scales[shape_id]
        if kind in {newton.GeoType.CONVEX_MESH, newton.GeoType.MESH}:
            source = model.shape_source[shape_id]
            if source is None:
                raise RuntimeError("Newton mesh collision source is missing")
            shape.vertices = tensor(source.vertices).reshape(-1, 3) * scale
            shape.triangles = tensor(source.indices, torch.int32).reshape(-1, 3)
        elif kind == newton.GeoType.BOX:
            shape.half_extents = scale
        else:
            shape.radius = float(scale[0])
            if kind == newton.GeoType.CAPSULE:
                shape.half_height = float(scale[1])
        result.append(shape)
    return result


def _from_physical_body(
    physical_body: object,
    *,
    object_uid: str,
    dtype: torch.dtype = torch.float32,
    convex_only: bool = False,
) -> list[CollisionShapeDesc]:
    """Decode physical shapes once for public snapshots and strict native queries.

    Public snapshots retain their float32 contract. Native convex queries select
    float64 and reject unsupported shapes rather than using a descriptor fallback.
    """
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

        if convex_only and not isinstance(geometry, (BoxGeometry, ConvexMeshGeometry)):
            raise ValueError(
                "Native queries require readable cooked convex or box shapes."
            )

        shape_name = physical_body.get_shape_name(shape_idx) or f"shape_{shape_idx}"
        local_pose = torch.tensor(geometry.local_pose, dtype=dtype)
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
            desc.half_extents = torch.tensor(geometry.half_extents, dtype=dtype)
        elif isinstance(geometry, SphereGeometry):
            desc.radius = float(geometry.radius)
        elif isinstance(geometry, CapsuleGeometry):
            desc.radius = float(geometry.radius)
            desc.half_height = float(geometry.half_height)
        elif isinstance(
            geometry, (ConvexMeshGeometry, TriangleMeshGeometry, SDFGeometry)
        ):
            vertices = torch.tensor(geometry.vertices, dtype=dtype)
            triangles = torch.tensor(
                geometry.triangles, dtype=torch.int64 if convex_only else torch.int32
            )
            if convex_only and (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or triangles.ndim != 2
                or triangles.shape[1] != 3
            ):
                raise ValueError("Native mesh topology is unavailable.")
            vertices = vertices.reshape(-1, 3)
            triangles = triangles.reshape(-1, 3)
            scale = getattr(geometry, "scale", None)
            if convex_only:
                scale_tensor = torch.as_tensor(scale, dtype=dtype)
                if (
                    scale_tensor.shape != (3,)
                    or not torch.isfinite(scale_tensor).all()
                    or (scale_tensor <= 0).any()
                ):
                    raise ValueError("Native shape scale is unavailable.")
            if scale is not None:
                vertices = vertices * torch.tensor(scale, dtype=dtype).reshape(1, 3)
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
    # Match native shape queries: disabling contact response keeps geometry
    # available to planners, including moving obstacles that exert no impulses.
    for shape_idx, collision in enumerate(collisions):
        geometry_type = getattr(collision, "geometry_type", None)
        geometry_type = getattr(geometry_type, "value", geometry_type)
        local_pose = (
            torch.as_tensor(
                getattr(collision, "local_transform", torch.eye(4)),
                dtype=torch.float32,
            )
            .reshape(4, 4)
            .clone()
        )
        basis = body_scale[:, None] * local_pose[:3, :3]
        shape_scale = torch.linalg.vector_norm(basis, dim=0)
        if bool((shape_scale <= 0).any()):
            raise ValueError("Collision shape scale must be positive.")
        rotation = basis / shape_scale[None, :]
        if not torch.allclose(
            rotation.T @ rotation, torch.eye(3), atol=1e-5, rtol=1e-5
        ):
            raise ValueError(
                "A rotated collision shape with nonuniform body scale cannot "
                "be represented by a rigid planner pose."
            )
        local_pose[:3, :3] = rotation
        local_pose[:3, 3] *= body_scale
        name = str(getattr(collision, "segment_name", None) or f"shape_{shape_idx}")

        if geometry_type == GeometryType.CUBE.value:
            size = torch.as_tensor(collision.size, dtype=torch.float32).reshape(3)
            shapes.append(
                CollisionShapeDesc(
                    name=name,
                    shape_type=RigidBodyShape.BOX,
                    local_pose=local_pose.clone(),
                    half_extents=(0.5 * size * shape_scale).clone(),
                )
            )
            continue
        if geometry_type == GeometryType.SPHERE.value:
            if not torch.allclose(shape_scale, shape_scale[0].expand_as(shape_scale)):
                raise ValueError(
                    "Nonuniform sphere scaling is not supported by collision planning."
                )
            shapes.append(
                CollisionShapeDesc(
                    name=name,
                    shape_type=RigidBodyShape.SPHERE,
                    local_pose=local_pose.clone(),
                    radius=float(collision.radius) * float(shape_scale[0]),
                )
            )
            continue
        if geometry_type == GeometryType.CAPSULE.value:
            if not torch.allclose(shape_scale, shape_scale[0].expand_as(shape_scale)):
                raise ValueError(
                    "Nonuniform capsule scaling is not supported by collision planning."
                )
            scale = float(shape_scale[0])
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
        if geometry_type == GeometryType.CYLINDER.value:
            if not torch.allclose(shape_scale, shape_scale[0].expand_as(shape_scale)):
                raise ValueError(
                    "Nonuniform cylinder scaling is not supported by collision planning."
                )
            from dexsim.utility.primitive_mesh import generate_cylinder_mesh

            # The simulator retains an analytic cylinder. The planner consumes
            # a conservative circumscribed convex proxy at its voxel resolution.
            divisions = 128
            radius = float(collision.radius) / math.cos(math.pi / divisions)
            vertices, triangles, _, _ = generate_cylinder_mesh(
                radius,
                2.0 * float(collision.half_height),
                divide=divisions,
                axis=collision.axis or "z",
            )
            shapes.append(
                CollisionShapeDesc(
                    name=name,
                    shape_type=RigidBodyShape.CONVEX,
                    local_pose=local_pose,
                    vertices=torch.as_tensor(vertices, dtype=torch.float32)
                    .reshape(-1, 3)
                    .mul(shape_scale.reshape(1, 3))
                    .clone(),
                    triangles=torch.as_tensor(triangles, dtype=torch.int32)
                    .reshape(-1, 3)
                    .clone(),
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
                        .mul(shape_scale.reshape(1, 3))
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
