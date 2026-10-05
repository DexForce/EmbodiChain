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
"""Source-bound E8 collision inputs, scaled once into their rigid-link frames."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .twist_binding import KnobBinding, _CALIBRATIONS

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class LinkMesh:
    """Immutable authored collider input, not a backend cooked collision shape."""

    path: str
    owner_path: str
    vertices: np.ndarray
    faces: np.ndarray
    approximation: str


@dataclass(frozen=True, slots=True)
class TwistGeometry:
    """Keep calibrated grip and each body's collider inputs distinct."""

    grip: LinkMesh
    target_collisions: tuple[LinkMesh, ...]
    parent_collisions: tuple[LinkMesh, ...]
    source_sha256: str
    scale: float
    target_body_path: str
    parent_body_path: str
    articulation_collisions: tuple[LinkMesh, ...]


def _owner(prim: Any) -> Any:
    from pxr import UsdPhysics

    current = prim
    while current and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def _triangles(
    vertices: np.ndarray, counts: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or counts.ndim != 1
        or not len(counts)
        or (counts < 3).any()
        or indices.ndim != 1
        or not len(indices)
        or counts.sum() != len(indices)
        or indices.min() < 0
        or indices.max() >= len(vertices)
    ):
        raise ValueError("E8 collision mesh has invalid topology.")
    faces: list[tuple[int, int, int]] = []
    offset = 0
    for count in counts:
        polygon = indices[offset : offset + count]
        points = vertices[polygon]
        if len(np.unique(polygon)) != count:
            raise ValueError("E8 collision polygon repeats a vertex.")
        edges = np.roll(points, -1, axis=0) - points
        turns = np.cross(edges, np.roll(edges, -1, axis=0))
        normal = turns[np.argmax(np.linalg.norm(turns, axis=1))]
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            raise ValueError("E8 collision polygon is degenerate.")
        normal /= norm
        tolerance = 1e-8 * max(float(np.ptp(points, axis=0).max()), 1e-12)
        if (
            np.abs((points - points[0]) @ normal).max() > tolerance
            or (turns @ normal < -(tolerance**2)).any()
        ):
            raise ValueError("E8 collision polygon must be planar and convex.")
        # Match the E6 fan for convex authored polygons; concave inputs fail closed.
        faces.extend(
            (int(polygon[0]), int(polygon[i]), int(polygon[i + 1]))
            for i in range(1, count - 1)
        )
        offset += int(count)
    result = np.asarray(faces, dtype=np.int64)
    triangle_points = vertices[result]
    if (
        np.linalg.norm(
            np.cross(
                triangle_points[:, 1] - triangle_points[:, 0],
                triangle_points[:, 2] - triangle_points[:, 0],
            ),
            axis=1,
        )
        == 0.0
    ).any():
        raise ValueError("E8 collision triangulation contains a degenerate triangle.")
    return result


def _uniform_transform(transform: np.ndarray, *, rigid: bool) -> None:
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
    ):
        raise ValueError("E8 collision geometry has an invalid transform.")
    linear = transform[:3, :3]
    gram = linear.T @ linear
    scale_squared = float(np.trace(gram) / 3.0)
    if (
        scale_squared <= 0.0
        or np.linalg.det(linear) <= 0.0
        or not np.allclose(gram, np.eye(3) * scale_squared, atol=1e-10, rtol=1e-6)
        or (rigid and not math.isclose(scale_squared, 1.0, rel_tol=1e-6))
    ):
        raise ValueError(
            "E8 requires unscaled rigid frames and positive uniform mesh scales."
        )


def _mesh(prim: Any, body: Any, cache: Any, scale: float) -> LinkMesh:
    from pxr import UsdGeom, UsdPhysics

    mesh = UsdGeom.Mesh(prim)
    if mesh.GetHoleIndicesAttr().Get():
        raise ValueError("E8 collision mesh holes require explicit qualified topology.")
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    faces = _triangles(vertices, counts, indices)
    body_pose = np.asarray(cache.GetLocalToWorldTransform(body), dtype=float).T
    _uniform_transform(body_pose, rigid=True)
    transform = (
        np.linalg.inv(body_pose)
        @ np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
    )
    _uniform_transform(transform, rigid=False)
    # Mesh xforms include authored translations; deployment scale affects both.
    vertices = (vertices @ transform[:3, :3].T + transform[:3, 3]) * scale
    if not np.isfinite(vertices).all():
        raise ValueError("E8 scaled collision vertices must remain finite.")
    # Bytes-backed arrays cannot be made writable through setflags().
    vertices = np.frombuffer(vertices.tobytes(), dtype=np.float64).reshape(-1, 3)
    faces = np.frombuffer(faces.tobytes(), dtype=np.int64).reshape(-1, 3)
    approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
    return LinkMesh(
        str(prim.GetPath()), str(body.GetPath()), vertices, faces, approximation
    )


def load_twist_geometry(path: str | Path, binding: KnobBinding) -> TwistGeometry:
    """Read qualified collider inputs in separate native rigid-link frames.

    The calibrated grip is never replaced by a union of cap, shaft or pointer.
    Vertices already include ``binding.scale``; consumers apply only each
    owning link's observed rigid pose, not another asset-scale multiplication.
    ``approximation`` retains USD cooking semantics, so these triangles alone
    do not certify backend convex decomposition or continuous path clearance.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != binding.source_sha256 or digest not in _CALIBRATIONS:
        raise ValueError("E8 collision source does not match its audited binding.")
    if (
        type(binding.scale) not in (int, float)
        or not math.isfinite(binding.scale)
        or binding.scale <= 0.0
    ):
        raise ValueError("E8 collision deployment scale must be finite and positive.")
    joint_name, grip_name, _, _ = _CALIBRATIONS[digest]
    if binding.joint != joint_name:
        raise ValueError("E8 collision binding selects a different calibrated joint.")
    stage = Usd.Stage.Open(str(path))
    if stage is None or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("E8 collision source must be metre-authored USD.")
    if any(
        layer not in (stage.GetRootLayer(), stage.GetSessionLayer())
        for layer in stage.GetUsedLayers()
    ):
        raise ValueError("E8 collision source must be self-contained and hash-bound.")
    joints = [prim for prim in stage.Traverse() if prim.GetName() == joint_name]
    if len(joints) != 1 or not joints[0].IsA(UsdPhysics.RevoluteJoint):
        raise ValueError("E8 collision source requires one calibrated revolute joint.")
    joint = UsdPhysics.RevoluteJoint(joints[0])
    parent_paths, target_paths = (
        joint.GetBody0Rel().GetTargets(),
        joint.GetBody1Rel().GetTargets(),
    )
    if (
        len(parent_paths) != 1
        or len(target_paths) != 1
        or joint.GetJointEnabledAttr().Get() is not True
    ):
        raise ValueError("E8 collision source requires one enabled parent/child joint.")
    parent, target = (
        stage.GetPrimAtPath(parent_paths[0]),
        stage.GetPrimAtPath(target_paths[0]),
    )
    for body, name in ((parent, binding.parent), (target, binding.link)):
        if (
            not body
            or body.GetName() != name
            or not body.HasAPI(UsdPhysics.RigidBodyAPI)
            or UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get() is not True
            or sum(
                prim.GetName() == name and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                for prim in stage.Traverse()
            )
            != 1
        ):
            raise ValueError("E8 collision ownership does not match its bound link.")
    roots = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        and parent.GetPath().HasPrefix(prim.GetPath())
        and target.GetPath().HasPrefix(prim.GetPath())
    ]
    if len(roots) != 1 or parent == target:
        raise ValueError("E8 collision source requires one bound articulation root.")
    root_path = roots[0].GetPath()
    cache = UsdGeom.XformCache()
    groups: dict[str, list[LinkMesh]] = {
        str(target.GetPath()): [],
        str(parent.GetPath()): [],
    }
    grip: LinkMesh | None = None
    articulation_meshes: list[LinkMesh] = []
    for prim in stage.Traverse():
        owner = _owner(prim)
        if owner is None or not owner.GetPath().HasPrefix(root_path):
            continue
        calibrated = prim.GetName() == grip_name and owner in (parent, target)
        collision = (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
        )
        if calibrated and (owner != target or not collision):
            raise ValueError("E8 calibrated grip must be an enabled target collider.")
        if not collision:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError("E8 collision source has an unsupported collider type.")
        value = _mesh(prim, owner, cache, float(binding.scale))
        articulation_meshes.append(value)
        if owner in (parent, target):
            groups[str(owner.GetPath())].append(value)
        if calibrated:
            if grip is not None:
                raise ValueError("E8 calibrated grip collider must be unambiguous.")
            grip = value
    target_meshes = tuple(groups[str(target.GetPath())])
    parent_meshes = tuple(groups[str(parent.GetPath())])
    if grip is None or not target_meshes or not parent_meshes:
        raise ValueError("E8 requires grip, target and parent collision meshes.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("E8 collision source changed while loading.")
    return TwistGeometry(
        grip,
        target_meshes,
        parent_meshes,
        digest,
        float(binding.scale),
        str(target.GetPath()),
        str(parent.GetPath()),
        tuple(articulation_meshes),
    )


def _vector(text: str, label: str, *, positive: bool = False) -> np.ndarray:
    try:
        value = np.asarray([float(item) for item in text.split()], dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"E8 {label} must contain three finite numbers.") from error
    if (
        value.shape != (3,)
        or not np.isfinite(value).all()
        or (positive and (value <= 0.0).any())
    ):
        raise ValueError(f"E8 {label} must contain three valid numbers.")
    return value


def load_robot_link_meshes(
    path: str | Path,
    link_names: tuple[str, ...],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, tuple[LinkMesh, ...]]:
    """Read URDF collision inputs in each robot link's native local frame.

    Visuals are never used. Collision mesh scale precedes its authored origin;
    robot body scale then affects both vertices and collision-origin translation.
    A consumer applies the observed rigid link pose without any knob-body scale.
    This is source input geometry, not a backend cooked-shape certificate.
    """
    import xml.etree.ElementTree as ET

    import trimesh
    from scipy.spatial.transform import Rotation

    path = Path(path)
    asset_scale = np.asarray(scale, dtype=np.float64)
    if (
        asset_scale.shape != (3,)
        or not np.isfinite(asset_scale).all()
        or (asset_scale <= 0.0).any()
        or not link_names
        or any(type(name) is not str or not name for name in link_names)
        or len(set(link_names)) != len(link_names)
    ):
        raise ValueError("E8 robot collision requires positive scale and exact links.")
    document = ET.parse(path).getroot()
    result: dict[str, tuple[LinkMesh, ...]] = {}
    for name in link_names:
        matches = [
            link for link in document.findall("link") if link.get("name") == name
        ]
        if len(matches) != 1:
            raise ValueError("E8 robot collision link must be unambiguous.")
        values: list[LinkMesh] = []
        for index, collision in enumerate(matches[0].findall("collision")):
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError("E8 robot collision requires one explicit geometry.")
            element = geometry[0]
            if element.tag == "mesh":
                source = element.get("filename", "")
                if not source or "://" in source:
                    raise ValueError("E8 robot collision mesh must be a local path.")
                mesh_path = path.parent / source
                shape = trimesh.load(mesh_path, force="mesh", process=False)
                mesh_scale = _vector(
                    element.get("scale", "1 1 1"), "robot mesh scale", positive=True
                )
                vertices = np.asarray(shape.vertices, dtype=np.float64) * mesh_scale
                label = str(mesh_path.resolve())
                approximation = "urdfMeshInput"
            elif element.tag == "box":
                size = _vector(element.get("size", ""), "robot box size", positive=True)
                shape = trimesh.creation.box(extents=size)
                vertices = np.asarray(shape.vertices, dtype=np.float64)
                label = f"box:{' '.join(map(str, size))}"
                approximation = "urdfBoxInput"
            else:
                raise ValueError("E8 robot collision supports only meshes and boxes.")
            faces = np.asarray(shape.faces, dtype=np.int64)
            if faces.ndim != 2 or faces.shape[1:] != (3,):
                raise ValueError("E8 robot collision mesh must be triangulated.")
            faces = _triangles(vertices, np.full(len(faces), 3), faces.reshape(-1))
            origin = collision.find("origin")
            translation, rotation = np.zeros(3), np.eye(3)
            if origin is not None:
                translation = _vector(origin.get("xyz", "0 0 0"), "robot origin xyz")
                rotation = Rotation.from_euler(
                    "xyz", _vector(origin.get("rpy", "0 0 0"), "robot origin rpy")
                ).as_matrix()
            vertices = (vertices @ rotation.T + translation) * asset_scale
            if not np.isfinite(vertices).all():
                raise ValueError(
                    "E8 robot collision scale produced nonfinite vertices."
                )
            values.append(
                LinkMesh(
                    f"{path.resolve()}#{name}/collision[{index}]:{label}",
                    name,
                    np.frombuffer(vertices.tobytes(), dtype=np.float64).reshape(-1, 3),
                    np.frombuffer(faces.tobytes(), dtype=np.int64).reshape(-1, 3),
                    approximation,
                )
            )
        if not values:
            raise ValueError("E8 robot link has no collision geometry.")
        result[name] = tuple(values)
    return result
