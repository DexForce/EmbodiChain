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

"""Bake an affine footprint fit into rigid USD links and joint frames."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.scene.articulation_geometry import read_articulation_geometry
from embodichain.lab.sim.spawn.source_geometry import read_usd_source

if TYPE_CHECKING:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

__all__ = ["bake_articulation_scale"]


def _has_unbakeable_visual_geometry(stage: Usd.Stage) -> bool:
    """Identify visual primitives under rigid links that need the legacy fit."""
    from pxr import Usd, UsdGeom, UsdPhysics

    prims = list(Usd.PrimRange(stage.GetDefaultPrim()))
    bodies = {
        prim.GetPath(): prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    return any(
        (prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer))
        and not prim.IsA(UsdGeom.Mesh)
        and (
            not prim.HasAPI(UsdPhysics.CollisionAPI)
            or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
        )
        and _body_owner(prim, bodies) is not None
        for prim in prims
    )


def bake_articulation_scale(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    native_scale_matrix: np.ndarray,
) -> Path:
    """Write a scaled USDC copy with exact mesh geometry at the reset state.

    The scale is a symmetric positive-definite linear map in the native
    root-body frame. Each link stays rigid after reset. Revolute axes follow
    the transformed hinge lines, so points moving off the reference pose
    follow rigid circular trajectories in the new mechanism.

    Args:
        source_path: Self-contained, Z-up, metre-authored mesh articulation.
        destination_path: Distinct destination for the generated USDC copy.
        native_scale_matrix: Root-local 3×3 affine scale, without rotation.

    Returns:
        The path of the generated asset.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    scale = np.asarray(native_scale_matrix, dtype=float)
    if (
        scale.shape != (3, 3)
        or not np.isfinite(scale).all()
        or not np.allclose(scale, scale.T, rtol=0.0, atol=1e-8)
        or (np.linalg.eigvalsh(scale) <= 0.0).any()
    ):
        raise ValueError("Articulation scaling needs a positive-definite 3×3 map.")
    source = read_usd_source(source_path)
    destination = Path(destination_path).resolve()
    if destination == source.path or destination.suffix != ".usdc":
        raise ValueError("Articulation scaling needs a distinct USDC destination.")
    geometry = read_articulation_geometry(source, reset_to_default=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}-",
        suffix=".usdc",
        dir=destination.parent,
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(source.raw)
    try:
        stage = Usd.Stage.Open(str(temporary_path))
        root = stage.GetDefaultPrim()
        prims = list(Usd.PrimRange(root))
        bodies = {
            prim.GetPath(): prim
            for prim in prims
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        }
        root_body = next(
            prim for prim in bodies.values() if prim.GetName() == geometry.root_link
        )
        before = UsdGeom.XformCache()
        root_world = _matrix(before.GetLocalToWorldTransform(root_body))
        authored_world = {
            path: _matrix(before.GetLocalToWorldTransform(body))
            for path, body in bodies.items()
        }
        initial_poses = {
            path: geometry.link_poses[body.GetName()] for path, body in bodies.items()
        }
        fitted_poses = {path: pose.copy() for path, pose in initial_poses.items()}
        for pose in fitted_poses.values():
            pose[:3, 3] = scale @ pose[:3, 3]

        meshes = []
        for prim in prims:
            owner_path = _body_owner(prim, bodies)
            if owner_path is None:
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    raise ValueError("Articulation collider has no link owner.")
                continue
            if prim.IsA(UsdGeom.Mesh):
                meshes.append(
                    (prim, owner_path, _matrix(before.GetLocalToWorldTransform(prim)))
                )
            elif (
                prim.IsA(UsdGeom.Gprim)
                or prim.IsA(UsdGeom.PointInstancer)
                or prim.HasAPI(UsdPhysics.CollisionAPI)
            ):
                raise ValueError(
                    f"Articulation scaling requires mesh geometry: {prim.GetPath()}."
                )

        for path, body in sorted(
            bodies.items(), key=lambda item: item[0].pathElementCount
        ):
            xform = UsdGeom.Xformable(body)
            if not xform or xform.TransformMightBeTimeVarying():
                raise ValueError(f"Articulation link transform must be static: {path}.")
            reset_xform = xform.GetResetXformStack()
            parent_world = (
                np.eye(4)
                if reset_xform
                else _matrix(
                    UsdGeom.XformCache().GetLocalToWorldTransform(body.GetParent())
                )
            )
            local_pose = np.linalg.solve(parent_world, root_world @ fitted_poses[path])
            xform.MakeMatrixXform().Set(Gf.Matrix4d(local_pose.T.tolist()))
            xform.SetResetXformStack(reset_xform)
            link_rotation = initial_poses[path][:3, :3]
            _scale_mass_properties(body, link_rotation.T @ scale @ link_rotation)

        after = UsdGeom.XformCache()
        affine = np.eye(4)
        affine[:3, :3] = scale
        for prim, owner, old_mesh_world in meshes:
            # Old mesh points are first posed at the native reset. Transform the
            # whole shape there, then express it in the new mesh-local frame.
            old_mesh_local = np.linalg.solve(authored_world[owner], old_mesh_world)
            fitted_mesh_world = (
                root_world @ affine @ initial_poses[owner] @ old_mesh_local
            )
            fitted_mesh_local = np.linalg.solve(
                _matrix(after.GetLocalToWorldTransform(prim)), fitted_mesh_world
            )
            _scale_mesh(UsdGeom.Mesh(prim), fitted_mesh_local)

        for prim in prims:
            if prim.IsA(UsdPhysics.Joint):
                _scale_joint(
                    UsdPhysics.Joint(prim),
                    root_body.GetPath(),
                    initial_poses,
                    fitted_poses,
                    scale,
                )
        if not stage.GetRootLayer().Save():
            raise OSError(f"Cannot save scaled articulation: {temporary_path}")
        source.assert_current()
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _matrix(value: object) -> np.ndarray:
    """Convert an OpenUSD row-vector transform to a column-vector transform."""
    return np.asarray(value, dtype=float).T


def _body_owner(prim: Usd.Prim, bodies: dict[Sdf.Path, Usd.Prim]) -> Sdf.Path | None:
    path = prim.GetPath()
    while not path.IsAbsoluteRootPath():
        if path in bodies:
            return path
        path = path.GetParentPath()
    return None


def _scale_mesh(mesh: UsdGeom.Mesh, transform: np.ndarray) -> None:
    from pxr import Vt

    points_attr = mesh.GetPointsAttr()
    normals_attr = mesh.GetNormalsAttr()
    if points_attr.GetNumTimeSamples() or normals_attr.GetNumTimeSamples():
        raise ValueError(f"Articulation meshes must be static: {mesh.GetPath()}.")
    points = np.asarray(points_attr.Get(), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"Articulation mesh has no points: {mesh.GetPath()}.")
    linear = transform[:3, :3]
    points = points @ linear.T + transform[:3, 3]
    points = np.ascontiguousarray(points, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError(
            f"Scaled articulation mesh has invalid points: {mesh.GetPath()}."
        )
    points_attr.Set(Vt.Vec3fArray.FromNumpy(points))
    if mesh.GetExtentAttr().HasAuthoredValueOpinion():
        extent = np.stack((points.min(axis=0), points.max(axis=0))).astype(np.float32)
        mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(extent))
    if normals_attr.HasAuthoredValueOpinion():
        normals = np.asarray(normals_attr.Get(), dtype=float) @ np.linalg.inv(linear)
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(length, np.finfo(float).tiny)
        normals_attr.Set(
            Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals, dtype=np.float32))
        )


def _scale_joint(
    joint: UsdPhysics.Joint,
    root_body_path: Sdf.Path,
    initial_poses: dict[Sdf.Path, np.ndarray],
    fitted_poses: dict[Sdf.Path, np.ndarray],
    scale: np.ndarray,
) -> None:
    from pxr import Gf, UsdPhysics

    if joint.GetJointEnabledAttr().Get() is False:
        return
    prim = joint.GetPrim()
    body0 = joint.GetBody0Rel().GetTargets()
    body1 = joint.GetBody1Rel().GetTargets()
    if not body0 and prim.IsA(UsdPhysics.FixedJoint):
        # A world-to-root fixed joint keeps the original root frame.
        if body1 != [root_body_path]:
            raise ValueError("Scaled articulation cannot fix a non-root body to world.")
        return
    parent, child = body0[0], body1[0]
    anchor_attrs = [
        (parent, joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
        (child, joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()),
    ]
    old_rots = [
        initial_poses[path][:3, :3] @ _rotation(rot_attr.Get())
        for path, _, rot_attr in anchor_attrs
    ]
    if prim.IsA(UsdPhysics.RevoluteJoint):
        movable = UsdPhysics.RevoluteJoint(prim)
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        movable = UsdPhysics.PrismaticJoint(prim)
    elif prim.IsA(UsdPhysics.FixedJoint):
        movable = None
    else:
        raise ValueError(f"Unsupported scaled articulation joint: {prim.GetPath()}.")
    realignment = np.eye(3)
    if movable is not None:
        axis = str(movable.GetAxisAttr().Get())
        old_axis = old_rots[0][:, "XYZ".index(axis)]
        new_axis = scale @ old_axis
        factor = np.linalg.norm(new_axis)
        realignment = Rotation.align_vectors([new_axis / factor], [old_axis])[
            0
        ].as_matrix()
        if prim.IsA(UsdPhysics.PrismaticJoint):
            if any("mimic" in schema.lower() for schema in prim.GetAppliedSchemas()):
                raise ValueError(
                    f"Unsupported scaled prismatic mimic: {prim.GetPath()}."
                )
            _scale_prismatic_joint(movable, float(factor))
    for (path, pos_attr, rot_attr), old_rotation in zip(
        anchor_attrs, old_rots, strict=True
    ):
        old_pose = initial_poses[path]
        new_pose = fitted_poses[path]
        old_anchor = old_pose[:3, :3] @ np.asarray(pos_attr.Get()) + old_pose[:3, 3]
        anchor = new_pose[:3, :3].T @ (scale @ old_anchor - new_pose[:3, 3])
        orientation = new_pose[:3, :3].T @ realignment @ old_rotation
        pos_attr.Set(Gf.Vec3f(*anchor))
        rot_attr.Set(_quat(orientation))


def _scale_prismatic_joint(joint: UsdPhysics.PrismaticJoint, factor: float) -> None:
    from pxr import UsdPhysics

    prim = joint.GetPrim()
    drive = UsdPhysics.DriveAPI.Get(prim, "linear")
    for attr in (
        joint.GetLowerLimitAttr(),
        joint.GetUpperLimitAttr(),
        prim.GetAttribute("gen_sim:closedPosition"),
        drive.GetTargetPositionAttr() if drive else None,
        drive.GetTargetVelocityAttr() if drive else None,
    ):
        if attr and attr.HasAuthoredValueOpinion():
            value = attr.Get()
            if value is not None:
                attr.Set(value * factor)


def _scale_mass_properties(prim: Usd.Prim, scale: np.ndarray) -> None:
    from pxr import Gf, UsdPhysics

    if not prim.HasAPI(UsdPhysics.MassAPI):
        return
    mass = UsdPhysics.MassAPI(prim)
    center = mass.GetCenterOfMassAttr()
    if center.HasAuthoredValueOpinion() and np.isfinite(center.Get()).all():
        center.Set(Gf.Vec3f(*(scale @ np.asarray(center.Get()))))
    inertia = mass.GetDiagonalInertiaAttr()
    values = np.asarray(inertia.Get(), dtype=float)
    if not inertia.HasAuthoredValueOpinion() or not (values > 0.0).all():
        return
    axes = mass.GetPrincipalAxesAttr()
    rotation = _rotation(axes.Get()) if axes.HasAuthoredValueOpinion() else np.eye(3)
    covariance = rotation @ np.diag(0.5 * values.sum() - values) @ rotation.T
    scaled = scale @ covariance @ scale.T
    moments, basis = np.linalg.eigh(np.trace(scaled) * np.eye(3) - scaled)
    if np.linalg.det(basis) < 0.0:
        basis[:, 0] *= -1.0
    inertia.Set(Gf.Vec3f(*moments))
    axes.Set(_quat(basis))


def _rotation(quaternion: Gf.Quatf) -> np.ndarray:
    return Rotation.from_quat(
        [*quaternion.GetImaginary(), quaternion.GetReal()]
    ).as_matrix()


def _quat(matrix: np.ndarray) -> Gf.Quatf:
    from pxr import Gf

    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return Gf.Quatf(float(w), Gf.Vec3f(x, y, z))
