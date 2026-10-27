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
"""USD authoring for portable assemblies; no collision re-decomposition."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import AssemblyAsset

__all__: list[str] = []


def write_assembly_usd(asset: AssemblyAsset, path: str | Path) -> Path:
    """Author one rigid root per part and one convex collider per source geom."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
    from scipy.spatial.transform import Rotation

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(target))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Assembly")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("assembly:contentHash", asset.content_hash)
    root.GetPrim().SetCustomDataByKey(
        "assembly:generatorRevision", asset.generator_revision
    )

    def pose(xform: UsdGeom.Xformable, pos: list[float], quat: list[float]) -> None:
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
        xform.AddOrientOp().Set(Gf.Quatf(quat[3], Gf.Vec3f(*quat[:3])))

    used = set()
    for body in asset.bodies:
        name = body.name
        if not Sdf.Path.IsValidIdentifier(name) or name in used:
            raise ValueError(
                f"Assembly body name is not a unique USD identifier: {name!r}."
            )
        used.add(name)
        container_path = f"/Assembly/{name}"
        body_path = container_path
        if body.joint.kind == "hinge":
            container = UsdGeom.Xform.Define(stage, container_path)
            UsdPhysics.ArticulationRootAPI.Apply(container.GetPrim())
            body_path = f"{container_path}/{name}_link"
        body_xform = UsdGeom.Xform.Define(stage, body_path)
        pose(body_xform, body.position, body.quaternion)
        body_prim = body_xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim).CreateRigidBodyEnabledAttr(
            body.body_type == "dynamic"
        )
        if body.body_type == "dynamic":
            mass = UsdPhysics.MassAPI.Apply(body_prim)
            mass.CreateMassAttr(body.mass)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*body.com))
            moments, axes = np.linalg.eigh(np.asarray(body.inertia))
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            q = Rotation.from_matrix(axes).as_quat()
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments.tolist()))
            mass.CreatePrincipalAxesAttr(
                Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3].tolist()))
            )
        for index, geom in enumerate(body.geometries):
            geom_path = f"{body_path}/geom_{index}"
            if geom.kind == "mesh":
                from ._surface import _mesh_surface_arrays

                gprim = UsdGeom.Mesh.Define(stage, geom_path)
                gprim.CreatePointsAttr([Gf.Vec3f(*v) for v in geom.vertices])
                gprim.CreateFaceVertexCountsAttr([3] * len(geom.triangles))
                gprim.CreateFaceVertexIndicesAttr(
                    np.asarray(geom.triangles, dtype=int).ravel().tolist()
                )
                gprim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                # Mesh vertices can be shared across sharp convex edges.
                # Keep their source topology while authoring each corner's
                # face normal in geometry-local coordinates, as for OBJ.
                _, _, normals, normal_indices = _mesh_surface_arrays(geom)
                gprim.CreateNormalsAttr(
                    [
                        Gf.Vec3f(*normal.tolist())
                        for normal in normals[normal_indices].reshape(-1, 3)
                    ]
                )
                gprim.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            elif geom.kind == "cylinder":
                gprim = UsdGeom.Cylinder.Define(stage, geom_path)
                gprim.CreateRadiusAttr(geom.size[0])
                gprim.CreateHeightAttr(2 * geom.size[1])
                gprim.CreateAxisAttr(UsdGeom.Tokens.z)
            elif geom.kind == "sphere":
                gprim = UsdGeom.Sphere.Define(stage, geom_path)
                gprim.CreateRadiusAttr(geom.size[0])
            elif geom.kind == "box":
                gprim = UsdGeom.Cube.Define(stage, geom_path)
                gprim.CreateSizeAttr(2.0)
            else:
                raise ValueError(f"Cannot author USD assembly geom {geom.kind!r}.")
            pose(gprim, geom.position, geom.quaternion)
            if geom.kind == "box":
                gprim.AddScaleOp().Set(Gf.Vec3f(*geom.size[:3]))
            gprim.CreateDisplayColorAttr([Gf.Vec3f(*geom.rgba[:3])])
            gprim.CreateDisplayOpacityAttr([geom.rgba[3]])
            if not geom.visible:
                gprim.CreatePurposeAttr(UsdGeom.Tokens.guide)
            if geom.collision:
                UsdPhysics.CollisionAPI.Apply(
                    gprim.GetPrim()
                ).CreateCollisionEnabledAttr(True)
                if geom.kind == "mesh":
                    UsdPhysics.MeshCollisionAPI.Apply(
                        gprim.GetPrim()
                    ).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
                material_path = f"{body_path}/contact_{index}"
                material_prim = stage.DefinePrim(material_path, "Material")
                material = UsdPhysics.MaterialAPI.Apply(material_prim)
                material.CreateDynamicFrictionAttr(geom.friction[0])
                material.CreateStaticFrictionAttr(geom.friction[0])
                from pxr import UsdShade

                UsdShade.MaterialBindingAPI.Apply(gprim.GetPrim()).Bind(
                    UsdShade.Material(material_prim), materialPurpose="physics"
                )
                # Preserve exact source values even where a backend lacks an
                # equivalent schema. The precision harness reports that gap.
                gprim.GetPrim().SetCustomDataByKey(
                    "assembly:solref", Vt.DoubleArray(geom.solref)
                )
                gprim.GetPrim().SetCustomDataByKey(
                    "assembly:solimp", Vt.DoubleArray(geom.solimp)
                )
                gprim.GetPrim().SetCustomDataByKey(
                    "assembly:friction", Vt.DoubleArray(geom.friction)
                )
                gprim.GetPrim().SetCustomDataByKey("assembly:condim", geom.condim)
                gprim.GetPrim().SetCustomDataByKey("assembly:priority", geom.priority)
                gprim.GetPrim().SetCustomDataByKey("assembly:solmix", geom.solmix)
        if body.joint.kind == "hinge":
            # Preserve the world attachment. Native USD parsers recognize the
            # revolute joint, not an animation of the target body's transform.
            anchor_path = f"{container_path}/{name}_anchor"
            anchor_xform = UsdGeom.Xform.Define(stage, anchor_path)
            pose(anchor_xform, body.position, body.quaternion)
            UsdPhysics.RigidBodyAPI.Apply(anchor_xform.GetPrim())
            fixed = UsdPhysics.FixedJoint.Define(
                stage, f"{container_path}/world_anchor"
            )
            fixed.CreateBody1Rel().SetTargets([Sdf.Path(anchor_path)])
            fixed.CreateLocalPos0Attr(Gf.Vec3f(*body.position))
            fixed.CreateLocalRot0Attr(
                Gf.Quatf(body.quaternion[3], Gf.Vec3f(*body.quaternion[:3]))
            )
            joint = UsdPhysics.RevoluteJoint.Define(
                stage, f"{container_path}/{name}_hinge"
            )
            joint.CreateBody0Rel().SetTargets([Sdf.Path(anchor_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
            axis = np.asarray(body.joint.axis)
            if not np.allclose(axis, [0, 0, 1]):
                raise ValueError(
                    "USD assembly hinge export currently requires the mjfit +Z axis."
                )
            joint.CreateAxisAttr(UsdPhysics.Tokens.z)
            joint.CreateLocalPos0Attr(Gf.Vec3f(*body.joint.position))
            joint.CreateLocalPos1Attr(Gf.Vec3f(*body.joint.position))
            joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
            joint.GetPrim().SetCustomDataByKey("assembly:armature", body.joint.armature)
            joint.GetPrim().SetCustomDataByKey("assembly:damping", body.joint.damping)
            joint.GetPrim().SetCustomDataByKey(
                "assembly:motorGear", body.joint.motor_gear
            )
    stage.GetRootLayer().Save()
    return target
