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

"""Translate the compiled reference hand into a native URDF source.

MuJoCo is used only as an asset compiler here. Simulation, replication and
state ownership remain with EmbodiChain SimulationManager and DexSim Spawn.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

__all__: list[str] = []


def _numbers(value: object) -> str:
    return " ".join(format(float(x), ".17g") for x in np.asarray(value).reshape(-1))


def _rpy(quat: np.ndarray) -> str:
    return _numbers(Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz"))


def _compile_hand(scene: Path, output: Path) -> tuple[Path, dict]:
    """Preserve compiled body/joint/mesh frames and inertia in the native asset."""
    output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(scene))
    root = ET.Element("robot", name="sharpa_reference_compiled")
    joints = {}
    for body_id in range(1, model.nbody):
        body = model.body(body_id)
        if body.name == "object":
            continue
        link = ET.SubElement(root, "link", name=body.name)
        inertia = ET.SubElement(link, "inertial")
        ET.SubElement(inertia, "origin", xyz=_numbers(body.ipos), rpy=_rpy(body.iquat))
        ET.SubElement(inertia, "mass", value=str(float(body.mass[0])))
        diag = body.inertia
        ET.SubElement(
            inertia,
            "inertia",
            ixx=str(diag[0]),
            iyy=str(diag[1]),
            izz=str(diag[2]),
            ixy="0",
            ixz="0",
            iyz="0",
        )
        for gid in range(int(body.geomadr[0]), int(body.geomadr[0] + body.geomnum[0])):
            geom = model.geom(gid)
            if geom.type[0] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError(f"Unexpected hand geometry {geom.name}")
            mid = int(geom.dataid[0])
            mesh = model.mesh(mid)
            path = output / f"mesh_{mid}.obj"
            if not path.exists():
                start, count = int(mesh.vertadr[0]), int(mesh.vertnum[0])
                face_start, face_count = int(mesh.faceadr[0]), int(mesh.facenum[0])
                vertices = model.mesh_vert[start : start + count]
                faces = model.mesh_face[face_start : face_start + face_count]
                path.write_text(
                    "".join("v " + _numbers(v) + "\n" for v in vertices)
                    + "".join(
                        "f " + " ".join(str(int(i) + 1) for i in f) + "\n"
                        for f in faces
                    )
                )
            kind = "collision" if (geom.contype[0] or geom.conaffinity[0]) else "visual"
            element = ET.SubElement(link, kind, name=geom.name)
            ET.SubElement(
                element, "origin", xyz=_numbers(geom.pos), rpy=_rpy(geom.quat)
            )
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "mesh", filename=str(path))
        parent = int(body.parentid[0])
        if parent == 0:
            continue
        if body.jntnum[0] != 1:
            raise ValueError("Expected one hinge per non-root hand body")
        joint = model.joint(int(body.jntadr[0]))
        if not np.allclose(joint.pos, 0):
            raise ValueError("Nonzero hinge anchor needs a separate URDF frame")
        element = ET.SubElement(root, "joint", name=joint.name, type="revolute")
        ET.SubElement(element, "parent", link=model.body(parent).name)
        ET.SubElement(element, "child", link=body.name)
        ET.SubElement(element, "origin", xyz=_numbers(body.pos), rpy=_rpy(body.quat))
        ET.SubElement(element, "axis", xyz=_numbers(joint.axis))
        actuator = np.flatnonzero(model.actuator_trnid[:, 0] == joint.id)
        if len(actuator) != 1:
            raise ValueError("Expected one position actuator per hand joint")
        aid = int(actuator[0])
        did = int(joint.dofadr[0])
        ET.SubElement(
            element,
            "limit",
            lower=str(joint.range[0]),
            upper=str(joint.range[1]),
            effort="1000",
            velocity="1000",
        )
        ET.SubElement(
            element,
            "dynamics",
            damping=str(model.dof_damping[did]),
            friction=str(model.dof_frictionloss[did]),
        )
        joints[joint.name] = {
            "kp": float(model.actuator_gainprm[aid, 0]),
            "kd": float(-model.actuator_biasprm[aid, 2]),
            "passive_damping": float(model.dof_damping[did]),
            "armature": float(model.dof_armature[did]),
            "friction": float(model.dof_frictionloss[did]),
            "ctrl_range": model.actuator_ctrlrange[aid].tolist(),
        }
    path = output / "hand.urdf"
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    identity = {
        "scene": str(scene),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "compiler_version": mujoco.__version__,
        "joints": joints,
        "root_position": model.body(1).pos.tolist(),
        "root_quat_wxyz": model.body(1).quat.tolist(),
        "body_count": model.nbody - 2,
        "reference_self_collision": True,
    }
    (output / "source.json").write_text(json.dumps(identity, indent=2) + "\n")
    return path, identity


def _compile_usd_hand(scene: Path, output: Path) -> tuple[Path, dict]:
    """Export geometry, inertia, joints and collision exclusions to OpenUSD."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    _, identity = _compile_hand(scene, output)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    path = output / "hand.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Hand")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().CreateAttribute(
        "newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool
    ).Set(True)
    body_paths = {}

    def pose(xform, pos, quat):
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        xform.AddOrientOp().Set(
            Gf.Quatf(float(quat[0]), Gf.Vec3f(*map(float, quat[1:])))
        )

    def qf(quat):
        return Gf.Quatf(float(quat[0]), Gf.Vec3f(*map(float, quat[1:])))

    # All body transforms are world poses, connected by explicit joints.
    for bid in range(1, model.nbody):
        body = model.body(bid)
        if body.name == "object":
            continue
        address = f"/Hand/{body.name}"
        body_paths[bid] = address
        link = UsdGeom.Xform.Define(stage, address)
        pose(link, data.xpos[bid], data.xquat[bid])
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(link.GetPrim())
        mass.CreateMassAttr(float(body.mass[0]))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, body.ipos)))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, body.inertia)))
        mass.CreatePrincipalAxesAttr(qf(body.iquat))
        for gid in range(int(body.geomadr[0]), int(body.geomadr[0] + body.geomnum[0])):
            geom = model.geom(gid)
            mid = int(geom.dataid[0])
            mesh = model.mesh(mid)
            vertices = model.mesh_vert[
                int(mesh.vertadr[0]) : int(mesh.vertadr[0] + mesh.vertnum[0])
            ]
            faces = model.mesh_face[
                int(mesh.faceadr[0]) : int(mesh.faceadr[0] + mesh.facenum[0])
            ]
            shape = UsdGeom.Mesh.Define(stage, address + "/" + geom.name)
            shape.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in vertices])
            shape.CreateFaceVertexCountsAttr([3] * len(faces))
            shape.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
            shape.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            shape.CreateDisplayColorAttr([Gf.Vec3f(*map(float, geom.rgba[:3]))])
            pose(shape, geom.pos, geom.quat)
            if geom.contype[0] or geom.conaffinity[0]:
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                material = UsdShade.Material.Define(
                    stage, f"/Hand/Materials/contact_{gid}"
                )
                surface = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                surface.CreateDynamicFrictionAttr(float(geom.friction[0]))
                surface.CreateStaticFrictionAttr(float(geom.friction[0]))
                UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
                    material, materialPurpose="physics"
                )
                UsdPhysics.MeshCollisionAPI.Apply(
                    shape.GetPrim()
                ).CreateApproximationAttr("convexHull")
                for name, value in (
                    ("condim", int(geom.condim[0])),
                    ("margin", float(geom.margin[0])),
                    ("gap", float(geom.gap[0])),
                ):
                    shape.GetPrim().CreateAttribute(
                        "mjc:" + name,
                        (
                            Sdf.ValueTypeNames.Int
                            if name == "condim"
                            else Sdf.ValueTypeNames.Float
                        ),
                    ).Set(value)
        parent = int(body.parentid[0])
        if parent == 0:
            joint = UsdPhysics.FixedJoint.Define(stage, "/Hand/world_joint")
            joint.CreateBody1Rel().SetTargets([address])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, data.xpos[bid])))
            joint.CreateLocalRot0Attr(qf(data.xquat[bid]))
        else:
            source = model.joint(int(body.jntadr[0]))
            if not np.allclose(source.axis, [0, 0, 1]):
                raise ValueError("Expected Z-axis hinges")
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/Hand/" + source.name)
            joint.CreateBody0Rel().SetTargets([body_paths[parent]])
            joint.CreateBody1Rel().SetTargets([address])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, body.pos)))
            joint.CreateLocalRot0Attr(qf(body.quat))
            joint.CreateAxisAttr("Z")
            joint.CreateLowerLimitAttr(float(np.rad2deg(source.range[0])))
            joint.CreateUpperLimitAttr(float(np.rad2deg(source.range[1])))
    for signature in model.exclude_signature:
        a, b = int(signature) >> 16, int(signature) & 65535
        if a in body_paths and b in body_paths:
            pairs = UsdPhysics.FilteredPairsAPI.Apply(
                stage.GetPrimAtPath(body_paths[a])
            )
            pairs.CreateFilteredPairsRel().AddTarget(body_paths[b])
    stage.GetRootLayer().Save()
    identity["native_source"] = str(path)
    identity["native_source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity["filtered_pairs"] = model.nexclude
    (output / "source.json").write_text(json.dumps(identity, indent=2) + "\n")
    return path, identity
