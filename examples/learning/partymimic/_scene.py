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

"""Prepare a portable stage from the PartyMimic RP1 and two native welds."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from embodichain.extensions.models.policy.partymimic import PartyMimicPolicy

from ._reference import CRATE_OFFSET, PHYSICS_DT, Reference
from ._rp_hand import add_rp_hands

__all__: list[str] = []

_PAYLOAD = "carry_crate"
# Leadshine DH116S envelope, in metres. The hand geometry is render-only.
_HAND_SIZE = (0.14754, 0.07353, 0.02635)
_BLUE_LINKS = ("hip_yaw_material", "shoulder_roll_material", "elbow_material")


class WristCrateClearance:
    """Conservative separation of the wrists and RP Hands from the crate."""

    def __init__(self, model: mujoco.MjModel, link_names: Sequence[str]) -> None:
        links = [
            f"{side}_wrist_{joint}_link"
            for side in ("left", "right")
            for joint in ("roll", "yaw", "pitch")
        ]
        self.link_indices = [link_names.index(name) for name in links]
        self._model = model
        self._data = mujoco.MjData(model)
        self._box = model.geom("crate_box").id
        self._data.geom_xmat[self._box] = np.eye(3).ravel()
        self._half_size = model.geom_size[self._box]
        self._parts: list[tuple[int, int, np.ndarray, float, bool]] = []
        for slot, name in enumerate(links):
            body = model.body(name).id
            for geom in np.flatnonzero(model.geom_bodyid == body):
                kind = model.geom_type[geom]
                if kind == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = model.geom_dataid[geom]
                    start = model.mesh_vertadr[mesh]
                    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
                    points = vertices[ConvexHull(vertices).vertices]
                    radius = 0.0
                elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE and model.geom_contype[geom]:
                    radius, half_length = model.geom_size[geom, :2]
                    points = np.array([[0, 0, -half_length], [0, 0, half_length]])
                else:
                    continue
                rotation = Rotation.from_quat(model.geom_quat[geom, [1, 2, 3, 0]])
                is_hand = model.geom(int(geom)).name.startswith(("lh_", "rh_"))
                self._parts.append(
                    (
                        slot,
                        int(geom),
                        rotation.apply(points) + model.geom_pos[geom],
                        radius,
                        is_hand,
                    )
                )
        if not self._parts:
            raise ValueError(
                "The RP1 scene needs native wrist geometry for the clearance check."
            )

    def measure(
        self, link_poses: np.ndarray, crate_pose: np.ndarray
    ) -> tuple[float, float | None]:
        """Return separate minimum gaps for wrist and optional hand geometry."""
        to_crate = Rotation.from_quat(crate_pose[3:]).inv()
        rotations = to_crate * Rotation.from_quat(link_poses[:, 3:])
        positions = to_crate.apply(link_poses[:, :3] - crate_pose[:3])
        wrist_gaps = []
        hand_gaps = []
        for slot, geom, points, radius, is_hand in self._parts:
            vertices = rotations[slot].apply(points) + positions[slot]
            axis_gap = np.maximum(vertices.min(axis=0), -vertices.max(axis=0))
            gap = float((axis_gap - self._half_size - radius).max())
            if gap < 0.0:
                # A tilted part can clear the box on an axis absent from its faces.
                local_rotation = Rotation.from_quat(
                    self._model.geom_quat[geom, [1, 2, 3, 0]]
                )
                self._data.geom_xpos[geom] = (
                    rotations[slot].apply(self._model.geom_pos[geom]) + positions[slot]
                )
                self._data.geom_xmat[geom] = (
                    (rotations[slot] * local_rotation).as_matrix().ravel()
                )
                gap = mujoco.mj_geomDistance(
                    self._model, self._data, geom, self._box, 0.1, None
                )
            gaps = hand_gaps if is_hand else wrist_gaps
            gaps.append(gap)
        return min(wrist_gaps), min(hand_gaps, default=None)


def _numbers(values: np.ndarray | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _armature(name: str) -> float:
    # PartyMimic's rp1_constants.py specifies these four actuator classes.
    if any(token in name for token in ("hip_pitch", "hip_roll", "knee", "waist")):
        return 0.0290856
    if any(token in name for token in ("hip_yaw", "shoulder_pitch")):
        return 0.00884
    if "ankle" in name:
        return 0.006266268
    return 0.00313313


def _hand_visual(
    wrist: ET.Element,
    pose: np.ndarray,
    side: int,
    crate_y: float,
    crate_half_width: float,
) -> None:
    length, width, thickness = _HAND_SIZE
    palm_length = 0.58 * length
    radius = width / 11
    wrist_rotation = Rotation.from_quat(pose[3:])
    inverse = wrist_rotation.inv()
    wrist_point = pose[:3]
    wrist_name = wrist.attrib["name"]
    capsule = wrist.find(f"geom[@name='{wrist_name}_collision']")
    flange = np.fromstring(capsule.attrib["fromto"], sep=" ", count=3)
    flange_world = wrist_point + wrist_rotation.apply(flange)
    contact_y = crate_y + side * (crate_half_width + radius + 0.002)
    lateral_reach = (contact_y - flange_world[1]) / (length - radius)
    hand_rotation = Rotation.from_euler(
        "z", np.arcsin(np.clip(lateral_reach, -0.7, 0.7))
    )
    # Let the palm heel seat on the end of the native wrist housing.
    start = flange_world - hand_rotation.apply([0.006, 0, 0])

    def local(x: float, z: float = 0.0) -> np.ndarray:
        return inverse.apply(start + hand_rotation.apply([x, 0, z]) - wrist_point)

    visual = {"class": "visual", "rgba": "0.95 0.95 0.95 1"}
    ET.SubElement(
        wrist,
        "geom",
        **visual,
        type="box",
        pos=_numbers(local(palm_length / 2)),
        size=_numbers((palm_length / 2, thickness / 2, width / 2)),
        quat=_numbers((inverse * hand_rotation).as_quat()[[3, 0, 1, 2]]),
    )
    for index in range(4):
        z = (index - 1.5) * width / 4
        end = length - radius - abs(index - 1.5) * 0.004
        ET.SubElement(
            wrist,
            "geom",
            **visual,
            type="capsule",
            size=str(radius),
            fromto=_numbers(np.r_[local(palm_length - radius, z), local(end, z)]),
        )


def write_mjcf(
    source: Path,
    destination: Path,
    scenario: str,
    reference: Reference,
    policy: PartyMimicPolicy,
    *,
    payload_kg: float,
    hand_model: str = "proxy",
) -> mujoco.MjModel:
    """Write the calibrated diagnostic MJCF also used as USD converter input."""
    root = ET.parse(source).getroot()
    root.set("model", "RP1Carry")
    root.find("compiler").set("meshdir", str(source.resolve().parent))
    root.find("option").attrib.update(
        timestep=str(PHYSICS_DT),
        integrator="implicitfast",
        iterations="100",
        cone="pyramidal",
    )
    root.remove(root.find("sensor"))
    actuator = root.find("actuator")
    actuator.clear()
    for name, kp, kd, scale in zip(
        policy.joint_names,
        policy.joint_stiffness,
        policy.joint_damping,
        policy.action_scale,
        strict=True,
    ):
        limit = round(float(4 * kp * scale), 5)
        joint = root.find(f".//worldbody//joint[@name='{name}']")
        joint.set("armature", str(_armature(name)))
        joint.set("frictionloss", "0.1")
        ET.SubElement(
            actuator,
            "position",
            name=name,
            joint=name,
            kp=str(kp),
            kv=str(kd),
            forcerange=_numbers((-limit, limit)),
        )

    world = root.find("worldbody")
    for material in root.findall("asset/material"):
        if material.get("name", "").endswith(_BLUE_LINKS):
            material.set("rgba", "0.05 0.25 0.59 1")
    wrists = reference.poses[0, 2:4]
    crate_position = wrists[:, :3].mean(axis=0) + CRATE_OFFSET
    half_size = np.asarray(reference.payload_size) / 2
    if hand_model == "rp":
        add_rp_hands(root)
    else:
        for side, name, pose in zip(
            (1, -1), policy.keypoint_names[2:4], wrists, strict=True
        ):
            _hand_visual(
                world.find(f".//body[@name='{name}']"),
                pose,
                side,
                crate_position[1],
                half_size[1],
            )
    # Assign stable geometry identifiers for preserved self-collision exclusions.
    for index, geom in enumerate(world.iter("geom")):
        geom.attrib.setdefault("name", f"rp1_geom_{index}")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="box",
        size="6 3 0.05",
        pos="2 0 -0.05",
        rgba="0.7 0.72 0.74 1",
        friction="0.9 0.01 0.001",
    )
    if scenario == "step-up":
        platform = reference.platform
        if platform is None:
            raise ValueError("The step-up reference requires a platform.")
        depth, width, height = platform.size
        ET.SubElement(
            world,
            "geom",
            name="platform",
            type="box",
            size=_numbers((depth / 2, width / 2, height / 2)),
            pos=_numbers((platform.front + depth / 2, 0.0, height / 2)),
            rgba="0.49 0.52 0.54 1",
            friction="1.0 0.01 0.001",
        )
    if payload_kg > 0:
        crate = ET.SubElement(
            world, "body", name=_PAYLOAD, pos=_numbers(crate_position)
        )
        ET.SubElement(crate, "freejoint", name="crate_free")
        half_size_sq = np.square(half_size)
        inertia = payload_kg * (half_size_sq.sum() - half_size_sq) / 3
        ET.SubElement(
            crate,
            "inertial",
            pos="0 0 0",
            mass=str(payload_kg),
            diaginertia=_numbers(inertia),
        )
        ET.SubElement(
            crate,
            "geom",
            name="crate_box",
            type="box",
            size=_numbers(half_size),
            rgba="0.50 0.34 0.18 1",
            friction="0.8 0.01 0.001",
        )
        equality = ET.SubElement(root, "equality")
        for side, name, pose in zip(
            ("left", "right"), policy.keypoint_names[2:4], wrists, strict=True
        ):
            inverse = Rotation.from_quat(pose[3:]).inv()
            relative = np.r_[
                inverse.apply(crate_position - pose[:3]),
                inverse.as_quat()[[3, 0, 1, 2]],
            ]
            ET.SubElement(
                equality,
                "weld",
                name=f"carry_{side}",
                body1=name,
                body2=_PAYLOAD,
                relpose=_numbers(relative),
                torquescale="0.25",
                solref="0.02 1",
            )
            ET.SubElement(root.find("contact"), "exclude", body1=name, body2=_PAYLOAD)

    ET.ElementTree(root).write(destination, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(destination))


def convert_scene(
    source: Path,
    model: mujoco.MjModel,
    destination: Path,
    converter_python: Path,
    policy: PartyMimicPolicy,
    reference: Reference,
) -> None:
    """Translate RP1 geometry and retain the native MuJoCo-specific controls."""
    with TemporaryDirectory(dir=destination.parent) as directory:
        bundle = Path(directory) / "stage"
        subprocess.run(
            [
                str(converter_python),
                "-I",
                "-m",
                "mujoco_usd_converter",
                str(source),
                str(bundle),
                "--no-layer-structure",
                "--no-openpbr",
            ],
            check=True,
        )
        stage = Usd.Stage.Open(str(bundle / "RP1Carry.usdc"))
        _retain_physics(stage, source, model, policy, reference)
        _add_checkerboard(stage, destination.with_name("carry_checker.png"))
        if not stage.GetRootLayer().Export(str(destination)):
            raise RuntimeError(f"Could not write RP1 scene to {destination}.")


def _add_checkerboard(stage: Usd.Stage, texture_path: Path) -> None:
    """Lay a repeating visual checker over the single physical floor."""
    light = (199, 206, 212)
    dark = (174, 182, 189)
    tile = np.empty((128, 128, 3), dtype=np.uint8)
    tile[:] = light
    tile[:64, 64:] = tile[64:, :64] = dark
    Image.fromarray(tile).save(texture_path)

    root = stage.GetDefaultPrim().GetPath()
    mesh = UsdGeom.Mesh.Define(stage, root.AppendChild("checkerboard"))
    edge = 96.0
    mesh.CreatePointsAttr(
        [
            (-edge, -edge, 0.001),
            (edge, -edge, 0.001),
            (edge, edge, 0.001),
            (-edge, edge, 0.001),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    repeats = edge / 0.70
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    uv.Set(
        [
            (-repeats, -repeats),
            (repeats, -repeats),
            (repeats, repeats),
            (-repeats, repeats),
        ]
    )

    material_path = root.AppendChild("CheckerMaterial")
    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, material_path.AppendChild("surface"))
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.93)
    reader = UsdShade.Shader.Define(stage, material_path.AppendChild("uv"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, material_path.AppendChild("albedo"))
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(f"./{texture_path.name}")
    )
    for axis in ("wrapS", "wrapT"):
        texture.CreateInput(axis, Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _retain_physics(
    stage: Usd.Stage,
    source: Path,
    model: mujoco.MjModel,
    policy: PartyMimicPolicy,
    reference: Reference,
) -> None:
    root = stage.GetDefaultPrim()
    equalities = {}
    for index in range(model.neq):
        name = model.equality(index).name
        equalities[name] = {
            "name": name,
            "type": int(model.eq_type[index]),
            "obj1_name": model.body(int(model.eq_obj1id[index])).name,
            "obj2_name": model.body(int(model.eq_obj2id[index])).name,
            "enabled": bool(model.eq_active0[index]),
            "data": Vt.DoubleArray(model.eq_data[index].tolist()),
            "solref": Vt.DoubleArray(model.eq_solref[index].tolist()),
            "solimp": Vt.DoubleArray(model.eq_solimp[index].tolist()),
        }
    # USD generic FixedJoints are for the default backend. The Newton
    # adapter lowers these compiled MuJoCo rows into solver-native welds.
    if equalities:
        root.SetCustomDataByKey("mujoco:equalityConstraints", equalities)
    root.SetCustomDataByKey(
        "mujoco:solverOptions",
        {
            "iterations": int(model.opt.iterations),
            "integrator": int(model.opt.integrator),
        },
    )
    geometry = {}
    for index in range(model.ngeom):
        if model.geom_contype[index] or model.geom_conaffinity[index]:
            geometry.setdefault(model.body(model.geom_bodyid[index]).name, []).append(
                model.geom(index).name
            )
    pairs = {}
    for item in ET.parse(source).iter("exclude"):
        for name1 in geometry.get(item.get("body1"), ()):
            for name2 in geometry.get(item.get("body2"), ()):
                pairs[f"pair_{len(pairs)}"] = Vt.StringArray([name1, name2])
    if pairs:
        root.SetCustomDataByKey("mujoco:filteredGeomPairs", pairs)

    indices = {name: i for i, name in enumerate(policy.joint_names)}
    geom_ids = {model.geom(i).name: i for i in range(model.ngeom)}
    for prim in list(stage.Traverse()):
        name = prim.GetDisplayName() or prim.GetName()
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            relationship = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
            if name == _PAYLOAD:
                relationship.ClearTargets(True)
            else:
                for target in relationship.GetTargets():
                    if target.name == _PAYLOAD:
                        relationship.RemoveTarget(target)
        if name in equalities and prim.IsA(UsdPhysics.FixedJoint):
            stage.RemovePrim(prim.GetPath())
        elif name in indices and prim.IsA(UsdPhysics.RevoluteJoint):
            _author_drive(prim, name, indices[name], model, policy, reference)
        elif name in geom_ids and prim.HasAPI(UsdPhysics.CollisionAPI):
            i = geom_ids[name]
            for field, value in (
                ("newton:torsionalFriction", model.geom_friction[i, 1]),
                ("newton:rollingFriction", model.geom_friction[i, 2]),
            ):
                prim.CreateAttribute(field, Sdf.ValueTypeNames.Double).Set(float(value))
            for field, value in (
                ("mjc:contype", model.geom_contype[i]),
                ("mjc:conaffinity", model.geom_conaffinity[i]),
            ):
                prim.CreateAttribute(field, Sdf.ValueTypeNames.Int).Set(int(value))


def _author_drive(
    prim: Usd.Prim,
    name: str,
    index: int,
    model: mujoco.MjModel,
    policy: PartyMimicPolicy,
    reference: Reference,
) -> None:
    joint = model.joint(name).dofadr[0]
    force_range = model.actuator(name).forcerange
    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(np.deg2rad(policy.joint_stiffness[index])))
    drive.CreateDampingAttr(float(np.deg2rad(policy.joint_damping[index])))
    drive.CreateMaxForceAttr(float(force_range[1]))
    drive.CreateTargetPositionAttr(float(np.rad2deg(reference.initial_joints[index])))
    for name, value in (
        ("armature", model.dof_armature[joint]),
        ("damping", model.dof_damping[joint]),
        ("frictionloss", model.dof_frictionloss[joint]),
    ):
        prim.CreateAttribute(f"mjc:{name}", Sdf.ValueTypeNames.Double).Set(float(value))
    prim.CreateAttribute("mjc:forcerange", Sdf.ValueTypeNames.DoubleArray).Set(
        Vt.DoubleArray(force_range.tolist())
    )
    prim.CreateAttribute("mjc:forcelimited", Sdf.ValueTypeNames.Bool).Set(True)
