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

"""Export and measure the pinned official Cube task before porting its physics.

This reference tool requires the upstream OGBench and stable-worldmodel checkouts.
The generated URDF preserves the compiled robot's kinematics, geometry and inertia;
its sidecar retains the actuator/contact/closed-loop information URDF cannot carry.
A successful export is not a physics-alignment result.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from ._reference import LEWM_REV, SWM_REV

OGBENCH_REV = "6eda27402c69681cfe584cc9f3a3fe2a24ec9163"
CUBE_SWM_REV = "abdced49809d5eae38e24b27dc7b635c502c4812"
CUBE_WEIGHTS_REV = "b0747c5002e86d2ce8f3cd8178004b97524c587d"
CONFIG = Path(__file__).with_name("configs") / "ogbench_cube.json"


def _revision(path: Path, expected: str) -> None:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    if actual != expected:
        raise ValueError(f"reference revision mismatch: {path}: {actual} != {expected}")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, text=True
    ).strip():
        raise ValueError(f"reference checkout has modified tracked files: {path}")


def _save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _numbers(value: Any) -> str:
    return " ".join(format(float(x), ".17g") for x in np.asarray(value).ravel())


def _origin(parent: ET.Element, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    rpy = Rotation.from_quat(np.roll(quat_wxyz, -1)).as_euler("xyz")
    ET.SubElement(parent, "origin", xyz=_numbers(pos), rpy=_numbers(rpy))


def _xml_pose(element: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4)
    if element is not None:
        matrix[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        matrix[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return matrix


def _set_origin(element: ET.Element, matrix: np.ndarray) -> None:
    previous = element.find("origin")
    if previous is not None:
        element.remove(previous)
    _origin(
        element,
        matrix[:3, 3],
        np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1),
    )


def _collapse_robot(
    robot: ET.Element, source_bodies: list[dict[str, Any]]
) -> tuple[ET.Element, dict[str, Any]]:
    """Fold massless fixed frames while retaining physical fixed bodies.

    A zero-mass site must not become a moving epsilon-mass PhysX link. Preserve
    it as a rigid binding. Positive-mass fixed bodies remain independent so
    their incoming-joint wrenches can recover source body external forces.
    """
    links = {x.attrib["name"]: x for x in robot.findall("link")}
    joints = {x.find("child").attrib["link"]: x for x in robot.findall("joint")}
    root_name = "ur5e/base"
    physical = {root_name} | {
        name for name, joint in joints.items() if joint.attrib["type"] != "fixed"
    }
    physical.update(body["name"] for body in source_bodies if body["mass"] > 0)
    transforms = {"world": np.eye(4)}

    def world_pose(name: str) -> np.ndarray:
        if name not in transforms:
            joint = joints[name]
            transforms[name] = world_pose(
                joint.find("parent").attrib["link"]
            ) @ _xml_pose(joint.find("origin"))
        return transforms[name]

    def owner(name: str) -> str:
        while name not in physical:
            if name == "world":
                return root_name
            name = joints[name].find("parent").attrib["link"]
        return name

    result = ET.Element("robot", name=robot.attrib["name"])
    new_links = {
        name: ET.SubElement(result, "link", name=name)
        for name in links
        if name in physical
    }
    pieces = {name: [] for name in new_links}
    for name, link in links.items():
        owning = owner(name)
        relative = np.linalg.inv(world_pose(owning)) @ world_pose(name)
        inertial = link.find("inertial")
        if inertial is not None:
            mass = float(inertial.find("mass").attrib["value"])
            pose = relative @ _xml_pose(inertial.find("origin"))
            values = inertial.find("inertia").attrib
            matrix = np.array(
                [
                    [float(values["ixx"]), float(values["ixy"]), float(values["ixz"])],
                    [float(values["ixy"]), float(values["iyy"]), float(values["iyz"])],
                    [float(values["ixz"]), float(values["iyz"]), float(values["izz"])],
                ]
            )
            tensor = pose[:3, :3] @ matrix @ pose[:3, :3].T
            pieces[owning].append((mass, pose[:3, 3], tensor, name))
        for role in ("visual", "collision"):
            for geometry in link.findall(role):
                child = deepcopy(geometry)
                _set_origin(child, relative @ _xml_pose(child.find("origin")))
                new_links[owning].append(child)
    bodies = []
    for name, elements in pieces.items():
        mass = sum(m for m, _, _, _ in elements)
        if mass <= 0:
            raise ValueError(f"surviving physical link has no source mass: {name}")
        com = sum(m * p for m, p, _, _ in elements) / mass
        tensor = np.zeros((3, 3))
        for m, p, inertia, _ in elements:
            delta = p - com
            tensor += inertia + m * (
                np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta)
            )
        principal, axes = np.linalg.eigh(tensor)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        quaternion = np.roll(Rotation.from_matrix(axes).as_quat(), 1)
        inertial = ET.SubElement(new_links[name], "inertial")
        # Author the full tensor in link axes. This avoids relying on each
        # URDF importer's treatment of an additional inertial RPY rotation.
        _origin(inertial, com, np.array([1, 0, 0, 0]))
        ET.SubElement(inertial, "mass", value=_numbers([mass]))
        ET.SubElement(
            inertial,
            "inertia",
            ixx=_numbers([tensor[0, 0]]),
            iyy=_numbers([tensor[1, 1]]),
            izz=_numbers([tensor[2, 2]]),
            ixy=_numbers([tensor[0, 1]]),
            ixz=_numbers([tensor[0, 2]]),
            iyz=_numbers([tensor[1, 2]]),
        )
        bodies.append(
            {
                "name": name,
                "mass": mass,
                "ipos": com.tolist(),
                "iquat_wxyz": quaternion.tolist(),
                "diaginertia": principal.tolist(),
                "urdf_frame_in_body": [0, 0, 0],
                "inertia_matrix": tensor.tolist(),
                "source_links": [piece[3] for piece in elements],
                "parent_link": (
                    None
                    if name == root_name
                    else owner(joints[name].find("parent").attrib["link"])
                ),
            }
        )
        if name != root_name:
            joint = deepcopy(joints[name])
            parent = owner(joint.find("parent").attrib["link"])
            joint.find("parent").set("link", parent)
            _set_origin(joint, np.linalg.inv(world_pose(parent)) @ world_pose(name))
            result.append(joint)
    frame_bindings = []
    pivots = {body["name"]: body["urdf_frame_in_body"] for body in source_bodies}
    for name in links:
        if name == "world":
            continue
        owning = owner(name)
        frame = np.linalg.inv(world_pose(owning)) @ world_pose(name)
        if name in pivots:
            body_offset = np.eye(4)
            body_offset[:3, 3] = -np.asarray(pivots[name])
            frame = frame @ body_offset
        frame_bindings.append(
            {
                "name": name,
                "link": owning,
                "transform": frame.tolist(),
                "kind": "body" if name in pivots else "site",
            }
        )
    return result, {
        "bodies": bodies,
        "source_bodies": source_bodies,
        "source_frames": frame_bindings,
        "root_pose": world_pose(root_name).tolist(),
        "fixed_bodies_collapsed": True,
        "fixed_body_policy": "collapse_massless_frames_only",
        "preserved_fixed_bodies": [
            name
            for name in new_links
            if name != root_name and joints[name].attrib["type"] == "fixed"
        ],
        "source_total_mass": sum(body["mass"] for body in source_bodies),
        "total_mass": sum(body["mass"] for body in bodies),
    }


def _welded_body_exclusions(
    bodies: list[dict[str, Any]], *, filter_parent: bool
) -> list[list[str]]:
    """Preserve MuJoCo weld-group filtering after retaining fixed links.

    Parent filtering excludes neighboring non-world weld groups. World weld
    group zero remains eligible, as in MuJoCo's collision driver.
    """
    result = []
    for first, second in combinations(bodies, 2):
        a, b = first["source_weld_id"], second["source_weld_id"]
        same = a == b
        adjacent = (
            filter_parent
            and a != 0
            and b != 0
            and (
                a == second["source_parent_weld_id"]
                or b == first["source_parent_weld_id"]
            )
        )
        if same or adjacent:
            result.append([first["name"], second["name"]])
    return result


def _effective_rgba(env: Any) -> dict[int, np.ndarray]:
    """Read MuJoCo's resolved geom/material colors before renderer lighting."""
    import mujoco

    scene = mujoco.MjvScene(env.model, maxgeom=env.model.ngeom + 1)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    mujoco.mjv_updateScene(
        env.model,
        env.data,
        option,
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    return {
        geom.objid: geom.rgba.copy()
        for geom in scene.geoms[: scene.ngeom]
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
    }


def _export_render_assets(env: Any, output: Path) -> None:
    """Retain the compiled checker texture and source lighting parameters."""
    from PIL import Image

    model = env.model
    textures = output / "textures"
    textures.mkdir(exist_ok=True)
    index = model.texture("grid").id
    width, height = int(model.tex_width[index]), int(model.tex_height[index])
    channels = int(model.tex_nchannel[index])
    start = model.tex_adr[index]
    pixels = model.tex_data[start : start + width * height * channels].reshape(
        height, width, channels
    )
    Image.fromarray(pixels).save(textures / "grid.png")
    _save(
        output / "render-contract.json",
        {
            "grid_texture": "textures/grid.png",
            "grid_texture_sha256": hashlib.sha256(
                (textures / "grid.png").read_bytes()
            ).hexdigest(),
            "headlight": {
                key: np.asarray(getattr(model.vis.headlight, key)).tolist()
                for key in ("ambient", "diffuse", "specular", "active")
            },
            "lights": [
                {
                    "name": model.light(i).name,
                    **{
                        key: getattr(model, "light_" + key)[i].tolist()
                        for key in (
                            "pos",
                            "dir",
                            "ambient",
                            "diffuse",
                            "specular",
                            "type",
                            "castshadow",
                        )
                    },
                }
                for i in range(model.nlight)
            ],
        },
    )


def _export_robot(env: Any, root: Path) -> dict[str, Any]:
    import mujoco

    model = env.model
    effective_rgba = _effective_rgba(env)
    root.mkdir(parents=True, exist_ok=True)
    meshes = root / "meshes"
    meshes.mkdir(exist_ok=True)
    robot = ET.Element("robot", name="ogbench_ur5e_robotiq")
    ET.SubElement(robot, "link", name="world")
    ids = [i for i in range(1, model.nbody) if model.body(i).name.startswith("ur5e/")]
    pivots = {0: np.zeros(3)}
    joints = []
    body_records = []
    exported_meshes = set()
    effort_limits = {}
    for aid in range(model.nu):
        if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
            jid = int(model.actuator_trnid[aid, 0])
            effort_limits[jid] = float(np.max(np.abs(model.actuator_forcerange[aid])))
    for bid in ids:
        nj = model.body_jntnum[bid]
        if nj > 1:
            raise ValueError("Cube URDF export requires at most one joint per body")
        jid = int(model.body_jntadr[bid]) if nj else None
        pivot = model.jnt_pos[jid].copy() if nj else np.zeros(3)
        pivots[bid] = pivot
        name = model.body(bid).name
        parent_id = int(model.body_parentid[bid])
        parent_name = model.body(parent_id).name if parent_id in ids else "world"
        if parent_id not in pivots:
            raise ValueError(f"robot body outside exported tree: {name}")
        link = ET.SubElement(robot, "link", name=name)
        if model.body_mass[bid] > 0:
            inertia = ET.SubElement(link, "inertial")
            _origin(inertia, model.body_ipos[bid] - pivot, model.body_iquat[bid])
            ET.SubElement(inertia, "mass", value=_numbers([model.body_mass[bid]]))
            ix, iy, iz = model.body_inertia[bid]
            ET.SubElement(
                inertia,
                "inertia",
                ixx=_numbers([ix]),
                iyy=_numbers([iy]),
                izz=_numbers([iz]),
                ixy="0",
                ixz="0",
                iyz="0",
            )
        body_records.append(
            {
                "name": name,
                "mass": float(model.body_mass[bid]),
                "ipos": model.body_ipos[bid].tolist(),
                "iquat_wxyz": model.body_iquat[bid].tolist(),
                "diaginertia": model.body_inertia[bid].tolist(),
                "urdf_frame_in_body": pivot.tolist(),
            }
        )
        if nj:
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"unsupported robot joint: {model.joint(jid).name}")
            if model.qpos0[model.jnt_qposadr[jid]] != 0:
                raise ValueError(
                    "nonzero joint reference requires an explicit frame conversion"
                )
            jname = model.joint(jid).name
            joint = ET.SubElement(robot, "joint", name=jname, type="revolute")
            ET.SubElement(joint, "axis", xyz=_numbers(model.jnt_axis[jid]))
            dof = model.jnt_dofadr[jid]
            ET.SubElement(
                joint,
                "limit",
                lower=_numbers([model.jnt_range[jid, 0]]),
                upper=_numbers([model.jnt_range[jid, 1]]),
                effort=_numbers([effort_limits.get(jid, 0.0)]),
                velocity="1e30",
            )
            ET.SubElement(
                joint,
                "dynamics",
                damping=_numbers([model.dof_damping[dof]]),
                friction=_numbers([model.dof_frictionloss[dof]]),
            )
            joints.append(jname)
        else:
            joint = ET.SubElement(robot, "joint", name=name + "__fixed", type="fixed")
        ET.SubElement(joint, "parent", link=parent_name)
        ET.SubElement(joint, "child", link=name)
        rotation = Rotation.from_quat(np.roll(model.body_quat[bid], -1))
        _origin(
            joint,
            model.body_pos[bid] - pivots[parent_id] + rotation.apply(pivot),
            model.body_quat[bid],
        )
        for gid in range(
            model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]
        ):
            collision = bool(model.geom_contype[gid] or model.geom_conaffinity[gid])
            visual = model.geom_group[gid] in (0, 1, 2)
            for role in (["visual"] if visual else []) + (
                ["collision"] if collision else []
            ):
                geom = ET.SubElement(link, role, name=f"geom_{gid}")
                _origin(geom, model.geom_pos[gid] - pivot, model.geom_quat[gid])
                geometry = ET.SubElement(geom, "geometry")
                kind, size = model.geom_type[gid], model.geom_size[gid]
                if kind == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = int(model.geom_dataid[gid])
                    fname = f"mesh_{mid}.obj"
                    if mid not in exported_meshes:
                        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                        with (meshes / fname).open("w") as file:
                            for v in model.mesh_vert[va : va + vn]:
                                file.write("v " + _numbers(v) + "\n")
                            for f in model.mesh_face[fa : fa + fn]:
                                file.write(
                                    "f " + " ".join(str(int(i) + 1) for i in f) + "\n"
                                )
                        exported_meshes.add(mid)
                    ET.SubElement(geometry, "mesh", filename=f"meshes/{fname}")
                elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                    ET.SubElement(geometry, "box", size=_numbers(2 * size))
                elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                    ET.SubElement(geometry, "sphere", radius=_numbers(size[:1]))
                elif kind in (
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    mujoco.mjtGeom.mjGEOM_CYLINDER,
                ):
                    capsule = kind == mujoco.mjtGeom.mjGEOM_CAPSULE
                    # DexSim's URDF capsule length includes both hemispheres.
                    length = 2 * (size[1] + (size[0] if capsule else 0))
                    ET.SubElement(
                        geometry,
                        "capsule" if capsule else "cylinder",
                        radius=_numbers(size[:1]),
                        length=_numbers([length]),
                    )
                else:
                    raise ValueError(f"unsupported robot geometry: {kind}")
                if role == "visual":
                    rgba = effective_rgba[gid]
                    material = ET.SubElement(geom, "material", name=f"material_{gid}")
                    ET.SubElement(material, "color", rgba=_numbers(rgba))
    for sid in (env._attach_site_id, env._pinch_site_id):
        name, bid = model.site(sid).name, int(model.site_bodyid[sid])
        ET.SubElement(robot, "link", name=name)
        joint = ET.SubElement(robot, "joint", name=name + "__fixed", type="fixed")
        ET.SubElement(joint, "parent", link=model.body(bid).name)
        ET.SubElement(joint, "child", link=name)
        _origin(joint, model.site_pos[sid] - pivots[bid], model.site_quat[sid])
    robot, composite = _collapse_robot(robot, body_records)
    by_joint_name = {joint.attrib["name"]: joint for joint in robot.findall("joint")}
    mimics = []
    for index, kind in enumerate(model.eq_type):
        if kind != mujoco.mjtEq.mjEQ_JOINT:
            continue
        coefficients = model.eq_data[index, :5]
        if np.any(coefficients[2:] != 0):
            raise ValueError("the Cube export requires linear joint equalities")
        follower = model.joint(int(model.eq_obj1id[index])).name
        leader = model.joint(int(model.eq_obj2id[index])).name
        ET.SubElement(
            by_joint_name[follower],
            "mimic",
            joint=leader,
            multiplier=_numbers(coefficients[1:2]),
            offset=_numbers(coefficients[:1]),
        )
        mimics.append(
            {
                "follower": follower,
                "leader": leader,
                "solref": model.eq_solref[index].tolist(),
                "solimp": model.eq_solimp[index].tolist(),
            }
        )
    ET.indent(robot)
    path = root / "robot.urdf"
    ET.ElementTree(robot).write(path, encoding="unicode", xml_declaration=False)
    record = {
        **composite,
        "mimics": mimics,
        "joint_names": joints,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "physics_alignment_verified": False,
        "unbounded_joint_velocity_encoding": 1e30,
        "urdf_effort_limits": "direct_joint_actuators_only; tendon remains in sidecar",
        "required_sidecar_features": [
            "closed_loop_connect",
            "joint_equality",
            "split_tendon_actuator",
            "per_geom_contact",
            "joint_armature",
            "passive_joint_springs",
        ],
    }
    for body in record["bodies"]:
        bid = model.body(body["name"]).id
        weld = int(model.body_weldid[bid])
        body["source_body_id"] = int(bid)
        body["source_root_id"] = int(model.body_rootid[bid])
        body["source_weld_id"] = weld
        body["source_parent_weld_id"] = int(
            model.body_weldid[model.body_parentid[weld]]
        )
    record["implicit_collision_exclusions"] = _welded_body_exclusions(
        record["bodies"],
        filter_parent=not bool(
            model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
        ),
    )
    _save(root / "robot.json", record)
    return record


def _physical_contract(env: Any) -> dict[str, Any]:
    import mujoco

    m = env.model
    geoms = []
    for i in range(m.ngeom):
        geoms.append(
            {
                "name": m.geom(i).name,
                "body": m.body(m.geom_bodyid[i]).name,
                **{
                    key: getattr(m, "geom_" + key)[i].tolist()
                    for key in (
                        "type",
                        "size",
                        "pos",
                        "quat",
                        "friction",
                        "condim",
                        "solref",
                        "solimp",
                        "solmix",
                        "margin",
                        "gap",
                        "contype",
                        "conaffinity",
                        "priority",
                        "rgba",
                        "group",
                    )
                },
            }
        )
    return {
        "timestep": m.opt.timestep,
        "control_timestep": env.control_timestep(),
        "substeps": env._n_steps,
        "gravity": m.opt.gravity.tolist(),
        "integrator": mujoco.mjtIntegrator(int(m.opt.integrator)).name,
        "cone": mujoco.mjtCone(int(m.opt.cone)).name,
        "solver": mujoco.mjtSolver(int(m.opt.solver)).name,
        "iterations": m.opt.iterations,
        "tolerance": m.opt.tolerance,
        "impratio": m.opt.impratio,
        "enableflags": int(m.opt.enableflags),
        "disableflags": int(m.opt.disableflags),
        "sleep_enabled": bool(m.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP),
        "nq": m.nq,
        "nv": m.nv,
        "nu": m.nu,
        "workspace_bounds": env._workspace_bounds.tolist(),
        "action_range": env.action_high.tolist(),
        "home_qpos": env._home_qpos.tolist(),
        "pinch_to_attachment": env._T_pa.as_matrix().tolist(),
        "actuators": [
            {
                "name": m.actuator(i).name,
                **{
                    key: getattr(m, "actuator_" + key)[i].tolist()
                    for key in (
                        "gainprm",
                        "biasprm",
                        "gaintype",
                        "biastype",
                        "trntype",
                        "trnid",
                        "gear",
                        "ctrlrange",
                        "forcerange",
                        "ctrllimited",
                        "forcelimited",
                    )
                },
            }
            for i in range(m.nu)
        ],
        "dof_damping": m.dof_damping.tolist(),
        "dof_armature": m.dof_armature.tolist(),
        "dof_invweight0": m.dof_invweight0.tolist(),
        "dof_frictionloss": m.dof_frictionloss.tolist(),
        "dof_solref": m.dof_solref.tolist(),
        "dof_solimp": m.dof_solimp.tolist(),
        "jnt_stiffness": m.jnt_stiffness.tolist(),
        **{
            "jnt_" + key: getattr(m, "jnt_" + key).tolist()
            for key in (
                "type",
                "qposadr",
                "dofadr",
                "range",
                "limited",
                "margin",
                "solref",
                "solimp",
            )
        },
        "qpos_spring": m.qpos_spring.tolist(),
        "joint_names": [m.joint(i).name for i in range(m.njnt)],
        "body_names": [m.body(i).name for i in range(m.nbody)],
        "body_invweight0": m.body_invweight0.tolist(),
        "excluded_body_pairs": [
            [m.body(int(signature) >> 16).name, m.body(int(signature) & 0xFFFF).name]
            for signature in m.exclude_signature
        ],
        "equality": {
            key: getattr(m, "eq_" + key).tolist()
            for key in ("type", "obj1id", "obj2id", "data", "solref", "solimp")
        },
        "tendon": {
            key: getattr(m, "tendon_" + key).tolist()
            for key in ("adr", "num", "stiffness", "damping")
        },
        "wrap": {
            key: getattr(m, "wrap_" + key).tolist() for key in ("type", "objid", "prm")
        },
        "geoms": geoms,
        "camera": {
            "name": "front_pixels",
            "pos": m.camera("front_pixels").pos.tolist(),
            "quat_wxyz": m.camera("front_pixels").quat.tolist(),
            "fovy_degrees": float(m.camera("front_pixels").fovy[0]),
            "znear": float(m.vis.map.znear * m.stat.extent),
            "zfar": float(m.vis.map.zfar * m.stat.extent),
        },
        "success": {
            "metric": "cube_3d_position_l2",
            "threshold": 0.04,
            "inclusive": True,
            "timing": "post",
            "orientation_in_metric": False,
        },
    }


def _validate_kinematics(
    env: Any, output: Path, robot: dict[str, Any]
) -> dict[str, Any]:
    import mujoco
    import pytorch_kinematics as pk
    import torch

    model = env.model
    chain = pk.build_chain_from_urdf((output / "robot/robot.urdf").read_text()).to(
        dtype=torch.float64
    )
    names = chain.get_joint_parameter_names()
    rng = np.random.default_rng(42)
    qpos = np.empty((100, len(names)))
    for col, name in enumerate(names):
        qpos[:, col] = rng.uniform(*model.joint(name).range, size=100)
    transforms = chain.forward_kinematics(torch.from_numpy(qpos))
    bindings = robot["source_frames"]
    frame_names = [binding["name"] for binding in bindings]
    source = np.empty((len(qpos), len(frame_names), 4, 4))
    data = mujoco.MjData(model)
    for row, positions in enumerate(qpos):
        data.qpos[:] = env.data.qpos
        for name, value in zip(names, positions):
            data.joint(name).qpos[:] = value
        mujoco.mj_kinematics(model, data)
        for col, binding in enumerate(bindings):
            matrix = np.eye(4)
            if binding["kind"] == "body":
                bid = model.body(binding["name"]).id
                matrix[:3, :3] = data.xmat[bid].reshape(3, 3)
                matrix[:3, 3] = data.xpos[bid]
            else:
                sid = model.site(binding["name"]).id
                matrix[:3, :3] = data.site_xmat[sid].reshape(3, 3)
                matrix[:3, 3] = data.site_xpos[sid]
            source[row, col] = matrix
    actual = np.stack(
        [
            np.array(robot["root_pose"])
            @ transforms[binding["link"]].get_matrix().numpy()
            @ np.array(binding["transform"])
            for binding in bindings
        ],
        axis=1,
    )
    errors = np.abs(actual - source)
    # PK's URDF parser constructs fixed rotations in float32 before .to(float64).
    # Report this import precision rather than treating the later cast as FP64 input.
    tolerance = 1e-6
    report = {
        "samples": len(qpos),
        "frames": len(frame_names),
        "max_position_error_m": float(errors[..., :3, 3].max()),
        "max_rotation_matrix_error": float(errors[..., :3, :3].max()),
        "urdf_import_precision": "float32",
        "tolerance": tolerance,
        "passed": bool(errors.max() < tolerance),
    }
    np.savez(
        output / "kinematics-reference.npz",
        joint_names=np.array(names),
        frame_names=np.array(frame_names),
        qpos=qpos,
        poses=source,
    )
    _save(output / "kinematics-report.json", report)
    if not report["passed"]:
        raise RuntimeError(f"URDF kinematics differ from MuJoCo: {report}")
    return report


def _validate_composite_inertia(
    env: Any, output: Path, robot: dict[str, Any]
) -> dict[str, Any]:
    """Compare generalized inertia and gravity with an independent MuJoCo import."""
    import mujoco

    xml = ET.fromstring((output / "robot/robot.urdf").read_text())
    # Geometry does not determine the explicitly authored inertias. Remove it
    # so this independent URDF import also works without mesh search paths.
    for link in xml.findall("link"):
        for element in list(link):
            if element.tag in {"visual", "collision"}:
                link.remove(element)
    imported = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    imported.opt.gravity[:] = env.model.opt.gravity
    rotation = np.array(robot["root_pose"])[:3, :3]
    if not np.allclose(
        rotation @ imported.opt.gravity, imported.opt.gravity, atol=1e-12
    ):
        raise ValueError(
            "this official Cube inertia check assumes a gravity-preserving root rotation"
        )
    data = mujoco.MjData(imported)
    source = mujoco.MjData(env.model)
    with np.load(output / "kinematics-reference.npz", allow_pickle=False) as file:
        names, configurations = file["joint_names"].tolist(), file["qpos"]
    ref_dofs = [int(env.model.joint(name).dofadr[0]) for name in names]
    dofs = [int(imported.joint(name).dofadr[0]) for name in names]
    armature = np.diag(env.model.dof_armature[ref_dofs])
    mass_error, gravity_error = [], []
    ref_matrix, matrix = np.empty((env.model.nv, env.model.nv)), np.empty(
        (imported.nv, imported.nv)
    )
    for row in configurations:
        source.qpos[:] = env.data.qpos
        for name, value in zip(names, row):
            source.joint(name).qpos[:] = value
            data.joint(name).qpos[:] = value
        mujoco.mj_forward(env.model, source)
        mujoco.mj_forward(imported, data)
        mujoco.mj_fullM(env.model, source, ref_matrix)
        mujoco.mj_fullM(imported, data, matrix)
        expected = ref_matrix[np.ix_(ref_dofs, ref_dofs)] - armature
        actual = matrix[np.ix_(dofs, dofs)]
        mass_error.append(float(np.max(np.abs(actual - expected))))
        gravity_error.append(
            float(np.max(np.abs(data.qfrc_bias[dofs] - source.qfrc_bias[ref_dofs])))
        )
    report = {
        "samples": len(configurations),
        "joint_names": names,
        "max_generalized_inertia_error": max(mass_error),
        "max_gravity_torque_error_nm": max(gravity_error),
        "armature_removed_from_reference": True,
        "passed": max(mass_error) < 1e-9 and max(gravity_error) < 1e-9,
    }
    _save(output / "composite-inertia-report.json", report)
    if not report["passed"]:
        raise RuntimeError(f"collapsed robot changes source dynamics: {report}")
    return report


def audit_reference(
    output: Path, swm_root: Path, ogbench_root: Path, *, steps: int = 100
) -> dict[str, Any]:
    """Compile, export and record the actual official environment parameters."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    _revision(swm_root, CUBE_SWM_REV)
    _revision(ogbench_root, OGBENCH_REV)
    sys.path[:0] = [str(swm_root), str(ogbench_root)]
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
    from PIL import Image

    recipe = json.loads(CONFIG.read_text())
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(recipe["seed"])
    env = CubeEnv(**recipe["world"])
    try:
        env.action_space.seed(recipe["seed"])
        env.reset(seed=recipe["seed"])
        contract = _physical_contract(env)
        _save(output / "physical-contract.json", contract)
        _export_render_assets(env, output)
        robot = _export_robot(env, output / "robot")
        kinematics = _validate_kinematics(env, output, robot)
        composite_inertia = _validate_composite_inertia(env, output, robot)
        descriptions = ogbench_root / "ogbench/manipspace/descriptions"
        for folder in ("universal_robots_ur5e", "robotiq_2f85"):
            shutil.copyfile(
                descriptions / folder / "LICENSE",
                output / "robot" / (folder + ".LICENSE"),
            )
        shutil.copyfile(ogbench_root / "LICENSE", output / "robot/OGBench.LICENSE")
        Image.fromarray(env.render()).save(output / "reference-reset.png")
        rng = np.random.default_rng(recipe["seed"])
        actions = rng.uniform(-1, 1, (steps, 5)).astype(np.float32)
        states = [env.data.qpos.copy()]
        velocities = [env.data.qvel.copy()]
        controls, times = [], [float(env.data.time)]
        for action in actions:
            env.step(action)
            states.append(env.data.qpos.copy())
            velocities.append(env.data.qvel.copy())
            controls.append(env.data.ctrl.copy())
            times.append(float(env.data.time))
        np.savez(
            output / "reference-random-replay.npz",
            qpos=np.array(states),
            qvel=np.array(velocities),
            controls=np.array(controls),
            actions=actions,
            time=np.array(times),
        )
        Image.fromarray(env.render()).save(output / "reference-after.png")
        report = {
            "revisions": {
                "ogbench": OGBENCH_REV,
                "stable_worldmodel": CUBE_SWM_REV,
                "lewm": LEWM_REV,
                "weights": CUBE_WEIGHTS_REV,
            },
            "runtime": {
                k: importlib.metadata.version(k)
                for k in ("mujoco", "dm_control", "gymnasium", "numpy")
            },
            "recipe": recipe,
            "reference_replay_steps": steps,
            "reference_replay_seconds": times[-1] - times[0],
            "robot_urdf_sha256": robot["sha256"],
            "kinematics": kinematics,
            "composite_inertia": composite_inertia,
            "dexsim_physics_alignment_verified": False,
        }
        _save(output / "reference-report.json", report)
        return report
    finally:
        if env._renderer is not None:
            env._renderer.close()
            env._renderer = None
        env.close()


def _compare_cem(output: Path) -> dict[str, Any]:
    """Compare all candidate populations to upstream on an analytic objective."""
    import gymnasium as gym
    import torch
    from types import SimpleNamespace
    from stable_worldmodel.solver.cem import CEMSolver
    from embodichain_models.dynamics.lewm import LeWMCEMCfg, LeWMCEMPlanner

    target = torch.linspace(-1.5, 1.5, 125).reshape(1, 1, 5, 25)

    class Objective(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.populations = []

        def get_cost(self, info: dict, candidates: torch.Tensor) -> torch.Tensor:
            self.populations.append(candidates.clone())
            return (candidates - target).square().sum((-1, -2))

    class Dynamics(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = SimpleNamespace(action_dim=25)
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.register_buffer("action_mean", torch.linspace(-0.3, 0.3, 25))
            self.register_buffer("action_std", torch.linspace(0.5, 1.5, 25))
            self.populations = []

        def rollout(
            self, history: torch.Tensor, past: torch.Tensor, raw: torch.Tensor
        ) -> torch.Tensor:
            normalized = (raw - self.action_mean) / self.action_std
            self.populations.append(normalized.clone())
            return normalized

        def goal_cost(
            self, predictions: torch.Tensor, goal: torch.Tensor
        ) -> torch.Tensor:
            return (predictions - target).square().sum((-1, -2))

    recipe = json.loads(CONFIG.read_text())
    config = recipe["planner"]
    objective = Objective()
    reference = CEMSolver(
        model=objective,
        batch_size=1,
        num_samples=config["num_samples"],
        var_scale=1,
        n_steps=config["iterations"],
        topk=config["topk"],
        device="cpu",
        seed=recipe["seed"],
    )
    reference.configure(
        action_space=gym.spaces.Box(-1, 1, (2, 5)),
        n_envs=2,
        config=SimpleNamespace(horizon=5, action_block=5),
    )
    expected = reference.solve({"context": torch.zeros(2, 1)})["actions"]
    model = Dynamics().eval()
    planner = LeWMCEMPlanner(
        model,
        -torch.ones(25),
        torch.ones(25),
        LeWMCEMCfg(**config),
        seed=recipe["seed"],
    )
    # The original solver finishes all iterations for one environment before
    # drawing noise for the next one, using a shared seeded generator.
    actual = torch.cat(
        [
            planner.plan_sequence(
                torch.zeros(1, 1, 1), torch.zeros(1, 0, 25), torch.zeros(1, 1)
            )[0]
            for _ in range(2)
        ]
    )
    expected_raw = expected * model.action_std + model.action_mean
    expected_pop = torch.stack(objective.populations)
    actual_pop = torch.stack(
        [p for p in model.populations if p.shape[1] == config["num_samples"]]
    )
    torch.testing.assert_close(actual_pop, expected_pop, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(actual, expected_raw, atol=2e-5, rtol=1e-5)
    report = {
        "passed": True,
        "objective": "analytic_standardized_action_l2",
        "environments": 2,
        "candidates": config["num_samples"],
        "iterations": config["iterations"],
        "horizon": 5,
        "action_dim": 25,
        "solver_batch_size": 1,
        "std_correction": config["std_correction"],
        "max_candidate_absolute_error": float((actual_pop - expected_pop).abs().max()),
        "max_action_absolute_error": float((actual - expected_raw).abs().max()),
        "dataset_action_statistics_used": False,
    }
    _save(output / "cem-reference.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--swm-root", type=Path, required=True)
    parser.add_argument("--ogbench-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--compare-cem", action="store_true")
    args = parser.parse_args()
    report = audit_reference(
        args.output, args.swm_root, args.ogbench_root, steps=args.steps
    )
    if args.compare_cem:
        report["cem"] = _compare_cem(args.output)
        _save(args.output / "reference-report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
