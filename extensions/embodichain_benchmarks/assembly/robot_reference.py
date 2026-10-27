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
"""Prepare the pinned public Rizon model for the URDF simulation boundary."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["prepare_rizon4s_urdf"]

_CONVERTER_VERSION = 4


def _numbers(values: np.ndarray) -> str:
    return " ".join(format(float(v), ".17g") for v in values)


def _rotation(quat_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.roll(quat_wxyz, -1))


def _origin(element: ET.Element, position: np.ndarray, rotation: Rotation) -> None:
    ET.SubElement(
        element,
        "origin",
        xyz=_numbers(position),
        rpy=_numbers(rotation.as_euler("xyz")),
    )


def _append_model(
    robot: ET.Element,
    model: object,
    output: Path,
    *,
    prefix: str = "",
    attach_parent: str | None = None,
    attach_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gripper_mimics: bool = False,
) -> None:
    """Copy compiled public mesh, inertial and joint data without re-fitting."""
    import mujoco

    mesh_paths: dict[int, str] = {}
    for mesh_id in range(model.nmesh):
        va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        mesh_path = output / "meshes" / f"{prefix}mesh_{mesh_id}.obj"
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        with mesh_path.open("w") as stream:
            for vertex in model.mesh_vert[va : va + vn]:
                stream.write("v " + _numbers(vertex) + "\n")
            for face in model.mesh_face[fa : fa + fn]:
                stream.write("f " + " ".join(str(int(i) + 1) for i in face) + "\n")
        mesh_paths[mesh_id] = str(mesh_path.relative_to(output))

    offsets = np.zeros((model.nbody, 3))
    for body_id in range(1, model.nbody):
        nj = int(model.body_jntnum[body_id])
        if nj > 1:
            raise ValueError(
                "The pinned robot tree requires at most one joint per body."
            )
        if nj:
            offsets[body_id] = model.jnt_pos[int(model.body_jntadr[body_id])]

    for body_id in range(1, model.nbody):
        body_name = prefix + model.body(body_id).name
        link = ET.SubElement(robot, "link", name=body_name)
        mass = float(model.body_mass[body_id])
        if mass > 0:
            inertial = ET.SubElement(link, "inertial")
            _origin(
                inertial,
                model.body_ipos[body_id] - offsets[body_id],
                _rotation(model.body_iquat[body_id]),
            )
            ET.SubElement(inertial, "mass", value=format(mass, ".17g"))
            principal = model.body_inertia[body_id]
            ET.SubElement(
                inertial,
                "inertia",
                ixx=str(principal[0]),
                iyy=str(principal[1]),
                izz=str(principal[2]),
                ixy="0",
                ixz="0",
                iyz="0",
            )
        start, count = int(model.body_geomadr[body_id]), int(
            model.body_geomnum[body_id]
        )
        for geom_id in range(start, start + count):
            collision = bool(
                model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]
            )
            for tag in (("visual", "collision") if collision else ("visual",)):
                element = ET.SubElement(link, tag)
                _origin(
                    element,
                    model.geom_pos[geom_id] - offsets[body_id],
                    _rotation(model.geom_quat[geom_id]),
                )
                geometry = ET.SubElement(element, "geometry")
                kind = int(model.geom_type[geom_id])
                if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                    ET.SubElement(
                        geometry,
                        "mesh",
                        filename=mesh_paths[int(model.geom_dataid[geom_id])],
                    )
                elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                    ET.SubElement(
                        geometry, "box", size=_numbers(2 * model.geom_size[geom_id])
                    )
                elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    ET.SubElement(
                        geometry, "sphere", radius=str(model.geom_size[geom_id, 0])
                    )
                elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                    ET.SubElement(
                        geometry,
                        "cylinder",
                        radius=str(model.geom_size[geom_id, 0]),
                        length=str(2 * model.geom_size[geom_id, 1]),
                    )
                else:
                    raise ValueError(f"Unsupported geometry in pinned robot: {kind}")
                if tag == "visual":
                    material = ET.SubElement(
                        element, "material", name=f"{body_name}_{geom_id}"
                    )
                    ET.SubElement(
                        material, "color", rgba=_numbers(model.geom_rgba[geom_id])
                    )
        parent_id = int(model.body_parentid[body_id])
        if parent_id == 0 and attach_parent is None:
            continue
        parent_name = (
            attach_parent if parent_id == 0 else prefix + model.body(parent_id).name
        )
        joint_count = int(model.body_jntnum[body_id])
        joint_id = int(model.body_jntadr[body_id]) if joint_count else -1
        joint_name = (
            prefix + model.joint(joint_id).name if joint_count else body_name + "_fixed"
        )
        joint = ET.SubElement(
            robot, "joint", name=joint_name, type="revolute" if joint_count else "fixed"
        )
        ET.SubElement(joint, "parent", link=parent_name)
        ET.SubElement(joint, "child", link=body_name)
        rotation = _rotation(model.body_quat[body_id])
        position = (
            model.body_pos[body_id]
            + rotation.apply(offsets[body_id])
            - offsets[parent_id]
        )
        if parent_id == 0:
            position = position + np.asarray(attach_position)
        _origin(joint, position, rotation)
        if joint_count:
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError("Pinned robot conversion only supports hinge joints.")
            ET.SubElement(joint, "axis", xyz=_numbers(model.jnt_axis[joint_id]))
            low, high = model.jnt_range[joint_id]
            effort = float(np.max(np.abs(model.jnt_actfrcrange[joint_id])))
            ET.SubElement(
                joint,
                "limit",
                lower=str(low),
                upper=str(high),
                effort=str(effort or 10.0),
                velocity="2.0",
            )
            ET.SubElement(
                joint,
                "dynamics",
                damping=str(float(model.dof_damping[int(model.jnt_dofadr[joint_id])])),
                friction="0",
            )
            if gripper_mimics and model.joint(joint_id).name != "right_driver_joint":
                multiplier = (
                    -1
                    if any(
                        s in model.joint(joint_id).name for s in ("coupler", "follower")
                    )
                    else 1
                )
                ET.SubElement(
                    joint,
                    "mimic",
                    joint=prefix + "right_driver_joint",
                    multiplier=str(multiplier),
                    offset="0",
                )


def prepare_rizon4s_urdf(
    source_root: str | Path, *, gripper_model: str = "four_bar_mimic"
) -> str:
    """Prepare a nominal Rizon arm and explicitly approximate mimic gripper.

    The source arm's geometry, inertia and kinematics are preserved. Robotiq's
    two four-bar connect constraints and tendon are represented by a parallel
    mimic model, which is *not* dynamically equivalent to the source MJCF.
    A fidelity manifest records this restriction next to the prepared URDF.

    Args:
        source_root: Downloaded ``FlexivRizon4SMenagerie`` source bundle.
        gripper_model: ``parallel_jaw`` or ``four_bar_mimic`` approximation.

    Returns:
        Path to the prepared URDF, reusing an existing versioned artifact.

    Raises:
        ValueError: If the requested gripper approximation is unsupported.
        ImportError: If preparation needs the optional MuJoCo dependency.
    """
    root = Path(source_root)
    if gripper_model not in {"four_bar_mimic", "parallel_jaw"}:
        raise ValueError("gripper_model must be four_bar_mimic or parallel_jaw.")
    output = root / (
        f"prepared_v{_CONVERTER_VERSION}"
        + ("_parallel_jaw" if gripper_model == "parallel_jaw" else "")
    )
    urdf = output / "rizon4s_robotiq2f85.urdf"
    marker = output / "fidelity.json"
    if urdf.exists() and marker.exists():
        return str(urdf)
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "Preparing the public Rizon assets requires the optional assembly extra; already prepared URDFs do not."
        ) from exc
    output.mkdir(parents=True, exist_ok=True)
    robot = ET.Element("robot", name="rizon4s_robotiq2f85")
    arm = mujoco.MjModel.from_xml_path(str(root / "flexiv_rizon4s/flexiv_rizon4s.xml"))
    hand = mujoco.MjModel.from_xml_path(str(root / "robotiq_2f85/2f85.xml"))
    _append_model(robot, arm, output)
    _append_model(
        robot,
        hand,
        output,
        prefix="gripper_",
        attach_parent="link7",
        attach_position=(0, 0, 0.088),
        gripper_mimics=True,
    )
    if gripper_model == "parallel_jaw":
        # Explicit nominal approximation: each source finger's complete mass
        # moves on a force-limited prismatic carriage. Finger link transforms
        # are fixed at the open four-bar configuration; only the real pad and
        # base collision geometry participates in contact.
        for element in list(robot.findall("joint")):
            if not element.attrib["name"].startswith("gripper_"):
                continue
            child = element.find("child").attrib["link"]
            for field in ("axis", "limit", "dynamics", "mimic"):
                found = element.find(field)
                if found is not None:
                    element.remove(found)
            element.set("type", "fixed")
            for side in ("left", "right"):
                if child in {f"gripper_{side}_driver", f"gripper_{side}_spring_link"}:
                    element.find("parent").set("link", f"gripper_{side}_carriage")
            if child in {"gripper_left_pad", "gripper_right_pad"}:
                origin = element.find("origin")
                position = np.fromstring(origin.attrib["xyz"], sep=" ")
                position[2] += 0.0146
                origin.set("xyz", _numbers(position))
        for side, direction in (("left", 1), ("right", -1)):
            ET.SubElement(robot, "link", name=f"gripper_{side}_carriage")
            slider = ET.SubElement(
                robot, "joint", name=f"gripper_{side}_jaw_joint", type="prismatic"
            )
            ET.SubElement(slider, "parent", link="gripper_base")
            ET.SubElement(slider, "child", link=f"gripper_{side}_carriage")
            _origin(slider, np.zeros(3), Rotation.identity())
            ET.SubElement(slider, "axis", xyz=f"0 {direction} 0")
            ET.SubElement(
                slider, "limit", lower="0", upper="0.0425", effort="30", velocity="0.2"
            )
            ET.SubElement(slider, "dynamics", damping="1", friction="0")
        for link in robot.findall("link"):
            name = link.attrib["name"]
            if name.startswith("gripper_") and name not in {
                "gripper_base",
                "gripper_base_mount",
                "gripper_left_pad",
                "gripper_right_pad",
            }:
                for collision in list(link.findall("collision")):
                    link.remove(collision)
    ET.SubElement(robot, "link", name="gripper_tcp")
    joint = ET.SubElement(robot, "joint", name="gripper_tcp_fixed", type="fixed")
    ET.SubElement(joint, "parent", link="gripper_base")
    ET.SubElement(joint, "child", link="gripper_tcp")
    _origin(joint, np.array([0, 0, 0.145]), Rotation.identity())
    ET.indent(robot)
    ET.ElementTree(robot).write(urdf, encoding="utf-8", xml_declaration=True)
    marker.write_text(
        json.dumps(
            {
                "converter_version": _CONVERTER_VERSION,
                "source_manifest": str(root / "source_manifest.json"),
                "arm": "source kinematics, mass, COM, principal inertia and compiled collision geometry",
                "gripper": (
                    "force-limited prismatic finger carriages; four-bar configuration fixed open within each moving carriage; source finger masses and base/pad collision geometry retained"
                    if gripper_model == "parallel_jaw"
                    else "parallel mimic approximation; source connect constraints and tendon not preserved"
                ),
                "gripper_model": gripper_model,
                "mount_z_m": 0.088,
                "mount_source": "engineering flange placement; not author calibration",
                "paper_dynamics_identified": False,
                "exact_source_dynamics": False,
            },
            indent=2,
        )
        + "\n"
    )
    return str(urdf)


if __name__ == "__main__":
    import argparse
    from embodichain.data import get_data_path

    parser = argparse.ArgumentParser(
        description="Prepare the historical nominal assembly robot reference."
    )
    parser.add_argument(
        "--gripper-model",
        choices=("parallel_jaw", "four_bar_mimic"),
        default="parallel_jaw",
    )
    args = parser.parse_args()
    print(
        prepare_rizon4s_urdf(
            get_data_path("FlexivRizon4SMenagerie"), gripper_model=args.gripper_model
        )
    )
