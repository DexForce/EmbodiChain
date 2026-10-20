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

"""GenSim robot variants sharing the canonical, complete Robotiq hand contract."""

from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.utils.utility import load_config
from .contracts import canonical_hash

__all__: list[str] = []

_FRANKA_COMPONENT = (
    Path(__file__).resolve().parents[3]
    / "embodichain_tasks/configs/components/embodiments/dual_franka_robotiq_arg2f_140.yaml"
)
_MARVIN_PROFILES = frozenset({"dual_tianji", "tianji_marvin"})
_NEW_PROFILES = frozenset({"dual_ur10"}) | _MARVIN_PROFILES
_NEW_EMBODIMENT_IDS = frozenset(
    {"dual_ur10_robotiq_arg2f_140", "tianji_marvin_robotiq_arg2f_140"}
)
_FRANKA_PROFILES = frozenset(
    {"dual_franka", "dual_franka_robotiq_arg2f_140", "dual_franka_pgi"}
)
_CLI_PROFILES = (
    "ur5",
    "ur10",
    "dual_ur5",
    "dual_ur10",
    "franka",
    "dual_franka",
    "dual_franka_pgi",
    "dual_tianji",
    "tianji_marvin",
)


def _canonical_robot_profile(profile: str) -> str:
    """Compare CLI aliases without rewriting serialized model identities or receipts."""
    return "dual_tianji" if profile == "tianji_marvin" else profile


def _validate_robot_profile(profile: str) -> None:
    if profile not in _FRANKA_PROFILES | _NEW_PROFILES:
        raise ValueError(
            "Semantic bundle generation supports only dual_franka, dual_ur10 "
            "and dual_tianji (legacy alias: tianji_marvin); "
            f"received robot_profile={profile!r}."
        )


def _require_robot_assets(profile: str) -> None:
    """Fail before generation/provider calls when a new robot's inputs are unavailable."""
    if profile not in _NEW_PROFILES:
        return
    from embodichain.data import get_data_path

    arm = (
        "UniversalRobots/UR10/UR10.urdf"
        if profile == "dual_ur10"
        else "TianjiMarvin/robot_acd.urdf"
    )
    for asset in (arm, "Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf"):
        try:
            path = Path(get_data_path(asset))
            if not path.is_file():
                raise FileNotFoundError(path)
        except (RuntimeError, OSError) as exc:
            raise ValueError(
                f"robot_profile={profile!r} requires the registered asset {asset!r}; "
                "resolve/download it before preparing this robot deployment."
            ) from exc


def _load_robotiq_embodiment(profile: str, output_dir: Path) -> dict[str, Any]:
    """Reuse the hand declaration without changing the source component or Lab defaults."""
    _validate_robot_profile(profile)
    payload = load_config(_FRANKA_COMPONENT)
    if profile not in _NEW_PROFILES:
        return payload
    simulation = payload["simulation"]
    if profile == "dual_ur10":
        from embodichain.lab.sim.robots import URRobotCfg

        arm = URRobotCfg.from_dict({"robot_type": "ur10"})
        simulation["urdf_cfg"]["fname"] = "dual_ur10_robotiq_arg2f_140"
        for component in simulation["urdf_cfg"]["components"]:
            side = component["component_type"].removesuffix("_arm")
            if side not in {"left", "right"}:
                continue
            component["urdf_path"] = "UniversalRobots/UR10/UR10.urdf"
            transform = np.eye(4)
            transform[1, 3] = 0.3 if side == "left" else -0.3
            component["transform"] = transform.tolist()
            simulation["control_parts"][f"{side}_arm"] = [
                f"{side}_{name}" for name in arm.control_parts["arm"]
            ]
            simulation["solver_cfg"][f"{side}_arm"] = {
                "class_type": "URSolver",
                "ur_type": "ur10",
                "root_link_name": f"{side}_base_link",
                "end_link_name": f"{side}_ee_link",
                "tcp": simulation["solver_cfg"][f"{side}_arm"]["tcp"],
            }
            for prop in ("stiffness", "damping", "max_effort"):
                simulation["joint_drive_props"][prop][f"{side}_arm"] = getattr(
                    arm.joint_drive_props, prop
                )["arm"]
        # At q1=0 this URDF reaches along local -X, away from the table after
        # the shared 180-degree mount yaw. Turn the shoulder, not the scene/TCP.
        home = [np.pi, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]
        # The two equally deep arm chains are interleaved in source DOF order.
        simulation["init_qpos"] = [q for q in home for _ in range(2)] + [0.0] * 12
    else:
        from embodichain.data import get_data_path
        from embodichain.lab.sim.robots import TianjiMarvinCfg

        source = Path(get_data_path("TianjiMarvin/robot_acd.urdf"))
        arm = TianjiMarvinCfg.from_dict({"with_gripper": False, "fpath": str(source)})
        assembled = _attach_marvin_hands(source, output_dir / "robot_model")
        simulation.pop("urdf_cfg")
        simulation["fpath"] = str(assembled)
        native = arm.to_dict()
        simulation["root_props"] = native["root_props"]
        for side in ("left", "right"):
            part = f"{side}_arm"
            simulation["control_parts"][part] = list(arm.control_parts[part])
            simulation["solver_cfg"][part].update(
                root_link_name=f"{side}_arm_base", end_link_name=f"{side}_ee"
            )
            for prop in ("stiffness", "damping", "max_effort"):
                simulation["joint_drive_props"][prop][part] = getattr(
                    arm.joint_drive_props, prop
                )[part]
        home = [0.0] * 7
        simulation["init_qpos"] = [0.0] * 26
    simulation["uid"] = "DualUR10" if profile == "dual_ur10" else "TianjiMarvin"
    simulation["init_pos"] = [0.55, 0.0, 0.0]
    simulation["control_parts"]["dual_arm"] = [
        name
        for side in ("left", "right")
        for name in simulation["control_parts"][f"{side}_arm"]
    ]
    # The public selector may change, but existing bundles own these model IDs.
    model_profile = "tianji_marvin" if profile in _MARVIN_PROFILES else profile
    payload["embodiment_id"] = f"{model_profile}_robotiq_arg2f_140"
    payload["skill_profile"]["profile_id"] = payload["embodiment_id"]
    for preset in payload["skill_profile"]["command_presets"]:
        if preset["preset_id"] in {"left_arm_postures", "right_arm_postures"}:
            preset["commands"]["park"] = list(home)
    return payload


def _attach_marvin_hands(source: Path, output_dir: Path) -> Path:
    """Keep the entire bare robot and attach the same two processed Robotiq hands."""
    from embodichain.data import get_data_path
    from embodichain.toolkits.urdf_assembly import URDFAssemblyManager

    bare = ET.parse(source).getroot()
    source_links = {link.attrib["name"] for link in bare.findall("link")}
    if not {"left_ee", "right_ee", "left_arm_base", "right_arm_base"} <= source_links:
        raise ValueError("TianjiMarvin bare URDF lacks the declared arm/tool frames.")
    movable = [j for j in bare.findall("joint") if j.get("type") != "fixed"]
    if len(movable) != 14 or any(j.find("mimic") is not None for j in movable):
        raise ValueError(
            "TianjiMarvin requires the complete bare fourteen-joint model."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manager = URDFAssemblyManager()
    manager.base_link_name = "gen_sim_base"
    manager.name_case = {"joint": "original", "link": "original"}
    if not manager.add_component("chassis", str(source)):
        raise ValueError("Unable to assemble the TianjiMarvin body.")
    hand_path = get_data_path("Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf")
    for side in ("left", "right"):
        if not manager.add_component(f"{side}_hand", hand_path):
            raise ValueError(f"Unable to assemble the {side} Robotiq hand.")
    path = output_dir / "tianji_marvin_robotiq_arg2f_140.urdf"
    root = manager.merge_urdfs(str(path), use_signature_check=False)
    # Whole-body URDFs are not split into the assembler's left/right arm slots.
    # Add explicit fixed wrist mounts without changing body or hand kinematics.
    for side in ("left", "right"):
        hand_root = f"{side}_robotiq_arg2f_base_link"
        connectors = [
            joint
            for joint in root.findall("joint")
            if joint.find("child").get("link") == hand_root
        ]
        if not connectors:
            joint = ET.SubElement(
                root, "joint", name=f"{side}_robotiq_mount", type="fixed"
            )
            ET.SubElement(joint, "parent", link=f"{side}_ee")
            ET.SubElement(joint, "child", link=hand_root)
            ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
        elif len(connectors) != 1 or connectors[0].get("type") != "fixed":
            raise ValueError(f"Ambiguous {side} Robotiq mounting connector.")
        else:
            connectors[0].find("parent").set("link", f"{side}_ee")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _fixed_frame_pose(path: Path, frame: str) -> np.ndarray:
    """Resolve an arm base through fixed joints, rejecting a moving torso."""
    root = ET.parse(path).getroot()
    links = {link.get("name") for link in root.findall("link")}
    joints = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
    if frame not in links:
        raise ValueError(f"Robot arm base frame {frame!r} is absent.")
    pose, visited = np.eye(4), set()
    while frame in joints:
        if frame in visited:
            raise ValueError("Robot base attachment chain contains a cycle.")
        visited.add(frame)
        joint = joints[frame]
        if joint.get("type") != "fixed":
            raise ValueError("GenSim mounting requires fixed arm-base attachments.")
        transform = np.eye(4)
        origin = joint.find("origin")
        if origin is not None:
            transform[:3, 3] = [float(x) for x in origin.get("xyz", "0 0 0").split()]
            transform[:3, :3] = Rotation.from_euler(
                "xyz", [float(x) for x in origin.get("rpy", "0 0 0").split()]
            ).as_matrix()
        pose = transform @ pose
        frame = joint.find("parent").get("link")
    return pose


def _arm_base_position(embodiment: dict[str, Any], resource: str) -> np.ndarray:
    simulation = embodiment["simulation"]
    for component in simulation.get("urdf_cfg", {}).get("components", ()):
        if component["component_type"] == f"{resource}_arm":
            return np.asarray(component["transform"], dtype=float)[:3, 3]
    solver = simulation["solver_cfg"][f"{resource}_arm"]
    return _fixed_frame_pose(Path(simulation["fpath"]), solver["root_link_name"])[:3, 3]


def _bind_new_mount(embodiment: dict[str, Any], table_top_z: float) -> bool:
    if embodiment.get("embodiment_id") not in _NEW_EMBODIMENT_IDS:
        return False
    simulation = embodiment["simulation"]
    anchors = np.stack(
        [_arm_base_position(embodiment, side) for side in ("left", "right")]
    )
    rotation = Rotation.from_euler(
        "XYZ", simulation["init_rot"], degrees=True
    ).as_matrix()
    # Match the old deployment's world arm-base midpoint, not its synthetic root.
    midpoint = rotation @ anchors.mean(axis=0)
    simulation["init_pos"] = (
        np.array([0.55, 0.0, table_top_z + 0.05]) - midpoint
    ).tolist()
    return True


def _robot_asset_digest(embodiment: dict[str, Any], root: Path) -> str | None:
    """Seal new robot sources and their mesh dependencies, not just their paths."""
    if embodiment.get("embodiment_id") not in _NEW_EMBODIMENT_IDS:
        return None
    from ._task_program.physical_inputs import _asset_files

    simulation = embodiment["simulation"]
    owned_model = root / "components/robot_model/tianji_marvin_robotiq_arg2f_140.urdf"
    if (
        embodiment["embodiment_id"] == "tianji_marvin_robotiq_arg2f_140"
        and owned_model.is_file()
    ):
        # Publication rewrites YAML before renaming the staging directory. Its
        # bundle-owned model remains at this physical location in both phases.
        simulation = {**simulation, "fpath": str(owned_model)}
    files = _asset_files(simulation, base_dir=root)
    return canonical_hash(
        {
            str(
                path.relative_to(root) if path.is_relative_to(root) else path
            ): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)
        }
    )


def _profile_lateral_positions(
    profile: str, embodiment: dict[str, Any] | None = None
) -> dict[str, float] | None:
    if profile in _NEW_PROFILES and embodiment is not None:
        simulation = embodiment["simulation"]
        if simulation.get("init_local_pose") is not None:
            pose = np.asarray(simulation["init_local_pose"], dtype=float)
        else:
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler(
                "XYZ", simulation.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
            ).as_matrix()
            pose[:3, 3] = simulation.get("init_pos", [0.0, 0.0, 0.0])
        return {
            side: float(
                (pose[:3, :3] @ _arm_base_position(embodiment, side) + pose[:3, 3])[1]
            )
            for side in ("left", "right")
        }
    if profile not in _MARVIN_PROFILES:
        return None
    from embodichain.data import get_data_path

    source = Path(get_data_path("TianjiMarvin/robot_acd.urdf"))
    lateral = {
        side: -float(_fixed_frame_pose(source, f"{side}_arm_base")[1, 3])
        for side in ("left", "right")
    }
    midpoint = sum(lateral.values()) / 2
    return {side: value - midpoint for side, value in lateral.items()}
