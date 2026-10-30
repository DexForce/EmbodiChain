# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Actual registered hand assets, openings and explicitly calibrated TCP frames.

Canonical contact frames have X closing and Z approaching. Changing a runtime
TCP calibration changes returned poses, never the physical collision template.
This module resolves data only; it neither constructs a robot nor a simulator.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.data import get_data_path
from .fixed_geometry import FixedGripperCollisionGeometry

__all__ = ["get_parallel_jaw_collision_geometry"]

_MODELS = (
    "dh_pgi_140_80",
    "franka_panda_hand",
    "cobotmagic_v100_gripper",
    "tianji_marvin_gripper",
    "pika_gripper",
)


def _pose(rotation: np.ndarray | None = None, xyz=(0.0, 0.0, 0.0)) -> np.ndarray:
    value = np.eye(4)
    if rotation is not None:
        value[:3, :3] = rotation
    value[:3, 3] = xyz
    return value


def get_parallel_jaw_collision_geometry(
    model_id: str,
    *,
    point_spacing: float = 0.012,
    side: str | None = None,
    root_from_tcp: np.ndarray | None = None,
) -> FixedGripperCollisionGeometry:
    """Resolve actual open-hand collision surfaces for one supported model.

    Args:
        model_id: One of the five registered physical hand models.
        point_spacing: Maximum triangle-grid edge spacing in metres.
        side: Required ``left`` or ``right`` for asymmetric Marvin assets;
            omitted for the other models.
        root_from_tcp: Optional calibrated runtime TCP in the returned
            provenance's ``root_link`` frame. Omission uses the registered
            default TCP; deployments with a different solver calibration must
            supply that calibration. This override preserves the physical contact frame
            and collision points, including when the TCP is a virtual frame.

    Returns:
        Owned collision geometry, exact source identities and the transform
        converting canonical candidates to this caller's runtime TCP.

    Raises:
        ValueError: Unknown models, missing/unsupported sides or invalid frames.
        FileNotFoundError: A registered asset cannot be resolved.
    """
    if model_id not in _MODELS:
        raise ValueError(
            f"No collision geometry for {model_id!r}; available models: {_MODELS}. "
            "Custom hands require explicit URDF collision geometry."
        )
    if model_id == "tianji_marvin_gripper":
        if side not in ("left", "right"):
            raise ValueError(
                "Marvin collision geometry requires explicit left/right side."
            )
    elif side is not None:
        raise ValueError(f"{model_id!r} has no side-specific geometry.")

    rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    if model_id == "franka_panda_hand":
        asset = "Franka/Panda/PandaWithHand.urdf"
        root = "fr3_hand"
        links = (root, "fr3_leftfinger", "fr3_rightfinger")
        joints = {"fr3_finger_joint1": 0.04, "fr3_finger_joint2": 0.04}
        default_tcp = _pose(xyz=(0, 0, 0.1034))
        canonical = default_tcp @ _pose(rz90)
        calibration = "PandaWithHand fr3_hand_tcp joint; native +Y closes"
    elif model_id == "dh_pgi_140_80":
        asset = "DH_PGI_140_80/DH_PGI_140_80.urdf"
        root = "gripper_base_link_1"
        links = (root, "gripper_finger1_link_1", "gripper_finger2_link_1")
        joints = {"gripper_finger1_joint_1": 0.0, "gripper_finger2_joint_1": 0.0}
        # The shipped UR5/PGI embodiment and tutorial grasp TCP name this
        # contact centre explicitly. General/tool-point overrides are distinct.
        canonical = default_tcp = _pose(xyz=(0, 0, 0.15))
        calibration = "ur5_dh_pgi_140_80 embodiment; PGI grasp TCP at base Z=.15"
    elif model_id == "cobotmagic_v100_gripper":
        asset = "CobotMagicArm/CobotMagicWithGripperV100.urdf"
        root = "gripper_base"
        links = (root, "link7", "link8")
        joints = {"joint7": 0.05, "joint8": 0.05}
        default_tcp = _pose(np.diag([-1.0, -1.0, 1.0]), (0, 0, 0.143))
        canonical = default_tcp @ _pose(rz90)
        calibration = "CobotMagicCfg link6 TCP; identity link6-to-gripper_base"
    elif model_id == "pika_gripper":
        asset = "PikaGripper/PikaGripper.urdf"
        root = "GRIPPER_ROOT_LINK"
        links = ("GRIPPER_BASE", "FINGER1", "FINGER2")
        joints = {"FINGER1_JOINT": 0.05, "FINGER2_JOINT": 0.05}
        # The asset includes its root adapter: base X approaches, base Y
        # closes. The shipped W1/Pika TCP already aligns canonical X/Z.
        canonical = default_tcp = _pose(rz90.T, (0.00064, 0, 0.16))
        calibration = (
            "dexforce_w1_030_pika embodiment; standalone root adapter included"
        )
    else:
        asset = "TianjiMarvin/robot_with_ee_acd.urdf"
        root = f"{side}_hand_base_link"
        links = (root, f"{side}_hand_finger_1_link", f"{side}_hand_finger_2_link")
        joints = {
            f"{side.upper()}_HAND_FINGER_1": 0.0,
            f"{side.upper()}_HAND_FINGER_2": 0.0,
        }
        # Preserve the actual rounded URDF rotation instead of replacing it
        # with an ideal quarter-turn. The solver offset is relative to TOOL.
        path = Path(get_data_path(asset))
        tool = ET.parse(path).find(f"./joint[@name='{side.upper()}_HAND_TOOL']/origin")
        tool_transform = _pose(
            Rotation.from_euler(
                "xyz", np.fromstring(tool.attrib["rpy"], sep=" ")
            ).as_matrix(),
            np.fromstring(tool.attrib["xyz"], sep=" "),
        )
        default_tcp = tool_transform @ _pose(
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), (0.13, 0, 0)
        )
        canonical = _pose(xyz=(0, 0, 0.13))
        calibration = (
            "TianjiMarvinCfg default hand_tool_link-to-TCP transform and actual URDF tool joint; "
            "canonical contact origin Z=.13 is a configuration convention, not a measured pad centre"
        )

    runtime_tcp = default_tcp if root_from_tcp is None else np.asarray(root_from_tcp)
    # This registered archive relocates its URDF to the asset root while
    # retaining ../Collision in three filenames. Bind those exact known paths;
    # the generic loader continues to reject unresolved paths without guessing.
    mesh_paths = (
        {
            f"../Collision/{name}": get_data_path(f"CobotMagicArm/Collision/{name}")
            for name in ("gripper_base_v100_acd.obj", "link7_acd.obj", "link8_acd.obj")
        }
        if model_id == "cobotmagic_v100_gripper"
        else None
    )
    geometry = FixedGripperCollisionGeometry.from_urdf(
        get_data_path(asset),
        root_link=root,
        tcp_link=root,
        link_from_tcp=runtime_tcp,
        collision_links=links,
        joint_positions=joints,
        grasp_from_tcp=np.linalg.inv(canonical) @ runtime_tcp,
        point_spacing=point_spacing,
        mesh_paths=mesh_paths,
    )
    return FixedGripperCollisionGeometry(
        geometry.points,
        {
            **geometry.provenance,
            "model_id": model_id,
            "side": side,
            "root_from_contact_frame": canonical.tolist(),
            "default_root_from_tcp": default_tcp.tolist(),
            "runtime_tcp_overridden": root_from_tcp is not None,
            "calibration_source": calibration,
        },
    )
