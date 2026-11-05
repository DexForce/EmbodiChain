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
from embodichain.data.asset_dependencies import asset_dependency_files
from types import MappingProxyType
from .fixed_geometry import FixedGripperCollisionGeometry, _rigid_transform

__all__ = ["get_parallel_jaw_collision_geometry", "get_parallel_jaw_model_asset_files"]

_MODEL_ASSETS = MappingProxyType(
    {
        "dh_pgi_140_80": "DH_PGI_140_80/DH_PGI_140_80.urdf",
        "franka_panda_hand": "Franka/Panda/PandaWithHand.urdf",
        "cobotmagic_v100_gripper": "CobotMagicArm/CobotMagicWithGripperV100.urdf",
        "tianji_marvin_gripper": "TianjiMarvin/robot_with_ee_acd.urdf",
        "pika_gripper": "PikaGripper/PikaGripper.urdf",
        "robotiq_2f85": "Robotiq/robotiq_arg2f_85/robotiq_arg2f_85.urdf",
        "robotiq_arg2f_140": "Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf",
    }
)
_MODELS = tuple(_MODEL_ASSETS)

_MODEL_COLLISION_LINKS = MappingProxyType(
    {
        "dh_pgi_140_80": (
            "gripper_base_link_1",
            "gripper_finger1_link_1",
            "gripper_finger2_link_1",
        ),
        "franka_panda_hand": ("fr3_hand", "fr3_leftfinger", "fr3_rightfinger"),
        "cobotmagic_v100_gripper": ("gripper_base", "link7", "link8"),
        "tianji_marvin_gripper": (
            "{side}_hand_base_link",
            "{side}_hand_finger_1_link",
            "{side}_hand_finger_2_link",
        ),
        "pika_gripper": ("GRIPPER_BASE", "FINGER1", "FINGER2"),
        "robotiq_2f85": (
            "robotiq_arg2f_base_link",
            "left_outer_knuckle",
            "left_outer_finger",
            "left_inner_knuckle",
            "left_inner_finger",
            "left_inner_finger_pad",
            "right_outer_knuckle",
            "right_outer_finger",
            "right_inner_knuckle",
            "right_inner_finger",
            "right_inner_finger_pad",
        ),
        "robotiq_arg2f_140": (
            "robotiq_arg2f_base_link",
            "left_outer_knuckle",
            "left_outer_finger",
            "left_inner_knuckle",
            "left_inner_finger",
            "left_inner_finger_pad",
            "right_outer_knuckle",
            "right_outer_finger",
            "right_inner_knuckle",
            "right_inner_finger",
            "right_inner_finger_pad",
        ),
    }
)


def _model_collision_links(model_id: str, side: str | None) -> tuple[str, ...]:
    return tuple(name.format(side=side) for name in _MODEL_COLLISION_LINKS[model_id])


def _model_mesh_paths(model_id: str) -> dict[str, str] | None:
    # This archive relocates its URDF while retaining these exact source names.
    return (
        {
            f"../Collision/{name}": get_data_path(f"CobotMagicArm/Collision/{name}")
            for name in ("gripper_base_v100_acd.obj", "link7_acd.obj", "link8_acd.obj")
        }
        if model_id == "cobotmagic_v100_gripper"
        else None
    )


def _validate_model_source(model_id: str, side: str | None) -> None:
    if model_id not in _MODELS:
        raise ValueError(
            f"No collision geometry for {model_id!r}; available models: {_MODELS}. Custom hands require explicit URDF collision geometry."
        )
    if model_id == "tianji_marvin_gripper":
        if side not in ("left", "right"):
            raise ValueError(
                "Marvin collision geometry requires explicit left/right side."
            )
    elif side is not None:
        raise ValueError(f"{model_id!r} has no side-specific geometry.")


def get_parallel_jaw_model_asset_files(
    model_id: str, *, side: str | None = None
) -> tuple[Path, ...]:
    """Enumerate declared files for the same registered collision model.

    Args:
        model_id: Registered physical hand model used by the geometry factory.
        side: Required left/right selection for standard Marvin, otherwise None.

    Returns:
        Sorted resolved source and transitive local dependencies, with the same
        explicit archive reference overrides as collision geometry. No sampler,
        geometry, robot, simulator or solver is instantiated.

    Raises:
        ValueError: Model/side is unsupported or dependencies cannot be inventoried.
        FileNotFoundError: A declared source or dependency is unavailable.
    """
    _validate_model_source(model_id, side)
    return asset_dependency_files(
        Path(get_data_path(_MODEL_ASSETS[model_id])),
        reference_overrides=_model_mesh_paths(model_id),
        urdf_collision_links=_model_collision_links(model_id, side),
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
    root_from_contact_frame: np.ndarray | None = None,
) -> FixedGripperCollisionGeometry:
    """Resolve actual open-hand collision surfaces for one supported model.

    Args:
        model_id: One of the registered physical hand models.
        point_spacing: Maximum triangle-grid edge spacing in metres.
        side: Required ``left`` or ``right`` for asymmetric Marvin assets;
            omitted for the other models.
        root_from_tcp: Optional calibrated runtime TCP in the returned
            provenance's ``root_link`` frame. Omission uses the registered
            default TCP; deployments with a different solver calibration must
            supply that calibration. This override preserves the physical contact frame
            and collision points, including when the TCP is a virtual frame.
        root_from_contact_frame: Optional calibrated canonical contact frame in
            hand-root coordinates. Omission retains the registered contact frame.
            This re-expresses collision samples without changing root-space surfaces.

    Returns:
        Owned collision geometry, exact source identities and the transform
        converting canonical candidates to this caller's runtime TCP.

    Raises:
        ValueError: Unknown models, missing/unsupported sides or invalid frames.
        FileNotFoundError: A registered asset cannot be resolved.
    """
    _validate_model_source(model_id, side)
    asset = _MODEL_ASSETS[model_id]
    links = _model_collision_links(model_id, side)

    rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    if model_id == "franka_panda_hand":
        root = "fr3_hand"
        joints = {"fr3_finger_joint1": 0.04, "fr3_finger_joint2": 0.04}
        default_tcp = _pose(xyz=(0, 0, 0.1034))
        canonical = default_tcp @ _pose(rz90)
        calibration = "PandaWithHand fr3_hand_tcp joint; native +Y closes"
    elif model_id in {"robotiq_2f85", "robotiq_arg2f_140"}:
        variant = "140" if model_id == "robotiq_arg2f_140" else "85"
        root = "robotiq_arg2f_base_link"
        joints = {
            name: 0.0
            for name in (
                "finger_joint",
                "left_inner_knuckle_joint",
                "left_inner_finger_joint",
                "right_outer_knuckle_joint",
                "right_inner_knuckle_joint",
                "right_inner_finger_joint",
            )
        }
        if variant == "140":
            # The GenSim arm-to-hand mount is identity; its solver TCP rotates
            # native Y closing into canonical X at root Z=.20.
            canonical = default_tcp = _pose(rz90, (0, 0, 0.20))
            calibration = (
                "GenSim Robotiq 2F-140 fully open collision meshes; "
                "native Y opens, root TCP Rz=90deg/Z=.20"
            )
        else:
            default_tcp = _pose(xyz=(0, 0, 0.15))
            canonical = default_tcp @ _pose(rz90)
            calibration = "Robotiq 2F-85 fully open collision meshes; native Y opens, tool TCP Z=.15"
    elif model_id == "dh_pgi_140_80":
        root = "gripper_base_link_1"
        joints = {"gripper_finger1_joint_1": 0.0, "gripper_finger2_joint_1": 0.0}
        # The shipped UR5/PGI embodiment and tutorial grasp TCP name this
        # contact centre explicitly. General/tool-point overrides are distinct.
        canonical = default_tcp = _pose(xyz=(0, 0, 0.15))
        calibration = "ur5_dh_pgi_140_80 embodiment; PGI grasp TCP at base Z=.15"
    elif model_id == "cobotmagic_v100_gripper":
        root = "gripper_base"
        joints = {"joint7": 0.05, "joint8": 0.05}
        default_tcp = _pose(np.diag([-1.0, -1.0, 1.0]), (0, 0, 0.143))
        canonical = default_tcp @ _pose(rz90)
        calibration = "CobotMagicCfg link6 TCP; identity link6-to-gripper_base"
    elif model_id == "pika_gripper":
        root = "GRIPPER_ROOT_LINK"
        joints = {"FINGER1_JOINT": 0.05, "FINGER2_JOINT": 0.05}
        # The asset includes its root adapter: base X approaches, base Y
        # closes. The shipped W1/Pika TCP already aligns canonical X/Z.
        canonical = default_tcp = _pose(rz90.T, (0.00064, 0, 0.16))
        calibration = (
            "dexforce_w1_030_pika embodiment; standalone root adapter included"
        )
    else:
        root = f"{side}_hand_base_link"
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

    if root_from_contact_frame is not None:
        canonical = _rigid_transform(
            root_from_contact_frame,
            error_message="root_from_contact_frame must be a proper rigid transform.",
        )
        calibration += "; explicit root_from_contact_frame override"
    runtime_tcp = default_tcp if root_from_tcp is None else np.asarray(root_from_tcp)
    # This registered archive relocates its URDF to the asset root while
    # retaining ../Collision in three filenames. Bind those exact known paths;
    # the generic loader continues to reject unresolved paths without guessing.
    mesh_paths = _model_mesh_paths(model_id)
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
