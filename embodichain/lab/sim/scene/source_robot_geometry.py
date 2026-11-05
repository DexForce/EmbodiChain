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
"""Read explicit robot source mount geometry without constructing a Robot or IK solver."""

from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["source_fixed_frame_pose", "source_arm_base_position"]


def source_fixed_frame_pose(path: str | Path, frame: str) -> np.ndarray:
    """Resolve a source link through fixed joints only.

    Args:
        path: Authored self-contained robot URDF.
        frame: Exact declared arm-base parent link.

    Returns:
        Source-root to link transform. Moving/cyclic chains reject instead of
        assuming the torso's current state or silently choosing a pose.
    """
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
            raise ValueError("Source mounting requires fixed arm-base attachments.")
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


def source_arm_base_position(
    simulation: Mapping[str, Any], control_part: str
) -> np.ndarray:
    """Locate the mount/shoulder of an explicitly selected arm control group.

    Args:
        simulation: Robot source/assembly configuration owned by the embodiment.
        control_part: Exact motion control-part name; no left/right alias inferred.

    Returns:
        Arm origin relative to the robot source root. Component transforms take
        their original priority; a single-file URDF uses the first declared joint.
    """
    for component in simulation.get("urdf_cfg", {}).get("components", ()):
        if component["component_type"] == control_part:
            return np.asarray(component["transform"], dtype=float)[:3, 3]
    path = Path(simulation["fpath"])
    first_joint = simulation["control_parts"][control_part][0]
    joint = ET.parse(path).getroot().find(f"joint[@name='{first_joint}']")
    if joint is None or joint.get("type") not in {"revolute", "continuous"}:
        raise ValueError("Robot shoulder anchor requires its first declared arm joint.")
    parent = source_fixed_frame_pose(path, joint.find("parent").get("link"))
    origin = joint.find("origin")
    translation = np.asarray(
        [float(value) for value in origin.get("xyz", "0 0 0").split()]
        if origin is not None
        else [0.0, 0.0, 0.0]
    )
    return parent[:3, :3] @ translation + parent[:3, 3]
