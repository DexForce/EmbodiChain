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

"""TCP calibration poses for the atomic samples' mounted grippers.

Each pose maps the configured arm end link to a tool point. Translation is in
meters, expressed in that end link's local frame. Separate mounted presets let
Franka/PGI calibration change without altering UR5/PGI or UR10/Robotiq.
"""

from __future__ import annotations

import math
from typing import Literal

RobotPreset = Literal["ur5", "franka", "ur10"]
ToolPoint = Literal["default", "grasp", "handover"]

# UR5 + DH_PGI_140_80: the original samples use a general tool point at
# 0.17 m, and a grasp/contact center at 0.15 m for manipulation and handover.
UR5_PGI_TCP = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.17),
    (0.0, 0.0, 0.0, 1.0),
)
UR5_PGI_GRASP_TCP = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.15),
    (0.0, 0.0, 0.0, 1.0),
)

# Franka + DH_PGI_140_80: single-arm poses are relative to fr3_link8.
# Edit this pose to calibrate --robot franka in pickup/place/move_joint.
FRANKA_PGI_TCP = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.15),
    (0.0, 0.0, 0.0, 1.0),
)
# Retain the independently authored dual-arm handover tool point.
FRANKA_PGI_HANDOVER_TCP = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.165),
    (0.0, 0.0, 0.0, 1.0),
)

# UR10 + Robotiq 2F-140: preserve the mounting rotation as well as translation.
UR10_ROBOTIQ_2F_140_TCP = (
    (0.0, -1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.21),
    (0.0, 0.0, 0.0, 1.0),
)


def sample_tool_tcp(
    robot_type: RobotPreset,
    *,
    point: ToolPoint = "default",
    tcp_z: float | None = None,
) -> list[list[float]]:
    """Resolve a mounted gripper's TCP as an independent homogeneous matrix.

    Args:
        robot_type: Existing sample robot/fixture combination.
        point: Sample tool point; grasp and handover preserve their original
            calibrations where they differ from the general tool point.
        tcp_z: Optional explicit local-Z calibration override for Python callers.
            The remaining translation and mounting rotation are preserved.

    Returns:
        A fresh 4x4 end-link-to-tool matrix.

    Raises:
        ValueError: If the robot, tool point, or scalar calibration is invalid.
    """
    if point not in ("default", "grasp", "handover"):
        raise ValueError(f"Unknown sample tool point {point!r}")
    if robot_type == "ur5":
        pose = UR5_PGI_TCP if point == "default" else UR5_PGI_GRASP_TCP
    elif robot_type == "franka":
        pose = FRANKA_PGI_HANDOVER_TCP if point == "handover" else FRANKA_PGI_TCP
    elif robot_type == "ur10":
        pose = UR10_ROBOTIQ_2F_140_TCP
    else:
        raise ValueError(f"Unknown sample robot/fixture {robot_type!r}")
    result = [list(row) for row in pose]
    if tcp_z is not None:
        if type(tcp_z) not in (int, float) or not math.isfinite(tcp_z):
            raise ValueError("tcp_z must be a finite number")
        result[2][3] = float(tcp_z)
    return result


__all__ = [
    "UR5_PGI_TCP",
    "UR5_PGI_GRASP_TCP",
    "FRANKA_PGI_TCP",
    "FRANKA_PGI_HANDOVER_TCP",
    "UR10_ROBOTIQ_2F_140_TCP",
    "sample_tool_tcp",
]
