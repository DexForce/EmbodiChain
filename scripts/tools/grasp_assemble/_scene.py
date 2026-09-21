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

"""Standalone UR5/PGI and assemble-object physics configuration for waypoint replay."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import RigidBodyPhysicsCfg, RobotCfg

__all__ = ["create_robot_cfg", "create_assemble_physics"]


def create_robot_cfg(tcp_z: float) -> RobotCfg:
    """Build the calibrated UR5 and PGI gripper from production robot APIs.

    Args:
        tcp_z: Tool-center-point offset along the gripper's local Z axis, in meters.

    Returns:
        Robot configuration with the harness drive gains, joint limits, and
        initial open-gripper posture.
    """
    from embodichain.lab.sim.robots import URRobotCfg

    hand_joint = "gripper_finger1_joint_1"
    return URRobotCfg.from_dict(
        {
            "robot_type": "ur5",
            "uid": "UR5",
            "urdf_cfg": {
                "components": [
                    {
                        "component_type": "hand",
                        "urdf_path": "DH_PGI_140_80/DH_PGI_140_80.urdf",
                    }
                ],
            },
            "control_parts": {"hand": [hand_joint]},
            "joint_drive_props": {
                "stiffness": {"arm": 5e4, hand_joint: 1e3},
                "damping": {"arm": 5e3, hand_joint: 1e2},
                "max_effort": {"arm": 1e6, hand_joint: 1e4},
            },
            "solver_cfg": {
                "arm": {
                    "tcp": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, tcp_z],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    # Match the URDF limits after native canonicalization.
                    "user_qpos_limits": tuple(
                        (-2.0 * math.pi, 2.0 * math.pi) for _ in range(6)
                    ),
                }
            },
            "init_qpos": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0],
            "init_pos": (0.0, 0.0, 0.0),
        }
    )


def create_assemble_physics() -> RigidBodyPhysicsCfg:
    """Return default contact parameters for a lightweight 0.15 kg rigid object."""
    from embodichain.lab.sim.cfg import (
        DefaultRigidBodyPropertiesCfg,
        MassPropertiesCfg,
        RigidBodyMaterialCfg,
        RigidBodyPhysicsCfg,
    )

    return RigidBodyPhysicsCfg(
        mass_props=MassPropertiesCfg(mass=0.15),
        rigid_props=DefaultRigidBodyPropertiesCfg(
            linear_damping=0.2,
            angular_damping=0.2,
        ),
        material_props=RigidBodyMaterialCfg(
            static_friction=0.99,
            dynamic_friction=0.97,
        ),
    )
