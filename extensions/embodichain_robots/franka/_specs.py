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

"""Supported Franka arm/tool combinations, independent of runtime objects."""

from __future__ import annotations

from copy import deepcopy
from embodichain_robots.robotiq.two_finger import _TOOL_SPEC

__all__: list[str] = []

_READY = [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741]
_COMBINATIONS = {
    ("fr3", "panda_hand"): dict(
        uid="FrankaPanda",
        arm_asset="Franka/Panda/PandaWithHand.urdf",
        root_link="base",
        end_link="fr3_hand_tcp",
        solver="fep",
        hand_asset=None,
        arm_joints=tuple(f"fr3_joint{i}" for i in range(1, 8)),
        eef_joints=("fr3_finger_joint1", "fr3_finger_joint2"),
        init_qpos=[*_READY, 0.04, 0.04],
        drive=dict(
            drive_type="force",
            stiffness={"fr3_joint[1-7]": 1e4, "fr3_finger_joint[1-2]": 1e3},
            damping={"fr3_joint[1-7]": 1e3, "fr3_finger_joint[1-2]": 1e2},
            max_effort={"fr3_joint[1-7]": 1e5, "fr3_finger_joint[1-2]": 1e4},
        ),
    ),
    ("panda", "robotiq_2f85"): dict(
        uid="FrankaPandaRobotiq85",
        arm_asset="FrankaPandaArm/Panda/PandaExactV1.urdf",
        root_link="base_link",
        end_link="ee_link",
        solver="pink",
        hand_asset=_TOOL_SPEC["asset"],
        tool=_TOOL_SPEC,
        arm_joints=tuple(f"Joint{i}" for i in range(1, 8)),
        eef_joints=_TOOL_SPEC["joints"],
        init_qpos=[*_READY, *_TOOL_SPEC["open_qpos"]],
        drive=dict(
            drive_type="force",
            stiffness={
                "Joint[1-7]": 1e5,
                ".*finger_joint|.*knuckle_joint": _TOOL_SPEC["drive"]["stiffness"],
            },
            damping={
                "Joint[1-7]": 1e4,
                ".*finger_joint|.*knuckle_joint": _TOOL_SPEC["drive"]["damping"],
            },
            max_effort={
                "Joint[1-4]": 87.0,
                "Joint[5-7]": 12.0,
                ".*finger_joint|.*knuckle_joint": _TOOL_SPEC["drive"]["max_effort"],
            },
            max_velocity={
                "Joint[1-4]": 2.175,
                "Joint[5-7]": 2.61,
                ".*finger_joint|.*knuckle_joint": _TOOL_SPEC["drive"]["max_velocity"],
            },
        ),
    ),
}


def combination(arm_model: str, gripper: str) -> dict:
    """Return an owned supported specification before any asset download."""
    if (arm_model, gripper) not in _COMBINATIONS:
        raise ValueError(
            f"unsupported Franka combination {arm_model!r}/{gripper!r}; supported: fr3/panda_hand, panda/robotiq_2f85"
        )
    return deepcopy(_COMBINATIONS[arm_model, gripper])
