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

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

__all__ = [
    "make_dual_franka_panda_robot_config",
    "make_dual_franka_v3_robot_config",
    "make_dual_ur_dh_pgi_robot_config",
    "make_dual_ur5_robot_config",
    "make_light_config",
    "make_sensor_config",
]

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_UR_URDF_DIRS = {
    "ur3": "UR3",
    "ur3e": "UR3e",
    "ur5": "UR5",
    "ur5e": "UR5e",
    "ur10": "UR10",
    "ur10e": "UR10e",
}
_UR_MAX_EFFORT = {
    "ur3": 56.0,
    "ur3e": 56.0,
    "ur5": 100000.0,
    "ur5e": 150.0,
    "ur10": 330.0,
    "ur10e": 330.0,
}


def make_dual_ur5_robot_config(*, robot_init_z: float) -> dict[str, Any]:
    """Return a fresh DualUR5 robot config template at the requested z position."""
    return make_dual_ur_dh_pgi_robot_config(ur_type="ur5", robot_init_z=robot_init_z)


def make_dual_ur_dh_pgi_robot_config(
    *,
    ur_type: str,
    robot_init_z: float,
) -> dict[str, Any]:
    """Return a fresh dual-UR + Robotiq gripper config for the requested UR arm."""
    ur_type = str(ur_type).lower()
    if ur_type not in _UR_URDF_DIRS:
        raise ValueError(
            f"Unsupported dual-UR action-agent profile arm {ur_type!r}; expected "
            f"one of {sorted(_UR_URDF_DIRS)}."
        )

    config = _load_template("dual_ur5_robot.json")
    urdf_dir = _UR_URDF_DIRS[ur_type]
    arm_urdf_path = f"UniversalRobots/{urdf_dir}/{urdf_dir}.urdf"
    display = ur_type.upper().replace("E", "e")

    config["uid"] = f"Dual{display}"
    config["urdf_cfg"]["fname"] = f"dual_{ur_type}_robotiq_arg2f_140_basket"
    config["init_pos"][2] = float(robot_init_z)

    for component in config["urdf_cfg"]["components"]:
        if str(component.get("component_type", "")).endswith("_arm"):
            component["urdf_path"] = arm_urdf_path

    for arm_name in ("left_arm", "right_arm"):
        solver_cfg = config["solver_cfg"][arm_name]
        solver_cfg["ur_type"] = ur_type
        solver_cfg["urdf_path"] = None
        config["drive_pros"]["max_effort"][arm_name] = _UR_MAX_EFFORT[ur_type]

    return config


def make_dual_franka_panda_robot_config(*, robot_init_z: float) -> dict[str, Any]:
    """Return a fresh Dual Franka Panda config at the requested z position."""
    return {
        "uid": "DualFrankaPanda",
        "urdf_cfg": {
            "fname": "dual_franka_panda_basket",
            "name_case": {
                "joint": "original",
                "link": "original",
            },
            "components": [
                {
                    "component_type": "left_arm",
                    "urdf_path": "Franka/Panda/PandaWithHandFake.urdf",
                    "transform": [
                        [0.0, -1.0, 0.0, 0.3],
                        [1.0, 0.0, 0.0, -1.25],
                        [0.0, 0.0, 1.0, 0.4],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                {
                    "component_type": "right_arm",
                    "urdf_path": "Franka/Panda/PandaWithHandFake.urdf",
                    "transform": [
                        [0.0, -1.0, 0.0, -0.3],
                        [1.0, 0.0, 0.0, -1.25],
                        [0.0, 0.0, 1.0, 0.4],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            ],
        },
        "init_pos": [0.7, 0.0, float(robot_init_z)],
        "init_rot": [0.0, 0.0, -90.0],
        "init_qpos": [
            0.0,  # Joint1
            0.0,  # Joint1
            -0.569,  # Joint2
            -0.569,  # Joint2
            0.0,  # Joint3
            0.0,  # Joint3
            -2.810,  # Joint4
            -2.810,  # Joint4
            0.0,  # Joint5
            0.0,  # Joint5
            3.037,  # Joint6
            3.037,  # Joint6
            0.741,  # Joint7
            0.741,  # Joint7
            0.06,  # finger_joint1
            0.06,  # finger_joint1
            0.06,  # finger_joint2
            0.06,  # finger_joint2
        ],
        "drive_pros": {
            "stiffness": {
                "left_arm": 10000.0,
                "right_arm": 10000.0,
                "left_eef": 100.0,
                "right_eef": 100.0,
            },
            "damping": {
                "left_arm": 1000.0,
                "right_arm": 1000.0,
                "left_eef": 10.0,
                "right_eef": 10.0,
            },
            "max_effort": {
                "left_arm": 1000.0,
                "right_arm": 1000.0,
                "left_eef": 100.0,
                "right_eef": 100.0,
            },
        },
        "control_parts": {
            "left_arm": [
                "left_Joint1",
                "left_Joint2",
                "left_Joint3",
                "left_Joint4",
                "left_Joint5",
                "left_Joint6",
                "left_Joint7",
            ],
            "left_eef": ["left_finger_joint[1-2]"],
            "right_arm": [
                "right_Joint1",
                "right_Joint2",
                "right_Joint3",
                "right_Joint4",
                "right_Joint5",
                "right_Joint6",
                "right_Joint7",
            ],
            "right_eef": ["right_finger_joint[1-2]"],
            "dual_arm": [
                "left_Joint1",
                "left_Joint2",
                "left_Joint3",
                "left_Joint4",
                "left_Joint5",
                "left_Joint6",
                "left_Joint7",
                "right_Joint1",
                "right_Joint2",
                "right_Joint3",
                "right_Joint4",
                "right_Joint5",
                "right_Joint6",
                "right_Joint7",
            ],
        },
        "observation_joint_parts": ["left_eef", "right_eef"],
        "qpos_control_part_order": [
            "left_arm",
            "left_eef",
            "right_arm",
            "right_eef",
        ],
        "solver_cfg": {
            "left_arm": {
                "class_type": "PinocchioSolver",
                "urdf_path": None,
                "end_link_name": "left_ee_link",
                "root_link_name": "left_base_link",
                "tcp": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "right_arm": {
                "class_type": "PinocchioSolver",
                "urdf_path": None,
                "end_link_name": "right_ee_link",
                "root_link_name": "right_base_link",
                "tcp": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        },
    }


def make_dual_franka_v3_robot_config(*, robot_init_z: float) -> dict[str, Any]:
    """Return a fresh Dual Franka V3/FR3 config at the requested z position."""
    config = make_dual_franka_panda_robot_config(robot_init_z=robot_init_z)
    config["uid"] = "DualFrankaV3"
    config["urdf_cfg"]["fname"] = "dual_franka_v3_basket"
    config["control_parts"] = {
        "left_arm": [
            "left_fr3_joint1",
            "left_fr3_joint2",
            "left_fr3_joint3",
            "left_fr3_joint4",
            "left_fr3_joint5",
            "left_fr3_joint6",
            "left_fr3_joint7",
        ],
        "left_eef": ["left_fr3_finger_joint[1-2]"],
        "right_arm": [
            "right_fr3_joint1",
            "right_fr3_joint2",
            "right_fr3_joint3",
            "right_fr3_joint4",
            "right_fr3_joint5",
            "right_fr3_joint6",
            "right_fr3_joint7",
        ],
        "right_eef": ["right_fr3_finger_joint[1-2]"],
        "dual_arm": [
            "left_fr3_joint1",
            "left_fr3_joint2",
            "left_fr3_joint3",
            "left_fr3_joint4",
            "left_fr3_joint5",
            "left_fr3_joint6",
            "left_fr3_joint7",
            "right_fr3_joint1",
            "right_fr3_joint2",
            "right_fr3_joint3",
            "right_fr3_joint4",
            "right_fr3_joint5",
            "right_fr3_joint6",
            "right_fr3_joint7",
        ],
    }
    config["solver_cfg"]["left_arm"]["class_type"] = "PytorchSolver"
    config["solver_cfg"]["left_arm"]["end_link_name"] = "left_fr3_hand_tcp"
    config["solver_cfg"]["left_arm"]["root_link_name"] = "left_base"
    config["solver_cfg"]["right_arm"]["class_type"] = "PytorchSolver"
    config["solver_cfg"]["right_arm"]["end_link_name"] = "right_fr3_hand_tcp"
    config["solver_cfg"]["right_arm"]["root_link_name"] = "right_base"
    return config


def make_sensor_config() -> list[dict[str, Any]]:
    """Return a fresh default sensor config template."""
    return _load_template("default_sensors.json")


def make_light_config() -> dict[str, Any]:
    """Return a fresh default light config template."""
    return _load_template("default_lights.json")


def _load_template(name: str) -> Any:
    return deepcopy(_read_template(name))


@lru_cache(maxsize=None)
def _read_template(name: str) -> Any:
    path = _TEMPLATE_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))
