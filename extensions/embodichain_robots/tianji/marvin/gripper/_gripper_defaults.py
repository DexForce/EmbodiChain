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

"""Pinned native gripper geometry and commands; no asset or backend imports."""

from __future__ import annotations

from copy import deepcopy

__all__ = ["get_gripper_defaults"]

_DEFAULTS = {
    "left": {
        "uid": "TianjiMarvinGripper",
        "fpath": "TianjiMarvin/robot_with_ee_acd.urdf",
        "source_root_link": "left_hand_base_link",
        "extract_subtree": True,
        "reference_link": "left_hand_tool_link",
        "root_from_reference": [
            [7.216449660063518e-16, 0.9999999999999997, -2.6794896412773994e-08, 0.0],
            [-2.6794896412773994e-08, 2.6794896468285145e-08, 0.9999999999999993, 0.0],
            [0.9999999999999997, -5.551115123125783e-17, 2.6794896468285145e-08, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "control_parts": {"eef": ["LEFT_HAND_FINGER_1", "LEFT_HAND_FINGER_2"]},
        "solver_cfg": {},
        "init_qpos": [0.0, 0.0],
        "joint_drive_props": {
            "drive_type": "force",
            "stiffness": 1000.0,
            "damping": 100.0,
            "max_effort": 40.0,
        },
        "grasp_cfg": {
            "catalog_id": "TianjiMarvinGripper",
            "model_id": "tianji_marvin_gripper",
            "min_opening_width": 0.001,
            "max_opening_width": 0.095,
            "root_link": "left_hand_base_link",
            "collision_links": [
                "left_hand_base_link",
                "left_hand_finger_1_link",
                "left_hand_finger_2_link",
            ],
            "pad_links": ["left_hand_finger_1_link", "left_hand_finger_2_link"],
            "root_from_contact_frame": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.13],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "joint_commands": {
            "open": {"LEFT_HAND_FINGER_1": 0.0, "LEFT_HAND_FINGER_2": 0.0},
            "grasp": {"LEFT_HAND_FINGER_1": -0.05, "LEFT_HAND_FINGER_2": -0.05},
        },
        "default_root_from_tcp": [
            [
                2.6794896412773994e-08,
                0.9999999999999997,
                7.216449660063518e-16,
                9.381384558082572e-17,
            ],
            [
                -0.9999999999999993,
                2.6794896468285145e-08,
                -2.6794896412773994e-08,
                -3.4833365336606193e-09,
            ],
            [
                -2.6794896468285145e-08,
                -5.551115123125783e-17,
                0.9999999999999997,
                0.12999999999999995,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "qpos_limits": {
            "LEFT_HAND_FINGER_1": [-0.05, 0.0],
            "LEFT_HAND_FINGER_2": [-0.05, 0.0],
        },
        "mesh_paths": {},
        "source_sha256": "691a5f5a08d3335a1e1a50487b0f886f17eb30da4ea7703bf320a0bf9eac874f",
    },
    "right": {
        "uid": "TianjiMarvinGripper",
        "fpath": "TianjiMarvin/robot_with_ee_acd.urdf",
        "source_root_link": "right_hand_base_link",
        "extract_subtree": True,
        "reference_link": "right_hand_tool_link",
        "root_from_reference": [
            [7.216449660063518e-16, 0.9999999999999997, -2.6794896412773994e-08, 0.0],
            [-2.6794896412773994e-08, 2.6794896468285145e-08, 0.9999999999999993, 0.0],
            [0.9999999999999997, -5.551115123125783e-17, 2.6794896468285145e-08, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "control_parts": {"eef": ["RIGHT_HAND_FINGER_1", "RIGHT_HAND_FINGER_2"]},
        "solver_cfg": {},
        "init_qpos": [0.0, 0.0],
        "joint_drive_props": {
            "drive_type": "force",
            "stiffness": 1000.0,
            "damping": 100.0,
            "max_effort": 40.0,
        },
        "grasp_cfg": {
            "catalog_id": "TianjiMarvinGripper",
            "model_id": "tianji_marvin_gripper",
            "min_opening_width": 0.001,
            "max_opening_width": 0.095,
            "root_link": "right_hand_base_link",
            "collision_links": [
                "right_hand_base_link",
                "right_hand_finger_1_link",
                "right_hand_finger_2_link",
            ],
            "pad_links": ["right_hand_finger_1_link", "right_hand_finger_2_link"],
            "root_from_contact_frame": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.13],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "joint_commands": {
            "open": {"RIGHT_HAND_FINGER_1": 0.0, "RIGHT_HAND_FINGER_2": 0.0},
            "grasp": {"RIGHT_HAND_FINGER_1": -0.05, "RIGHT_HAND_FINGER_2": -0.05},
        },
        "default_root_from_tcp": [
            [
                2.6794896412773994e-08,
                0.9999999999999997,
                7.216449660063518e-16,
                9.381384558082572e-17,
            ],
            [
                -0.9999999999999993,
                2.6794896468285145e-08,
                -2.6794896412773994e-08,
                -3.4833365336606193e-09,
            ],
            [
                -2.6794896468285145e-08,
                -5.551115123125783e-17,
                0.9999999999999997,
                0.12999999999999995,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "qpos_limits": {
            "RIGHT_HAND_FINGER_1": [-0.05, 0.0],
            "RIGHT_HAND_FINGER_2": [-0.05, 0.0],
        },
        "mesh_paths": {},
        "source_sha256": "691a5f5a08d3335a1e1a50487b0f886f17eb30da4ea7703bf320a0bf9eac874f",
    },
}


def get_gripper_defaults(variant: str | None = None) -> dict[str, object]:
    """Return native model data without asset access or runtime imports.

    Args:
        variant: Explicit native variant, or None for a model without variants.

    Returns:
        Independently owned serializable simulation and grasp defaults.

    Raises:
        ValueError: The requested native variant is missing or unsupported.
    """
    key = "default" if variant is None else variant
    if type(key) is not str or key not in _DEFAULTS:
        raise ValueError(
            f"Explicit supported gripper variant required; available: {tuple(_DEFAULTS)}"
        )
    return deepcopy(_DEFAULTS[key])
