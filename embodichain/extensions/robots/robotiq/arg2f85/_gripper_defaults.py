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
    "default": {
        "uid": "RobotiqArg2F85",
        "fpath": "Robotiq/robotiq_arg2f_85/robotiq_arg2f_85.urdf",
        "control_parts": {
            "eef": [
                "finger_joint",
                "left_inner_knuckle_joint",
                "left_inner_finger_joint",
                "right_outer_knuckle_joint",
                "right_inner_knuckle_joint",
                "right_inner_finger_joint",
            ]
        },
        "solver_cfg": {},
        "init_qpos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "joint_drive_props": {
            "drive_type": "force",
            "stiffness": 1000.0,
            "damping": 100.0,
            "max_effort": 40.0,
        },
        "joint_commands": {
            "open": {
                "finger_joint": 0.0,
                "left_inner_knuckle_joint": 0.0,
                "left_inner_finger_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
                "right_inner_finger_joint": 0.0,
            },
            "grasp": {
                "finger_joint": 0.8,
                "left_inner_knuckle_joint": 0.8,
                "left_inner_finger_joint": -0.8,
                "right_outer_knuckle_joint": 0.8,
                "right_inner_knuckle_joint": 0.8,
                "right_inner_finger_joint": -0.8,
            },
        },
        "default_root_from_tcp": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.15],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "qpos_limits": {
            "finger_joint": [0.0, 0.8],
            "left_inner_knuckle_joint": [0.0, 0.8757],
            "left_inner_finger_joint": [-0.8, 0.0],
            "right_outer_knuckle_joint": [0.0, 0.81],
            "right_inner_knuckle_joint": [0.0, 0.8757],
            "right_inner_finger_joint": [-0.8, 0.0],
        },
        "limit_overrides": {
            "left_inner_finger_joint": [-0.8, 0.0],
            "right_inner_finger_joint": [-0.8, 0.0],
        },
        "mesh_paths": {},
        "source_sha256": "968003796fa35d9e90607677893eea0cdbda5ac989fdda80d7de551ce0fbe459",
        "catalog_id": "RobotiqArg2F85",
        "model_id": "robotiq_2f85",
        "min_opening_width": 0.001,
        "max_opening_width": 0.085,
        "root_link": "robotiq_arg2f_base_link",
        "collision_links": [
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
        ],
        "pad_links": ["left_inner_finger_pad", "right_inner_finger_pad"],
        "root_from_contact_frame": [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.15],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "grasp_generators": ["antipodal_parallel_jaw"],
    }
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
