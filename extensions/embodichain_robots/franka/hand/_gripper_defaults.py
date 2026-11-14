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
        "uid": "FrankaPandaHand",
        "fpath": "Franka/Panda/Fr3HandV1.urdf",
        "control_parts": {"eef": ["fr3_finger_joint1", "fr3_finger_joint2"]},
        "solver_cfg": {},
        "init_qpos": [0.04, 0.04],
        "joint_drive_props": {
            "drive_type": "force",
            "stiffness": 1000.0,
            "damping": 100.0,
            "max_effort": 40.0,
        },
        "joint_commands": {
            "open": {"fr3_finger_joint1": 0.04, "fr3_finger_joint2": 0.04},
            "grasp": {"fr3_finger_joint1": 0.0, "fr3_finger_joint2": 0.0},
        },
        "default_root_from_tcp": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.1034],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "qpos_limits": {
            "fr3_finger_joint1": [0.0, 0.04],
            "fr3_finger_joint2": [0.0, 0.04],
        },
        "mesh_paths": {},
        "source_sha256": "0a227ab180b032e8e268ad618f2e564f7d6c0cb44c13011797dca4ffa1a0e79a",
        "catalog_id": "FrankaPandaHand",
        "model_id": "franka_panda_hand",
        "min_opening_width": 0.001,
        "max_opening_width": 0.08,
        "root_link": "fr3_hand",
        "collision_links": ["fr3_hand", "fr3_leftfinger", "fr3_rightfinger"],
        "pad_links": ["fr3_leftfinger", "fr3_rightfinger"],
        "root_from_contact_frame": [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.1034],
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
