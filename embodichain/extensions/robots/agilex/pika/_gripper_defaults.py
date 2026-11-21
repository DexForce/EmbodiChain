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
        "uid": "PikaGripper",
        "fpath": "PikaGripper/PikaGripper.urdf",
        "control_parts": {"eef": ["FINGER1_JOINT", "FINGER2_JOINT"]},
        "solver_cfg": {},
        "init_qpos": [0.05, 0.05],
        "joint_drive_props": {
            "drive_type": "force",
            "stiffness": 1000.0,
            "damping": 100.0,
            "max_effort": 40.0,
        },
        "joint_commands": {
            "open": {"FINGER1_JOINT": 0.05, "FINGER2_JOINT": 0.05},
            "grasp": {"FINGER1_JOINT": 0.0, "FINGER2_JOINT": 0.0},
        },
        "default_root_from_tcp": [
            [0.0, 1.0, 0.0, 0.00064],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.16],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "qpos_limits": {"FINGER1_JOINT": [0.0, 0.05], "FINGER2_JOINT": [0.0, 0.05]},
        "mesh_paths": {},
        "source_sha256": "cb99efefd74f26011c56d5afcf13839c13a5374160fecb02ac02ba9fa172d952",
        "catalog_id": "PikaGripper",
        "model_id": "pika_gripper",
        "min_opening_width": 0.001,
        "max_opening_width": 0.094,
        "root_link": "GRIPPER_ROOT_LINK",
        "collision_links": ["GRIPPER_BASE", "FINGER1", "FINGER2"],
        "pad_links": ["FINGER1", "FINGER2"],
        "root_from_contact_frame": [
            [0.0, 1.0, 0.0, 0.00064],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.16],
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
