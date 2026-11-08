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
        "uid": "DHPGI14080",
        "fpath": "DH_PGI_140_80/DH_PGI_140_80.urdf",
        "control_parts": {
            "eef": ["gripper_finger1_joint_1", "gripper_finger2_joint_1"]
        },
        "solver_cfg": {},
        "init_qpos": [0.0, 0.0],
        "joint_drive_props": {
            "drive_type": "force",
            "stiffness": 1000.0,
            "damping": 100.0,
            "max_effort": 40.0,
        },
        "joint_commands": {
            "open": {"gripper_finger1_joint_1": 0.0, "gripper_finger2_joint_1": 0.0},
            "grasp": {
                "gripper_finger1_joint_1": 0.024,
                "gripper_finger2_joint_1": 0.024,
            },
        },
        "default_root_from_tcp": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.15],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "qpos_limits": {
            "gripper_finger1_joint_1": [0.0, 0.04],
            "gripper_finger2_joint_1": [0.0, 0.04],
        },
        "mesh_paths": {},
        "source_sha256": "09997a476916af8a9bca51d2cb110bd044275eec6a0707ab84d4bc67c2476b1f",
        "catalog_id": "DHPGI14080",
        "model_id": "dh_pgi_140_80",
        "min_opening_width": 0.005,
        "max_opening_width": 0.1,
        "root_link": "gripper_base_link_1",
        "collision_links": [
            "gripper_base_link_1",
            "gripper_finger1_link_1",
            "gripper_finger2_link_1",
        ],
        "pad_links": ["gripper_finger1_link_1", "gripper_finger2_link_1"],
        "root_from_contact_frame": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
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
