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
        "uid": "CobotMagicV100Gripper",
        "fpath": "CobotMagicArm/CobotMagicWithGripperV100.urdf",
        "extract_subtree": True,
        "reference_link": "link6",
        "root_from_reference": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "control_parts": {"eef": ["joint7", "joint8"]},
        "solver_cfg": {},
        "init_qpos": [0.05, 0.05],
        "joint_drive_props": {
            "drive_type": "force",
            "stiffness": 1000.0,
            "damping": 100.0,
            "max_effort": 40.0,
        },
        "joint_commands": {
            "open": {"joint7": 0.05, "joint8": 0.05},
            "grasp": {"joint7": 0.0, "joint8": 0.0},
        },
        "default_root_from_tcp": [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.143],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "qpos_limits": {"joint7": [0.0, 0.05], "joint8": [0.0, 0.05]},
        "mesh_paths": {
            "../Visual/gripper_base_v100.dae": "CobotMagicArm/Visual/gripper_base_v100.dae",
            "../Visual/link7.dae": "CobotMagicArm/Visual/link7.dae",
            "../Visual/link8.dae": "CobotMagicArm/Visual/link8.dae",
            "../Collision/gripper_base_v100_acd.obj": "CobotMagicArm/Collision/gripper_base_v100_acd.obj",
            "../Collision/link7_acd.obj": "CobotMagicArm/Collision/link7_acd.obj",
            "../Collision/link8_acd.obj": "CobotMagicArm/Collision/link8_acd.obj",
        },
        "source_sha256": "4340728998d271c3fbc677016baeac1bea014888e1c08fe40e6969b27b61dee6",
        "catalog_id": "CobotMagicV100Gripper",
        "model_id": "cobotmagic_v100_gripper",
        "min_opening_width": 0.001,
        "max_opening_width": 0.1,
        "root_link": "gripper_base",
        "collision_links": ["gripper_base", "link7", "link8"],
        "pad_links": ["link7", "link8"],
        "root_from_contact_frame": [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.143],
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
