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

"""Bare seven-joint Franka arm specifications."""

from __future__ import annotations

from copy import deepcopy

__all__: list[str] = []

_READY = [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741]
_ARMS = {
    "fr3": dict(
        asset="Franka/Panda/Panda.urdf",
        root_link="base",
        end_link="fr3_link8",
        solver="fep",
        joints=tuple(f"fr3_joint{i}" for i in range(1, 8)),
        drive=dict(
            drive_type="force",
            stiffness={"fr3_joint[1-7]": 1e4},
            damping={"fr3_joint[1-7]": 1e3},
            max_effort={"fr3_joint[1-7]": 1e5},
        ),
    ),
    "panda": dict(
        asset="FrankaPandaArm/Panda/PandaExactV1.urdf",
        root_link="base_link",
        end_link="ee_link",
        solver="pink",
        joints=tuple(f"Joint{i}" for i in range(1, 8)),
        drive=dict(
            drive_type="force",
            stiffness={"Joint[1-7]": 1e5},
            damping={"Joint[1-7]": 1e4},
            max_effort={"Joint[1-4]": 87.0, "Joint[5-7]": 12.0},
            max_velocity={"Joint[1-4]": 2.175, "Joint[5-7]": 2.61},
        ),
    ),
}


def arm_spec(arm_model: str) -> dict:
    """Return independently owned arm data without resolving any tool."""
    if not isinstance(arm_model, str) or arm_model not in _ARMS:
        raise ValueError(
            f"unsupported Franka arm_model {arm_model!r}; supported: fr3, panda"
        )
    return {**deepcopy(_ARMS[arm_model]), "init_qpos": list(_READY)}
