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

"""Joint-controlled RP1 humanoid configurations."""

from __future__ import annotations

from typing import Any, Literal

from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
)
from embodichain.utils import configclass

from .._cfg import (
    _HAND_DAMPING,
    _HAND_MAX_EFFORT,
    _HAND_MAX_VELOCITY,
    _HAND_STIFFNESS,
    _RoboPartyRobotCfg,
    _source_drive_limit,
)

__all__ = ["RoboPartyRP1Cfg"]

# Match the finger armature in the source MJCF hand class.
_FINGER_ARMATURE = 0.01

_URDF_NAMES = {
    "28dof": "rp1.urdf",
    "24dof": "rp1_24dof.urdf",
    "hand": "rp1_hand.urdf",
    "hand_fixed": "rp1_hand_fixed.urdf",
}


@configclass
class RoboPartyRP1Cfg(_RoboPartyRobotCfg):
    """Configure a floating-base RP1 with the selected upstream joint layout."""

    model: Literal["28dof", "24dof", "hand", "hand_fixed"] = "28dof"
    """Upstream body and hand variant."""

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        self.model = (init_dict or {}).get("model", self.model)
        if self.model not in _URDF_NAMES:
            raise ValueError(
                f"Unknown RP1 model {self.model!r}; choose from {tuple(_URDF_NAMES)}."
            )
        joints = self._load_joints(f"RoboPartyRP1/{_URDF_NAMES[self.model]}", init_dict)
        independent = [joint.name for joint in joints if joint.mimic_parent is None]

        self.uid = "RoboPartyRP1"
        self.init_pos = [0.0, 0.0, 0.78]
        self.root_props = ArticulationRootPropertiesCfg(fixed_base=False)
        max_effort = _source_drive_limit(joints, "effort_limit")
        max_velocity = _source_drive_limit(joints, "velocity_limit")

        if self.model == "hand":
            self.control_parts = {
                "locomotion": [
                    name for name in independent if not name.startswith(("lh_", "rh_"))
                ],
                "left_eef": [name for name in independent if name.startswith("lh_")],
                "right_eef": [name for name in independent if name.startswith("rh_")],
            }
            body = self.control_parts["locomotion"]
            fingers = self.control_parts["left_eef"] + self.control_parts["right_eef"]
            self.joint_drive_props = JointDrivePropertiesCfg(
                drive_type="force",
                stiffness=dict.fromkeys(body, 100.0)
                | dict.fromkeys(fingers, _HAND_STIFFNESS),
                damping=dict.fromkeys(body, 4.0)
                | dict.fromkeys(fingers, _HAND_DAMPING),
                max_effort=dict.fromkeys(fingers, _HAND_MAX_EFFORT) | max_effort,
                max_velocity=dict.fromkeys(fingers, _HAND_MAX_VELOCITY) | max_velocity,
                armature={
                    joint.name: _FINGER_ARMATURE
                    for joint in joints
                    if joint.name.startswith(("lh_", "rh_"))
                },
            )
            self._set_hand_joint_defaults(joints, ("lh_finger12", "rh_finger12"))
        else:
            self.control_parts = {"locomotion": independent}
            self.joint_drive_props = JointDrivePropertiesCfg(
                drive_type="force",
                stiffness=100.0,
                damping=4.0,
                max_effort=max_effort,
                max_velocity=max_velocity,
            )
            self._set_initial_joint_positions(joints, {})
