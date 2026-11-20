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

"""Fixed-base left and right RP Hand configurations."""

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
    _THUMB_FLEXION,
    _RoboPartyRobotCfg,
)

__all__ = ["RoboPartyRPHandCfg"]


@configclass
class RoboPartyRPHandCfg(_RoboPartyRobotCfg):
    """Configure one six-actuator RP Hand with its five source mimic joints."""

    side: Literal["left", "right"] = "right"
    """Native hand side."""

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        self.side = (init_dict or {}).get("side", self.side)
        if self.side not in ("left", "right"):
            raise ValueError(
                f"Unknown RP Hand side {self.side!r}; choose left or right."
            )
        joints = self._load_joints(
            f"RoboPartyRPHand/rp_hand_{self.side}.urdf", init_dict
        )
        self.uid = f"RoboPartyRPHand{self.side.title()}"
        self.init_pos = [0.0, 0.0, 0.3]
        self.root_props = ArticulationRootPropertiesCfg(
            fixed_base=True,
            min_position_iters=16,
            min_velocity_iters=4,
            newton_mimic_stabilizer=True,
        )
        self.control_parts = {
            "eef": [
                joint.attrib["name"] for joint in joints if joint.find("mimic") is None
            ]
        }
        prefix = "lh" if self.side == "left" else "rh"
        self._set_initial_joint_positions(
            joints, {f"{prefix}_finger12": _THUMB_FLEXION}
        )
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness=_HAND_STIFFNESS,
            damping=_HAND_DAMPING,
            max_effort=_HAND_MAX_EFFORT,
            max_velocity=_HAND_MAX_VELOCITY,
        )
