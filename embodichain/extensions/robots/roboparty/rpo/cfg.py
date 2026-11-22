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

"""Joint-controlled Roboto Origin humanoid configuration."""

from __future__ import annotations

from typing import Any

from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
)
from embodichain.utils import configclass

from .._cfg import _RoboPartyRobotCfg, _source_drive_limit

__all__ = ["RoboPartyRPOCfg"]


@configclass
class RoboPartyRPOCfg(_RoboPartyRobotCfg):
    """Configure the 23-joint RPO humanoid with a floating base and PD drives."""

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        joints = self._load_joints("RoboPartyRPO/rpo.urdf", init_dict)
        self.uid = "RoboPartyRPO"
        self.control_parts = {"locomotion": [joint.name for joint in joints]}
        self.init_pos = [0.0, 0.0, 0.75]
        self._set_initial_joint_positions(joints, {})
        self.root_props = ArticulationRootPropertiesCfg(fixed_base=False)
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness=100.0,
            damping=4.0,
            max_effort=_source_drive_limit(joints, "effort_limit"),
            max_velocity=_source_drive_limit(joints, "velocity_limit"),
        )
