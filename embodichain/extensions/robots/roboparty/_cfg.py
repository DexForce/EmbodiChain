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

"""Common defaults for joint-controlled RoboParty URDF presets."""

from __future__ import annotations

from typing import Any, Self, cast
import xml.etree.ElementTree as ET

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

from .._joint_drives import _prepare_joint_drive_overrides

__all__ = []

# Simulation caps for independently driven finger joints whose source URDF
# specifies zero effort and velocity.
_HAND_MAX_EFFORT = 2.0
_HAND_MAX_VELOCITY = 5.0
_HAND_STIFFNESS = 3.0
_HAND_DAMPING = 0.1

# A small flexion keeps the thumb follower inside its position range despite
# its negative source mimic offset.
_THUMB_FLEXION = 0.02


@configclass
class _RoboPartyRobotCfg(RobotCfg):
    """Use native articulation joints without a serial-chain IK solver."""

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> Self:
        """Build the selected URDF defaults and merge caller overrides.

        Args:
            init_dict: Preset fields and ordinary RobotCfg overrides.

        Returns:
            The selected robot configuration.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        joint_names = [name for names in cfg.control_parts.values() for name in names]
        _prepare_joint_drive_overrides(cfg, init_dict, joint_names)
        return cast(Self, merge_robot_cfg(cfg, init_dict))

    def _load_joints(
        self, asset: str, init_dict: dict[str, Any] | None
    ) -> list[ET.Element]:
        path = (init_dict or {}).get("fpath") or asset
        self.fpath = get_data_path(path)
        self.build_pk_chain = False
        self.solver_cfg = {}

        return [
            joint
            for joint in ET.parse(self.fpath).findall("joint")
            if joint.attrib["type"] != "fixed"
        ]

    def _set_initial_joint_positions(
        self, joints: list[ET.Element], initial_positions: dict[str, float]
    ) -> None:
        """Initialize in source order and satisfy the source mimic constraints."""
        positions = {
            joint.attrib["name"]: initial_positions.get(joint.attrib["name"], 0.0)
            for joint in joints
        }
        for joint in joints:
            mimic = joint.find("mimic")
            if mimic is not None:
                positions[joint.attrib["name"]] = float(
                    mimic.get("multiplier", "1")
                ) * positions[mimic.attrib["joint"]] + float(mimic.get("offset", "0"))
        self.init_qpos = list(positions.values())


def _source_drive_limit(joints: list[ET.Element], field: str) -> dict[str, float]:
    """Read positive URDF drive limits for backends with native fallback caps."""
    limits = {}
    for joint in joints:
        limit = joint.find("limit")
        if limit is not None:
            value = float(limit.get(field, "0"))
            if value > 0:
                limits[joint.attrib["name"]] = value
    return limits
