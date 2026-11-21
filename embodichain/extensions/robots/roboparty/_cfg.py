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

"""Common defaults for joint-controlled RoboParty presets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Self, cast
import xml.etree.ElementTree as ET

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import RobotCfg, UsdSourceCfg
from embodichain.lab.sim.cfg.source import _asset_file_path, _usd_source_from_value
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


@dataclass(frozen=True)
class _Joint:
    """Source joint fields consumed by the RoboParty presets."""

    name: str
    effort_limit: float | None
    velocity_limit: float | None
    mimic_parent: str | None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@configclass
class _RoboPartyRobotCfg(RobotCfg):
    """Use native articulation joints without a serial-chain IK solver."""

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> Self:
        """Build defaults for the selected asset and merge caller overrides.

        Args:
            init_dict: Preset fields and ordinary RobotCfg overrides.

        Returns:
            The selected robot configuration.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        _prepare_joint_drive_overrides(
            cfg, init_dict, list(cfg.init_qpos_joint_names or ())
        )
        overrides = dict(init_dict)
        # The joint reader already selected and resolved these file fields.
        overrides.pop("fpath", None)
        overrides.pop("source", None)
        if "init_qpos" in overrides and "init_qpos_joint_names" not in overrides:
            cfg.init_qpos_joint_names = None
        return cast(Self, merge_robot_cfg(cfg, overrides))

    def _load_joints(
        self, asset: str, init_dict: dict[str, Any] | None
    ) -> list[_Joint]:
        fields = init_dict or {}
        self.source = deepcopy(_usd_source_from_value(fields.get("source")))
        fpath = fields.get("fpath")
        self.fpath = (
            get_data_path(str(fpath or asset)) if self.source is None else fpath
        )
        self.build_pk_chain = False
        self.solver_cfg = {}

        path = _asset_file_path(self)
        if path.lower().endswith((".usd", ".usda", ".usdc", ".usdz")):
            return _read_usd_joints(path, self.source)
        return _read_urdf_joints(path)

    def _set_initial_joint_positions(
        self, joints: list[_Joint], initial_positions: dict[str, float]
    ) -> None:
        """Name each coordinate so reset can align it to the backend layout."""
        positions = {
            joint.name: initial_positions.get(joint.name, 0.0) for joint in joints
        }
        for joint in joints:
            if joint.mimic_parent is not None:
                positions[joint.name] = (
                    joint.mimic_multiplier * positions[joint.mimic_parent]
                    + joint.mimic_offset
                )
        self.init_qpos = list(positions.values())
        self.init_qpos_joint_names = list(positions)


def _read_urdf_joints(path: str) -> list[_Joint]:
    joints = []
    for element in ET.parse(path).findall("joint"):
        if element.attrib["type"] == "fixed":
            continue
        limit = element.find("limit")
        limits = {} if limit is None else limit.attrib
        mimic = element.find("mimic")
        relation = {} if mimic is None else mimic.attrib
        joints.append(
            _Joint(
                name=element.attrib["name"],
                effort_limit=float(limits.get("effort", "0")),
                velocity_limit=float(limits.get("velocity", "0")),
                mimic_parent=relation.get("joint"),
                mimic_multiplier=float(relation.get("multiplier", "1")),
                mimic_offset=float(relation.get("offset", "0")),
            )
        )
    return joints


def _read_usd_joints(path: str, source: UsdSourceCfg | None) -> list[_Joint]:
    from dexsim.engine import JointType

    from embodichain.lab.sim.spawn.usd import _parse_singleton

    _, articulation = _parse_singleton(
        path, "articulations", "articulation", source=source
    )
    return [
        _Joint(
            name=joint.name,
            effort_limit=joint.effort_limit,
            velocity_limit=joint.velocity_limit,
            mimic_parent=joint.mimic_parent,
            mimic_multiplier=joint.mimic_multiplier,
            mimic_offset=joint.mimic_offset,
        )
        for joint in articulation.joints
        if joint.joint_type != JointType.FIXED
    ]


def _source_drive_limit(
    joints: list[_Joint], field: Literal["effort_limit", "velocity_limit"]
) -> dict[str, float]:
    """Read positive source drive limits for backends with native fallback caps."""
    return {
        joint.name: value
        for joint in joints
        if (value := getattr(joint, field)) is not None and value > 0
    }
