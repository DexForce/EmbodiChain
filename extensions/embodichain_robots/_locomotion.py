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

"""Build ordinary RobotCfg presets from shared locomotion model descriptions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast
import re

import torch

from embodichain.lab.sim.cfg import JointDrivePropertiesCfg, RobotCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

from .locomotion_profiles import load_locomotion_robot_profile

if TYPE_CHECKING:
    import pytorch_kinematics as pk

_CfgT = TypeVar("_CfgT", bound="_LocomotionRobotCfg")


@configclass
class _LocomotionRobotCfg(RobotCfg):
    """Share decoding for floating-base presets without serial-chain solvers."""

    @classmethod
    def from_dict(cls: type[_CfgT], init_dict: dict[str, Any]) -> _CfgT:
        """Build the model defaults, then apply explicit caller overrides.

        Args:
            init_dict: RobotCfg fields overriding the model's defaults.

        Returns:
            An isolated configuration of the selected concrete model.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        cfg._prepare_joint_drive_overrides(init_dict)
        return cast(_CfgT, merge_robot_cfg(cfg, init_dict))

    def _prepare_joint_drive_overrides(self, init_dict: dict[str, Any]) -> None:
        """Remove model rules selected by explicit caller drive rules.

        Preserve caller expressions for the runtime matcher, including its
        direct-rule priority over control parts and ambiguity validation.
        """
        overrides = init_dict.get("joint_drive_props")
        if not isinstance(overrides, dict):
            return
        parts = dict(self.control_parts)
        if isinstance(init_dict.get("control_parts"), dict):
            parts.update(init_dict["control_parts"])
        joint_names = self.control_parts["locomotion"]
        for field in (
            "stiffness",
            "damping",
            "max_effort",
            "max_velocity",
            "friction",
            "armature",
        ):
            provided = overrides.get(field)
            if not isinstance(provided, dict):
                continue
            defaults = getattr(self.joint_drive_props, field)
            if isinstance(defaults, (int, float)):
                defaults = {name: defaults for name in joint_names}
            if not isinstance(defaults, dict):
                continue
            shadowed = {
                name
                for name in joint_names
                if any(
                    re.fullmatch(expression, name)
                    for pattern in provided
                    for expression in parts.get(pattern, [pattern])
                )
            }
            setattr(
                self.joint_drive_props,
                field,
                {
                    name: value
                    for name, value in defaults.items()
                    if name not in shadowed
                },
            )

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Return no serial-chain solvers for the multi-branch locomotion group.

        Args:
            device: Reserved for the shared robot configuration interface.
            **kwargs: Reserved for the shared robot configuration interface.

        Returns:
            An empty mapping. These presets retain ``build_pk_chain=False``;
            adding a limb IK solver requires an explicit limb-chain definition.
        """
        return {}


def _apply_profile(
    cfg: RobotCfg, robot_id: str, init_dict: dict[str, Any] | None
) -> None:
    profile = load_locomotion_robot_profile(robot_id)
    asset = profile["asset"]
    description = profile["robot"]
    joint_names = description["joint_names"]
    asset["uid"] = robot_id
    asset["control_parts"] = {"locomotion": joint_names}
    # A caller-supplied source must not download an unused default asset.
    for source_field in ("fpath", "source"):
        if init_dict and source_field in init_dict:
            asset.pop("fpath", None)
            asset[source_field] = init_dict[source_field]
    defaults = RobotCfg.from_dict(asset)

    def drive_value(name: str, field_name: str) -> float | dict[str, float] | None:
        value = description.get(name)
        if value is None:
            return getattr(defaults.joint_drive_props, field_name)
        if isinstance(value, (int, float)):
            return float(value)
        if len(value) != len(joint_names):
            raise ValueError(
                f"Robot {robot_id!r} field {name!r} has {len(value)} values "
                f"for {len(joint_names)} joints."
            )
        return dict(zip(joint_names, map(float, value)))

    defaults.joint_drive_props = JointDrivePropertiesCfg(
        drive_type="force",
        target_mode=defaults.joint_drive_props.target_mode,
        stiffness=drive_value("stiffness", "stiffness"),
        damping=drive_value("damping", "damping"),
        max_effort=drive_value("effort_limit", "max_effort"),
        max_velocity=drive_value("velocity_limit", "max_velocity"),
        friction=defaults.joint_drive_props.friction,
        armature=drive_value("armature", "armature"),
    )
    for field_name in RobotCfg.__dataclass_fields__:
        setattr(cfg, field_name, getattr(defaults, field_name))
