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

"""Preserve unselected preset joints when merging caller drive rules."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import RobotCfg

__all__ = []


def _prepare_joint_drive_overrides(
    cfg: RobotCfg, init_dict: dict[str, Any], joint_names: list[str]
) -> None:
    """Remove model defaults shadowed by explicit caller drive rules."""
    overrides = init_dict.get("joint_drive_props")
    if not isinstance(overrides, dict):
        return
    parts = dict(cfg.control_parts)
    if isinstance(init_dict.get("control_parts"), dict):
        parts.update(init_dict["control_parts"])
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
        defaults = getattr(cfg.joint_drive_props, field)
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
            cfg.joint_drive_props,
            field,
            {name: value for name, value in defaults.items() if name not in shadowed},
        )
