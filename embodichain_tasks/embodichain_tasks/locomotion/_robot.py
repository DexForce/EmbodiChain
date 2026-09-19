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

"""Shared robot configuration helpers for locomotion tasks."""

from __future__ import annotations

from typing import Any

from embodichain.lab.sim.cfg import JointDrivePropertiesCfg, RobotCfg

__all__ = ["apply_task_joint_drive_properties", "task_joint_drive_properties"]


def task_joint_drive_properties(
    config: Any,
    defaults: JointDrivePropertiesCfg | None = None,
) -> JointDrivePropertiesCfg | None:
    """Map task joint parameters to exact asset joint names."""
    robot = config.data["robot"]
    joint_names = config.joint_names

    def resolve(name: str) -> float | dict[str, float] | None:
        value = robot.get(name)
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        values = tuple(float(item) for item in value)
        if len(values) != len(joint_names):
            raise ValueError(
                f"Task robot field {name!r} has {len(values)} values for "
                f"{len(joint_names)} policy joints."
            )
        return dict(zip(joint_names, values))

    def resolve_or_default(name: str, field_name: str):
        value = resolve(name)
        if value is not None or defaults is None:
            return value
        return getattr(defaults, field_name)

    stiffness = resolve_or_default("stiffness", "stiffness")
    damping = resolve_or_default("damping", "damping")
    if stiffness is None and damping is None:
        return None
    return JointDrivePropertiesCfg(
        drive_type="force",
        stiffness=stiffness,
        damping=damping,
        max_effort=resolve_or_default("effort_limit", "max_effort"),
        max_velocity=resolve_or_default("velocity_limit", "max_velocity"),
        friction=None if defaults is None else defaults.friction,
        armature=resolve_or_default("armature", "armature"),
    )


def apply_task_joint_drive_properties(robot_cfg: RobotCfg, config: Any) -> None:
    """Apply task-owned joint drives while retaining other asset properties."""
    drive_properties = task_joint_drive_properties(config, robot_cfg.joint_drive_props)
    if drive_properties is None:
        return
    robot_cfg.asset_physics_mode = "overlay"
    robot_cfg.joint_drive_props = drive_properties
