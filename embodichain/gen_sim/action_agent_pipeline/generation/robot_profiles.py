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

"""Robot profiles for action-agent config generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_templates import (
    make_dual_franka_panda_robot_config,
    make_dual_ur_dh_pgi_robot_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _DUAL_FRANKA_TABLETOP_CLEARANCE,
    _DUAL_UR10_ARM_COMPONENT_Z,
    _DUAL_UR10_LEGACY_INIT_Z,
    _DUAL_UR10_TABLETOP_CLEARANCE,
    _DUAL_UR5_ARM_COMPONENT_Z,
    _DUAL_UR5_LEGACY_INIT_Z,
    _DUAL_UR5_TABLETOP_CLEARANCE,
)

__all__ = [
    "DEFAULT_ROBOT_PROFILE_ID",
    "RobotProfile",
    "available_robot_profile_choices",
    "available_robot_profile_ids",
    "resolve_robot_profile",
]

DEFAULT_ROBOT_PROFILE_ID = "dual_ur10"
_PI = 3.141592653589793
_ROBOTIQ_ARG2F_140_OPEN_QPOS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_ROBOTIQ_ARG2F_140_CLOSE_QPOS = (0.7, -0.7, 0.7, -0.7, -0.7, 0.7)


@dataclass(frozen=True)
class RobotProfile:
    """Action-agent robot template and runtime metadata."""

    id: str
    display_name: str
    robot_meta_type: str
    robot_config_factory: Callable[[float], dict[str, Any]]
    agent_arm_slots: Mapping[str, Mapping[str, str]]
    gripper_open_state: tuple[float, ...]
    gripper_close_state: tuple[float, ...]
    arm_aim_yaw_offset: Mapping[str, float]
    prompt_description: str
    prompt_slot_description: str
    tabletop_clearance: float = _DUAL_UR5_TABLETOP_CLEARANCE
    arm_component_z: float = _DUAL_UR5_ARM_COMPONENT_Z
    legacy_init_z: float = _DUAL_UR5_LEGACY_INIT_Z
    grasp_runtime_defaults: Mapping[str, float] = field(default_factory=dict)

    def robot_init_z_from_table_top(self, table_top_z: float | None) -> float:
        """Return the robot root z for a generated tabletop scene."""
        if table_top_z is None:
            return float(self.legacy_init_z)
        return round(
            float(table_top_z) + self.tabletop_clearance - self.arm_component_z,
            6,
        )

    def make_robot_config(self, table_top_z: float | None) -> dict[str, Any]:
        """Build a fresh robot config for this profile."""
        return self.robot_config_factory(self.robot_init_z_from_table_top(table_top_z))

    def runtime_extensions(self) -> dict[str, Any]:
        """Return env extension fields needed by the action-agent runtime."""
        extensions: dict[str, Any] = {
            "agent_robot_profile": self.id,
            "agent_arm_slots": _deep_plain_dict(self.agent_arm_slots),
            "gripper_open_state": [float(value) for value in self.gripper_open_state],
            "gripper_close_state": [float(value) for value in self.gripper_close_state],
            "arm_aim_yaw_offset": {
                str(side): float(value)
                for side, value in self.arm_aim_yaw_offset.items()
            },
        }
        if self.grasp_runtime_defaults:
            extensions["agent_grasp_runtime_defaults"] = {
                str(key): float(value)
                for key, value in self.grasp_runtime_defaults.items()
            }
        return extensions

    def prompt_robot_context(self) -> str:
        """Return prompt text that describes the robot and semantic arm slots."""
        return f"{self.prompt_description}\n{self.prompt_slot_description}".strip()

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable profile metadata for generated summaries."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "robot_meta_type": self.robot_meta_type,
            "agent_arm_slots": _deep_plain_dict(self.agent_arm_slots),
        }


def available_robot_profile_ids() -> tuple[str, ...]:
    """Return canonical robot profile IDs supported by action-agent generation."""
    return tuple(_ROBOT_PROFILES)


def available_robot_profile_choices() -> tuple[str, ...]:
    """Return canonical robot profile IDs and CLI-friendly aliases."""
    return tuple(sorted({*_ROBOT_PROFILES, *_ROBOT_PROFILE_ALIASES}))


def resolve_robot_profile(profile: str | RobotProfile | None) -> RobotProfile:
    """Resolve a profile ID, alias, or profile instance."""
    if isinstance(profile, RobotProfile):
        return profile
    key = DEFAULT_ROBOT_PROFILE_ID if profile is None else str(profile).strip().lower()
    key = key.replace("-", "_")
    key = _ROBOT_PROFILE_ALIASES.get(key, key)
    if key not in _ROBOT_PROFILES:
        expected = ", ".join(available_robot_profile_ids())
        raise ValueError(
            f"Unknown robot profile {profile!r}; expected one of: {expected}"
        )
    return _ROBOT_PROFILES[key]


def _deep_plain_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(value))


def _dual_ur_profile(
    *,
    profile_id: str,
    ur_type: str,
    display_name: str,
    tabletop_clearance: float = _DUAL_UR5_TABLETOP_CLEARANCE,
    arm_component_z: float = _DUAL_UR5_ARM_COMPONENT_Z,
    legacy_init_z: float = _DUAL_UR5_LEGACY_INIT_Z,
) -> RobotProfile:
    return RobotProfile(
        id=profile_id,
        display_name=display_name,
        robot_meta_type=display_name.replace(" ", ""),
        robot_config_factory=(
            lambda robot_init_z, ur_type=ur_type: make_dual_ur_dh_pgi_robot_config(
                ur_type=ur_type,
                robot_init_z=robot_init_z,
            )
        ),
        tabletop_clearance=tabletop_clearance,
        arm_component_z=arm_component_z,
        legacy_init_z=legacy_init_z,
        agent_arm_slots={
            "left": {
                "arm": "right_arm",
                "eef": "right_eef",
            },
            "right": {
                "arm": "left_arm",
                "eef": "left_eef",
            },
        },
        gripper_open_state=_ROBOTIQ_ARG2F_140_OPEN_QPOS,
        gripper_close_state=_ROBOTIQ_ARG2F_140_CLOSE_QPOS,
        arm_aim_yaw_offset={
            "left": _PI,
            "right": 0.0,
        },
        grasp_runtime_defaults={
            "max_open_length": 0.115,
            "min_open_length": 0.02,
            "grasp_finger_length": 0.13,
        },
        prompt_description=(
            f"The robot is a {display_name} composite robot with "
            "Robotiq ARG2F-140 large parallel grippers (140 mm stroke)."
        ),
        prompt_slot_description=(
            "- left_arm is the semantic robot-view left slot, mapped to the "
            "physical right_arm control part.\n"
            "- right_arm is the semantic robot-view right slot, mapped to the "
            "physical left_arm control part."
        ),
    )


def _dual_franka_profile() -> RobotProfile:
    return RobotProfile(
        id="dual_franka",
        display_name="Dual Franka Panda",
        robot_meta_type="DualFrankaPanda",
        robot_config_factory=(
            lambda robot_init_z: make_dual_franka_panda_robot_config(
                robot_init_z=robot_init_z,
            )
        ),
        tabletop_clearance=_DUAL_FRANKA_TABLETOP_CLEARANCE,
        agent_arm_slots={
            "left": {
                "arm": "right_arm",
                "eef": "right_eef",
            },
            "right": {
                "arm": "left_arm",
                "eef": "left_eef",
            },
        },
        gripper_open_state=_ROBOTIQ_ARG2F_140_OPEN_QPOS,
        gripper_close_state=_ROBOTIQ_ARG2F_140_CLOSE_QPOS,
        arm_aim_yaw_offset={
            "left": _PI,
            "right": 0.0,
        },
        grasp_runtime_defaults={
            "max_open_length": 0.115,
            "min_open_length": 0.02,
            "grasp_finger_length": 0.13,
        },
        prompt_description=(
            "The robot is a Dual Franka Panda composite robot with Panda "
            "parallel grippers."
        ),
        prompt_slot_description=(
            "- left_arm is the semantic robot-view left slot, mapped to the "
            "physical right_arm control part.\n"
            "- right_arm is the semantic robot-view right slot, mapped to the "
            "physical left_arm control part."
        ),
    )


_ROBOT_PROFILES: dict[str, RobotProfile] = {
    "dual_ur3": _dual_ur_profile(
        profile_id="dual_ur3",
        ur_type="ur3",
        display_name="Dual UR3",
    ),
    "dual_ur5": _dual_ur_profile(
        profile_id="dual_ur5",
        ur_type="ur5",
        display_name="Dual UR5",
    ),
    "dual_ur10": _dual_ur_profile(
        profile_id="dual_ur10",
        ur_type="ur10",
        display_name="Dual UR10",
        tabletop_clearance=_DUAL_UR10_TABLETOP_CLEARANCE,
        arm_component_z=_DUAL_UR10_ARM_COMPONENT_Z,
        legacy_init_z=_DUAL_UR10_LEGACY_INIT_Z,
    ),
    "dual_franka": _dual_franka_profile(),
}

_ROBOT_PROFILE_ALIASES = {
    "ur3": "dual_ur3",
    "dual_ur3_dh_pgi": "dual_ur3",
    "dual_ur3_robotiq": "dual_ur3",
    "dual_ur3_robotiq_arg2f_140": "dual_ur3",
    "ur5": "dual_ur5",
    "dual_ur5_dh_pgi": "dual_ur5",
    "dual_ur5_robotiq": "dual_ur5",
    "dual_ur5_robotiq_arg2f_140": "dual_ur5",
    "ur10": "dual_ur10",
    "dual_ur10_dh_pgi": "dual_ur10",
    "dual_ur10_robotiq": "dual_ur10",
    "dual_ur10_robotiq_arg2f_140": "dual_ur10",
    "franka": "dual_franka",
    "panda": "dual_franka",
    "dual_panda": "dual_franka",
    "dual_franka_panda": "dual_franka",
}
