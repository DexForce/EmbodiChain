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

"""Validated parameters shared by the configured assembly components."""

from __future__ import annotations

from dataclasses import MISSING
import math
from typing import Any
from collections.abc import Mapping

from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.utils import configclass

__all__: list[str] = []


@configclass
class AssemblyCfg:
    """Bind a mating problem and its measurement/control parameters.

    The environment's ``extensions.assembly`` mapping owns this declaration.
    Entity selectors retain the manifest and robot's authored names.
    """

    kind: str = MISSING
    robot_entity_cfg: SceneEntityCfg = MISSING
    moving_entity_cfg: SceneEntityCfg = MISSING
    fixture_entity_cfg: SceneEntityCfg = MISSING
    arm_part: str = MISSING
    gripper_part: str = MISSING
    action_term: str = MISSING
    orientation_representation: str = MISSING
    hinge_entity_cfg: SceneEntityCfg | None = None
    mating_sensor: str = "assembly_contacts"
    grasp_sensor: str = "grasp_contacts"
    contact_points_per_ring: int = 64
    assembly_profile: str = "study"
    assembly_manifest: str | None = None
    assembly_seed: int | None = None
    assembly_config: dict[str, Any] = {}
    preserve_source_contacts: bool = True
    require_heterogeneous_assets: bool = False
    require_exact_robot_dynamics: bool = False
    assembly_contact_capacity: int = 2048
    grasp_contact_capacity: int = 128
    grasp_warmup_steps: int = 10
    initial_position_range: float = 0.0125
    grasp_position_range: float = 0.0025
    joint_stiffness: float = 40.0
    joint_damping: float = 20.0
    proprioceptive_joint_limits: list[list[float]] | None = None
    proprioceptive_velocity_scale: list[float] | None = None
    gripper_model: str = MISSING
    robot_asset_source: str = MISSING
    grasp_link_names: list[str] = MISSING
    grasp_aperture_coefficients: list[float] = MISSING
    grasp_joint_upper: float = MISSING
    grasp_force_threshold: float = 0.05
    grasp_max_air_time: float = 0.005
    penetration_tolerance: float = 0.0001
    penetration_clearance_fraction: float = 0.5
    success_hold_steps: int = 3
    failure_distance: float = 0.25
    grasp_failure_after_steps: int = 25
    contact_band: float = 0.001
    gear_tilt_tolerance: float = 0.05
    nut_radial_tolerance: float = 0.001
    nut_tilt_tolerance: float = 0.15
    nut_capture_pitch_fraction: float = 0.29
    nut_min_turn: float = math.pi / 2
    nut_min_advance_fraction: float = 0.2
    nut_advance_tolerance_fraction: float = 0.1
    milestone_depth: float = 0.001

    def __post_init__(self) -> None:
        for name in (
            "robot_entity_cfg",
            "moving_entity_cfg",
            "fixture_entity_cfg",
            "hinge_entity_cfg",
        ):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, SceneEntityCfg(**value))

    def check(self) -> None:
        """Reject incomplete, ambiguous or physically invalid declarations."""
        self.validate()
        if self.kind not in {"peg", "gear", "nut"}:
            raise ValueError("Assembly kind must be peg, gear or nut.")
        if self.orientation_representation not in {"euler_xyz", "rotation6d"}:
            raise ValueError("Unknown assembly orientation_representation.")
        if (self.kind == "gear") != (self.hinge_entity_cfg is not None):
            raise ValueError("Only gear assembly requires a hinge_entity_cfg.")
        for name in (
            "arm_part",
            "gripper_part",
            "action_term",
            "mating_sensor",
            "grasp_sensor",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"Assembly {name} must be a nonempty name.")
        if (
            self.arm_part == self.gripper_part
            or self.mating_sensor == self.grasp_sensor
        ):
            raise ValueError(
                "Assembly arm/gripper and contact sensors must be distinct."
            )
        if len(self.grasp_link_names) != 2 or len(set(self.grasp_link_names)) != 2:
            raise ValueError("Assembly grasp requires two distinct contact links.")
        if len(self.grasp_aperture_coefficients) != 3 or not all(
            math.isfinite(value) for value in self.grasp_aperture_coefficients
        ):
            raise ValueError("Assembly aperture requires three finite coefficients.")
        for name in (
            "contact_points_per_ring",
            "assembly_contact_capacity",
            "grasp_contact_capacity",
            "success_hold_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"Assembly {name} must be a positive integer.")
        for name in ("grasp_warmup_steps", "grasp_failure_after_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Assembly {name} must be a nonnegative integer.")
        for name in (
            "initial_position_range",
            "grasp_position_range",
            "joint_stiffness",
            "joint_damping",
            "grasp_force_threshold",
            "grasp_max_air_time",
            "penetration_tolerance",
            "penetration_clearance_fraction",
            "failure_distance",
            "contact_band",
            "gear_tilt_tolerance",
            "nut_radial_tolerance",
            "nut_tilt_tolerance",
            "nut_capture_pitch_fraction",
            "nut_min_turn",
            "nut_min_advance_fraction",
            "nut_advance_tolerance_fraction",
            "milestone_depth",
            "grasp_joint_upper",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Assembly {name} must be finite and nonnegative.")


def assembly_cfg(extensions: Mapping[str, Any]) -> AssemblyCfg:
    """Decode the single assembly namespace without legacy field aliases."""
    if not isinstance(extensions.get("assembly"), Mapping):
        raise ValueError("Assembly requires an env.extensions.assembly mapping.")
    cfg = AssemblyCfg(**extensions["assembly"])
    cfg.check()
    return cfg
