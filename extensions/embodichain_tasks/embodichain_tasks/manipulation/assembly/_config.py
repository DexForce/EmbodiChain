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
from ._defaults import assembly_defaults

_PARAMETERS = assembly_defaults("parameters")

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
    hinge_entity_cfg: SceneEntityCfg | None = _PARAMETERS["hinge_entity_cfg"]
    mating_sensor: str = _PARAMETERS["mating_sensor"]
    grasp_sensor: str = _PARAMETERS["grasp_sensor"]
    contact_points_per_ring: int = _PARAMETERS["contact_points_per_ring"]
    assembly_profile: str = _PARAMETERS["assembly_profile"]
    assembly_manifest: str | None = _PARAMETERS["assembly_manifest"]
    assembly_seed: int | None = _PARAMETERS["assembly_seed"]
    assembly_config: dict[str, Any] = _PARAMETERS["assembly_config"]
    preserve_source_contacts: bool = _PARAMETERS["preserve_source_contacts"]
    file_visuals: bool = _PARAMETERS["file_visuals"]
    require_heterogeneous_assets: bool = _PARAMETERS["require_heterogeneous_assets"]
    require_exact_robot_dynamics: bool = _PARAMETERS["require_exact_robot_dynamics"]
    # Dense gear contacts can exceed small grasp buffers after long training.
    # Defaults cover the official profiles' entire per-world contact budget;
    # smaller development profiles declare their measured capacities explicitly.
    assembly_contact_capacity: int = _PARAMETERS["assembly_contact_capacity"]
    grasp_contact_capacity: int = _PARAMETERS["grasp_contact_capacity"]
    grasp_warmup_steps: int = _PARAMETERS["grasp_warmup_steps"]
    initial_position_range: float = _PARAMETERS["initial_position_range"]
    grasp_position_range: float = _PARAMETERS["grasp_position_range"]
    joint_stiffness: float = _PARAMETERS["joint_stiffness"]
    joint_damping: float = _PARAMETERS["joint_damping"]
    proprioceptive_joint_limits: list[list[float]] | None = _PARAMETERS[
        "proprioceptive_joint_limits"
    ]
    proprioceptive_velocity_scale: list[float] | None = _PARAMETERS[
        "proprioceptive_velocity_scale"
    ]
    gripper_model: str = MISSING
    robot_asset_source: str = MISSING
    grasp_link_names: list[str] = MISSING
    grasp_aperture_coefficients: list[float] = MISSING
    grasp_joint_upper: float = MISSING
    grasp_force_threshold: float = _PARAMETERS["grasp_force_threshold"]
    grasp_max_air_time: float = _PARAMETERS["grasp_max_air_time"]
    penetration_tolerance: float = _PARAMETERS["penetration_tolerance"]
    penetration_clearance_fraction: float = _PARAMETERS[
        "penetration_clearance_fraction"
    ]
    success_hold_steps: int = _PARAMETERS["success_hold_steps"]
    failure_distance: float = _PARAMETERS["failure_distance"]
    grasp_failure_after_steps: int = _PARAMETERS["grasp_failure_after_steps"]
    contact_band: float = _PARAMETERS["contact_band"]
    gear_tilt_tolerance: float = _PARAMETERS["gear_tilt_tolerance"]
    nut_radial_tolerance: float = _PARAMETERS["nut_radial_tolerance"]
    nut_tilt_tolerance: float = _PARAMETERS["nut_tilt_tolerance"]
    nut_capture_pitch_fraction: float = _PARAMETERS["nut_capture_pitch_fraction"]
    nut_min_turn: float = _PARAMETERS["nut_min_turn"]
    nut_min_advance_fraction: float = _PARAMETERS["nut_min_advance_fraction"]
    nut_advance_tolerance_fraction: float = _PARAMETERS[
        "nut_advance_tolerance_fraction"
    ]
    milestone_depth: float = _PARAMETERS["milestone_depth"]

    cylinder_boundary_samples: int = _PARAMETERS["cylinder_boundary_samples"]
    grip_section_fraction: float = _PARAMETERS["grip_section_fraction"]
    grasp_aperture_clearance: float = _PARAMETERS["grasp_aperture_clearance"]
    parallel_grasp_target_upper: float = _PARAMETERS["parallel_grasp_target_upper"]
    parallel_grasp_closure: float = _PARAMETERS["parallel_grasp_closure"]
    four_bar_grasp_closure: float = _PARAMETERS["four_bar_grasp_closure"]
    grip_depth: float = _PARAMETERS["grip_depth"]
    gear_grip_offsets: list[float] = _PARAMETERS["gear_grip_offsets"]

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
        if type(self.file_visuals) is not bool:
            raise ValueError("Assembly file_visuals must be a boolean.")
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
        if (
            type(self.cylinder_boundary_samples) is not int
            or self.cylinder_boundary_samples < 3
        ):
            raise ValueError(
                "Assembly cylinder_boundary_samples must be an integer >= 3."
            )
        if (
            isinstance(self.grip_section_fraction, bool)
            or not 0 < self.grip_section_fraction <= 1
        ):
            raise ValueError("Assembly grip_section_fraction must be in (0, 1].")
        if len(self.gear_grip_offsets) != 2 or not all(
            not isinstance(v, bool) and math.isfinite(v) and v >= 0
            for v in self.gear_grip_offsets
        ):
            raise ValueError(
                "Assembly gear_grip_offsets requires two finite nonnegative values."
            )
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
            "grasp_aperture_clearance",
            "parallel_grasp_target_upper",
            "parallel_grasp_closure",
            "four_bar_grasp_closure",
            "grip_depth",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Assembly {name} must be finite and nonnegative.")


def assembly_cfg(extensions: Mapping[str, Any]) -> AssemblyCfg:
    """Decode the validated assembly namespace."""
    if not isinstance(extensions.get("assembly"), Mapping):
        raise ValueError("Assembly requires an env.extensions.assembly mapping.")
    cfg = AssemblyCfg(**extensions["assembly"])
    cfg.check()
    return cfg
