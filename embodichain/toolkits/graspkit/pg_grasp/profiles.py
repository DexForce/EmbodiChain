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

"""End-effector-owned geometry and action-independent grasp sampling policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, pi
from typing import Any, Final

from .antipodal_generator import GraspGeneratorCfg
from .antipodal_sampler import AntipodalSamplerCfg
from .gripper_collision_checker import GripperCollisionCfg

__all__ = [
    "AntipodalGraspPolicy",
    "ParallelJawEefProfile",
    "get_parallel_jaw_eef_profile",
    "parallel_jaw_eef_profiles",
]


@dataclass(frozen=True, slots=True)
class ParallelJawEefProfile:
    """Physical identity and calibrated box proxy for one parallel-jaw EEF."""

    profile_id: str
    asset_id: str
    jaw_opening_min: float
    jaw_opening_max: float
    finger_length: float
    x_thickness: float
    y_thickness: float
    root_z_width: float
    open_check_margin: float
    contact_penetration_tolerance: float
    point_sample_dense: float

    def __post_init__(self) -> None:
        for name in ("profile_id", "asset_id"):
            if not isinstance(getattr(self, name), str) or not getattr(
                self, name
            ).strip():
                raise ValueError(f"{name} must be a non-empty string.")
        numeric = (
            "jaw_opening_min",
            "jaw_opening_max",
            "finger_length",
            "x_thickness",
            "y_thickness",
            "root_z_width",
            "open_check_margin",
            "contact_penetration_tolerance",
            "point_sample_dense",
        )
        for name in numeric:
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.jaw_opening_max <= self.jaw_opening_min:
            raise ValueError("jaw_opening_max must exceed jaw_opening_min.")
        for name in (
            "finger_length",
            "x_thickness",
            "y_thickness",
            "root_z_width",
            "point_sample_dense",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")

    def collision_config(
        self,
        *,
        max_decomposition_hulls: int,
    ) -> GripperCollisionCfg:
        """Build the graspkit collision proxy owned by this EEF profile."""
        return GripperCollisionCfg(
            max_open_length=float(self.jaw_opening_max),
            finger_length=float(self.finger_length),
            x_thickness=float(self.x_thickness),
            y_thickness=float(self.y_thickness),
            root_z_width=float(self.root_z_width),
            open_check_margin=float(self.open_check_margin),
            contact_penetration_tolerance=float(
                self.contact_penetration_tolerance
            ),
            point_sample_dense=float(self.point_sample_dense),
            max_decomposition_hulls=int(max_decomposition_hulls),
        )

    def as_mapping(self) -> dict[str, Any]:
        """Return a JSON-compatible profile snapshot."""
        return {
            "profile_id": self.profile_id,
            "asset_id": self.asset_id,
            "jaw_opening_min": float(self.jaw_opening_min),
            "jaw_opening_max": float(self.jaw_opening_max),
            "collision_proxy": {
                "finger_length": float(self.finger_length),
                "x_thickness": float(self.x_thickness),
                "y_thickness": float(self.y_thickness),
                "root_z_width": float(self.root_z_width),
                "open_check_margin": float(self.open_check_margin),
                "contact_penetration_tolerance": float(
                    self.contact_penetration_tolerance
                ),
                "point_sample_dense": float(self.point_sample_dense),
            },
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ParallelJawEefProfile:
        """Build a strict EEF profile from one persisted snapshot."""
        expected = {
            "profile_id",
            "asset_id",
            "jaw_opening_min",
            "jaw_opening_max",
            "collision_proxy",
        }
        if set(value) != expected:
            raise ValueError("End-effector profile fields do not match the schema.")
        collision = value.get("collision_proxy")
        collision_fields = {
            "finger_length",
            "x_thickness",
            "y_thickness",
            "root_z_width",
            "open_check_margin",
            "contact_penetration_tolerance",
            "point_sample_dense",
        }
        if not isinstance(collision, Mapping) or set(collision) != collision_fields:
            raise ValueError(
                "End-effector collision_proxy fields do not match the schema."
            )
        return cls(
            profile_id=str(value["profile_id"]),
            asset_id=str(value["asset_id"]),
            jaw_opening_min=float(value["jaw_opening_min"]),
            jaw_opening_max=float(value["jaw_opening_max"]),
            **{name: float(collision[name]) for name in collision_fields},
        )


@dataclass(frozen=True, slots=True)
class AntipodalGraspPolicy:
    """Algorithm policy resolved against, but not owned by, an EEF profile."""

    n_sample: int = 10000
    max_angle: float = pi / 12
    min_contact_span: float = 0.003
    max_contact_span: float | None = None
    max_deviation_angle: float = pi / 9
    n_deviated_approach_directions: int = 4
    n_top_grasps: int = 50
    viser_port: int = 11801
    max_decomposition_hulls: int = 16
    filter_support_collision: bool = True

    def __post_init__(self) -> None:
        for name in (
            "n_sample",
            "n_deviated_approach_directions",
            "n_top_grasps",
            "viser_port",
            "max_decomposition_hulls",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("max_angle", "min_contact_span", "max_deviation_angle"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.max_contact_span is not None:
            maximum = float(self.max_contact_span)
            if not isfinite(maximum) or maximum <= self.min_contact_span:
                raise ValueError(
                    "max_contact_span must exceed min_contact_span when provided."
                )
        if not isinstance(self.filter_support_collision, bool):
            raise TypeError("filter_support_collision must be a bool.")

    def resolved_opening_range(
        self,
        eef_profile: ParallelJawEefProfile,
    ) -> tuple[float, float]:
        """Intersect contact-span policy with physical EEF opening limits."""
        minimum = max(
            float(self.min_contact_span),
            float(eef_profile.jaw_opening_min),
        )
        maximum = float(eef_profile.jaw_opening_max)
        if self.max_contact_span is not None:
            maximum = min(maximum, float(self.max_contact_span))
        if maximum <= minimum:
            raise ValueError(
                "Resolved contact span is empty for the selected EEF profile."
            )
        return minimum, maximum

    def generator_config(
        self,
        eef_profile: ParallelJawEefProfile,
    ) -> GraspGeneratorCfg:
        """Build a grasp generator configuration for one EEF."""
        minimum, maximum = self.resolved_opening_range(eef_profile)
        return GraspGeneratorCfg(
            viser_port=int(self.viser_port),
            antipodal_sampler_cfg=AntipodalSamplerCfg(
                n_sample=int(self.n_sample),
                max_angle=float(self.max_angle),
                min_length=minimum,
                max_length=maximum,
            ),
            max_deviation_angle=float(self.max_deviation_angle),
            n_deviated_approach_directions=int(
                self.n_deviated_approach_directions
            ),
            n_top_grasps=int(self.n_top_grasps),
            is_partial_annotate=False,
            is_filter_ground_collision=bool(self.filter_support_collision),
        )


_PARALLEL_JAW_EEF_PROFILES: Final[dict[str, ParallelJawEefProfile]] = {
    "robotiq_arg2f_140": ParallelJawEefProfile(
        profile_id="robotiq_arg2f_140",
        asset_id="Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf",
        jaw_opening_min=0.0,
        jaw_opening_max=0.115,
        finger_length=0.13,
        x_thickness=0.01,
        y_thickness=0.03,
        root_z_width=0.08,
        open_check_margin=0.01,
        contact_penetration_tolerance=0.005,
        point_sample_dense=0.012,
    ),
    "dh_pgi_140_80": ParallelJawEefProfile(
        profile_id="dh_pgi_140_80",
        asset_id="DH_PGI_140_80/DH_PGI_140_80.urdf",
        jaw_opening_min=0.0,
        jaw_opening_max=0.1,
        finger_length=0.1,
        x_thickness=0.01,
        y_thickness=0.04,
        root_z_width=0.096,
        open_check_margin=0.03,
        contact_penetration_tolerance=0.0,
        point_sample_dense=0.012,
    ),
}


def parallel_jaw_eef_profiles() -> dict[str, ParallelJawEefProfile]:
    """Return the registered immutable parallel-jaw EEF profiles."""
    return dict(_PARALLEL_JAW_EEF_PROFILES)


def get_parallel_jaw_eef_profile(profile_id: str) -> ParallelJawEefProfile:
    """Resolve one registered EEF profile by stable ID."""
    try:
        return _PARALLEL_JAW_EEF_PROFILES[str(profile_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown parallel-jaw EEF profile {profile_id!r}.") from exc
