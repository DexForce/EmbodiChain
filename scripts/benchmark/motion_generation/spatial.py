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

"""Spatial parameters and seeded domain sampling for Atomic Skill cases.

A skill provider declares the spatial quantities its cases expose.  A domain
samples some of them from regions in the robot base frame and freezes each
sample as an ordinary case configuration, so planners, replay, and scoring
stay unchanged.  This module is pure: it never reads a simulator or robot.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch

__all__ = [
    "ARTICULATION_POSITION",
    "OBJECT_POSITION",
    "ROBOT_START_OFFSET",
    "TARGET_OFFSET",
    "CylinderRegion",
    "SpatialDomain",
    "SpatialParameter",
    "UniformRange",
    "parse_spatial_domain",
    "resolve_parameter_sizes",
    "sample_spatial_parameters",
]

SpatialLayer = Literal["scene_layout", "robot_start", "task_goal"]
_LAYERS = ("scene_layout", "robot_start", "task_goal")
_DOMAIN_KEYS = {"samples", "seed", "max_attempts", "parameters"}


@dataclass(frozen=True)
class SpatialParameter:
    """One spatial quantity a skill exposes to domain sampling.

    Interaction geometry (approach direction, pre-grasp and lift distances,
    contact offsets) is deliberately not declared: domains hold it fixed so a
    success-rate change is attributable to where the task is, not how the
    skill is parameterized.

    Attributes:
        name: Stable dotted name used by domain configs, e.g. ``object.position``.
        layer: Scene layout, robot start, or task goal.
        region: ``cylinder`` for a position in the robot base frame, or
            ``uniform`` for a value in the parameter's own unit.
        config_key: Case configuration key that receives the sampled value.
        size: Number of values a ``uniform`` sample holds; ``None`` means one
            per controlled arm joint, resolved by
            :func:`resolve_parameter_sizes`.
        unit: Unit of the sampled value.
    """

    name: str
    layer: SpatialLayer
    region: Literal["cylinder", "uniform"]
    config_key: str
    size: int | None = 1
    unit: str = "m"

    def __post_init__(self) -> None:
        if not self.name or not self.config_key:
            raise ValueError("Spatial parameters need a name and a config key.")
        if self.layer not in _LAYERS:
            raise ValueError(f"Spatial layer must be one of {_LAYERS}.")
        if self.region not in {"cylinder", "uniform"}:
            raise ValueError("Spatial region must be 'cylinder' or 'uniform'.")
        if self.size is not None and (type(self.size) is not int or self.size < 1):
            raise ValueError("Spatial parameter size must be a positive integer.")


OBJECT_POSITION = SpatialParameter(
    "object.position", "scene_layout", "cylinder", "object_position_in_base_m"
)
"""Where the manipulated object starts, before it settles on its support."""

ARTICULATION_POSITION = SpatialParameter(
    "articulation.position",
    "scene_layout",
    "cylinder",
    "articulation_position_in_base_m",
)
"""Where the articulated object's root stands; its orientation is unchanged."""

ROBOT_START_OFFSET = SpatialParameter(
    "robot.start_offset",
    "robot_start",
    "uniform",
    "start_qpos_offset_rad",
    size=None,
    unit="rad",
)
"""Offset added to every arm joint of the nominal start state."""

TARGET_OFFSET = SpatialParameter(
    "target.offset", "task_goal", "uniform", "target_object_offset_m", size=3
)
"""Goal position of the object relative to where it settled, in the arena frame."""


@dataclass(frozen=True)
class CylinderRegion:
    """An annular sector in the robot base frame.

    Positions are uniform over the sector's area: radius from the base z
    axis, azimuth from the base +x axis toward +y.  Without ``height_m`` the
    sample keeps the object's nominal height, so a resting object stays on
    its support.
    """

    radius_m: tuple[float, float]
    azimuth_rad: tuple[float, float]
    height_m: tuple[float, float] | None = None

    def sample(self, unit: torch.Tensor) -> dict[str, Any]:
        """Map three uniform values in ``[0, 1)`` to one base-frame position.

        Args:
            unit: Three independent uniform values.

        Returns:
            Radius, azimuth, optional height, and the base-frame position.
            The position has two entries when the height is kept nominal.
        """
        low, high = self.radius_m
        radius = math.sqrt(low**2 + float(unit[0]) * (high**2 - low**2))
        azimuth = self.azimuth_rad[0] + float(unit[1]) * (
            self.azimuth_rad[1] - self.azimuth_rad[0]
        )
        position = [radius * math.cos(azimuth), radius * math.sin(azimuth)]
        height = None
        if self.height_m is not None:
            height = self.height_m[0] + float(unit[2]) * (
                self.height_m[1] - self.height_m[0]
            )
            position.append(height)
        return {
            "radius_m": radius,
            "azimuth_rad": azimuth,
            "height_m": height,
            "position_in_base_m": position,
        }


@dataclass(frozen=True)
class UniformRange:
    """Independent uniform values in the parameter's own unit."""

    low: tuple[float, ...]
    high: tuple[float, ...]

    def sample(self, unit: torch.Tensor) -> dict[str, Any]:
        """Map uniform values in ``[0, 1)`` to one value per dimension."""
        value = [
            low + float(fraction) * (high - low)
            for low, high, fraction in zip(self.low, self.high, unit, strict=False)
        ]
        return {"value": value}


@dataclass(frozen=True)
class SpatialDomain:
    """Seeded sampling plan that freezes one domain's case population.

    Attributes:
        samples: Feasible cases to keep for every case template.
        seed: Domain seed; sample ``k`` of a template is identical across runs.
        max_attempts: Draws allowed per template before the domain is rejected
            as mostly infeasible.
        parameters: Region for every sampled parameter, by parameter name.
    """

    samples: int
    seed: int
    max_attempts: int
    parameters: Mapping[str, CylinderRegion | UniformRange]


def _pair(value: object, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must be a [low, high] pair.")
    low, high = (float(item) for item in value)
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError(f"{name} must be finite with low <= high.")
    return low, high


def _cylinder(name: str, config: Mapping[str, object]) -> CylinderRegion:
    unknown = set(config) - {"region", "radius_m", "azimuth_deg", "height_m"}
    if unknown:
        raise ValueError(f"{name}: unknown cylinder fields {sorted(unknown)}.")
    radius = _pair(config.get("radius_m"), f"{name}.radius_m")
    if radius[0] < 0.0:
        raise ValueError(f"{name}.radius_m must be non-negative.")
    azimuth = _pair(config.get("azimuth_deg"), f"{name}.azimuth_deg")
    if azimuth[1] - azimuth[0] > 360.0:
        raise ValueError(f"{name}.azimuth_deg must span at most 360 degrees.")
    height = config.get("height_m")
    return CylinderRegion(
        radius_m=radius,
        azimuth_rad=(math.radians(azimuth[0]), math.radians(azimuth[1])),
        height_m=None if height is None else _pair(height, f"{name}.height_m"),
    )


def _uniform(
    name: str, config: Mapping[str, object], parameter: SpatialParameter
) -> UniformRange:
    unknown = set(config) - {"region", "low", "high"}
    if unknown:
        raise ValueError(f"{name}: unknown uniform fields {sorted(unknown)}.")
    bounds = []
    for field in ("low", "high"):
        value = config.get(field)
        values = [value] if isinstance(value, (int, float)) else value
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != parameter.size
        ):
            raise ValueError(f"{name}.{field} must hold {parameter.size} value(s).")
        values = tuple(float(item) for item in values)
        if not all(math.isfinite(item) for item in values):
            raise ValueError(f"{name}.{field} must be finite.")
        bounds.append(values)
    if any(low > high for low, high in zip(*bounds, strict=True)):
        raise ValueError(f"{name}: every low must not exceed its high.")
    return UniformRange(low=bounds[0], high=bounds[1])


def resolve_parameter_sizes(
    declared: Sequence[SpatialParameter], arm_joints: int
) -> tuple[SpatialParameter, ...]:
    """Give per-joint parameters the controlled arm's joint count.

    Args:
        declared: Parameters as a provider declares them.
        arm_joints: Number of joints in the controlled arm.

    Returns:
        Parameters whose sizes are all concrete.
    """
    return tuple(
        replace(parameter, size=arm_joints) if parameter.size is None else parameter
        for parameter in declared
    )


def parse_spatial_domain(
    perturbations: Mapping[str, object],
    declared: Sequence[SpatialParameter],
) -> SpatialDomain | None:
    """Validate a domain's ``perturbations`` against a skill's parameters.

    Args:
        perturbations: The domain's ``perturbations`` mapping.
        declared: Spatial parameters the skill provider exposes.

    Returns:
        The sampling plan, or ``None`` when the domain samples nothing.

    Raises:
        ValueError: Unknown keys, undeclared parameters, or invalid regions.
    """
    if not perturbations:
        return None
    unknown = set(perturbations) - _DOMAIN_KEYS
    if unknown:
        raise ValueError(f"Unknown spatial domain fields {sorted(unknown)}.")
    samples = perturbations.get("samples")
    if type(samples) is not int or samples < 1:
        raise ValueError("Spatial domains need a positive integer 'samples'.")
    seed = perturbations.get("seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("Spatial domain 'seed' must be a non-negative integer.")
    max_attempts = perturbations.get("max_attempts", 4 * samples)
    if type(max_attempts) is not int or max_attempts < samples:
        raise ValueError("Spatial domain 'max_attempts' must be at least 'samples'.")
    raw = perturbations.get("parameters")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("Spatial domains need a non-empty 'parameters' mapping.")
    by_name = {parameter.name: parameter for parameter in declared}
    regions: dict[str, CylinderRegion | UniformRange] = {}
    for name, config in raw.items():
        parameter = by_name.get(str(name))
        if parameter is None:
            raise ValueError(
                f"Spatial parameter {name!r} is not declared by this skill; "
                f"declared: {sorted(by_name)}."
            )
        if not isinstance(config, Mapping):
            raise ValueError(f"Spatial parameter {name!r} needs a region mapping.")
        region = config.get("region", parameter.region)
        if region != parameter.region:
            raise ValueError(
                f"Spatial parameter {name!r} takes a {parameter.region!r} region."
            )
        regions[parameter.name] = (
            _cylinder(parameter.name, config)
            if region == "cylinder"
            else _uniform(parameter.name, config, parameter)
        )
    return SpatialDomain(
        samples=samples, seed=seed, max_attempts=max_attempts, parameters=regions
    )


def _unit_values(seed: int, attempt: int, name: str, size: int) -> torch.Tensor:
    """Draw one parameter's uniform values independently of the others."""
    stream = zlib.crc32(name.encode("utf-8"))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        (int(seed) * 1_000_003 + int(attempt) * 97_409 + stream) % (2**63 - 1)
    )
    return torch.rand(size, generator=generator, dtype=torch.float64)


def sample_spatial_parameters(
    domain: SpatialDomain,
    declared: Sequence[SpatialParameter],
    attempt: int,
) -> tuple[dict[str, object], dict[str, dict[str, Any]]]:
    """Draw one deterministic assignment for a case template.

    Each parameter uses its own random stream, so adding a parameter to a
    domain does not move the values already sampled for the others.

    Args:
        domain: Validated sampling plan.
        declared: The skill's spatial parameters.
        attempt: Draw index within the template, starting at zero.

    Returns:
        Case configuration values by config key, and the sampled quantities by
        parameter name for the case manifest.
    """
    by_name = {parameter.name: parameter for parameter in declared}
    config: dict[str, object] = {}
    record: dict[str, dict[str, Any]] = {}
    for name, region in domain.parameters.items():
        parameter = by_name[name]
        size = 3 if isinstance(region, CylinderRegion) else parameter.size
        sample = region.sample(_unit_values(domain.seed, attempt, name, size))
        value = (
            sample["position_in_base_m"]
            if isinstance(region, CylinderRegion)
            else sample["value"]
        )
        config[parameter.config_key] = (
            value
            if isinstance(region, CylinderRegion) or parameter.size > 1
            else value[0]
        )
        record[name] = {"layer": parameter.layer, "unit": parameter.unit, **sample}
    return config, record
