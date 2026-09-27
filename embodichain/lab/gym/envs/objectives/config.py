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

"""Strict declarations for independent measured physical objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import MISSING
from pathlib import Path
from typing import Any

import yaml

from embodichain.utils import configclass

__all__ = [
    "GoalRegionCfg",
    "OrderedPlacementObjectiveCfg",
    "decode_objective",
    "load_objective_component",
]


@configclass
class GoalRegionCfg:
    """Inclusive axis-aligned bounds in environment-local metres."""

    name: str = MISSING
    lower: tuple[float, float, float] = MISSING
    upper: tuple[float, float, float] = MISSING


@configclass
class OrderedPlacementObjectiveCfg:
    """Ordered stable-region observations; these do not prove gripper release.

    A region counts after consecutive stable samples span ``hold_seconds`` of
    control time. Completed milestones persist; success also requires current
    stability in the final region. Speeds are measured in m/s and rad/s.
    """

    object_uid: str = MISSING
    regions: list[GoalRegionCfg] = MISSING
    hold_seconds: float = MISSING
    max_linear_speed: float = MISSING
    max_angular_speed: float = MISSING
    type: str = "ordered_stable_regions"


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a mapping with string keys.")
    if set(value) != expected:
        raise ValueError(
            f"{label} fields must be {sorted(expected)}; got {sorted(value)}."
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty string without outer whitespace.")
    return value


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data.")
    if value < 0 or (positive and value == 0):
        raise ValueError(
            f"{label} must be {'positive' if positive else 'nonnegative'}."
        )
    return float(value)


def _bound(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three finite numbers.")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError(f"{label} must contain exactly three finite numbers.")
    return tuple(float(item) for item in value)


def decode_objective(value: object) -> OrderedPlacementObjectiveCfg:
    """Decode and validate the minimal ordered stable-region declaration.

    Args:
        value: Mapping containing all required objective fields.

    Returns:
        Validated declaration with normalized numeric bounds and thresholds.
    """
    fields = {
        "type",
        "object_uid",
        "regions",
        "hold_seconds",
        "max_linear_speed",
        "max_angular_speed",
    }
    data = _fields(value, fields, "objective")
    if data["type"] != "ordered_stable_regions":
        raise ValueError("objective.type must be 'ordered_stable_regions'.")
    raw_regions = data["regions"]
    if not isinstance(raw_regions, (list, tuple)) or not raw_regions:
        raise ValueError("objective.regions must be a nonempty ordered sequence.")
    regions = []
    for raw in raw_regions:
        region = _fields(raw, {"name", "lower", "upper"}, "region")
        lower, upper = _bound(region["lower"], "region.lower"), _bound(
            region["upper"], "region.upper"
        )
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError(
                "region.lower must be strictly below region.upper on every axis."
            )
        regions.append(
            GoalRegionCfg(
                name=_text(region["name"], "region.name"), lower=lower, upper=upper
            )
        )
    cfg = OrderedPlacementObjectiveCfg(
        object_uid=_text(data["object_uid"], "objective.object_uid"),
        regions=regions,
        hold_seconds=_number(data["hold_seconds"], "hold_seconds", positive=True),
        max_linear_speed=_number(data["max_linear_speed"], "max_linear_speed"),
        max_angular_speed=_number(data["max_angular_speed"], "max_angular_speed"),
    )
    cfg.validate()
    return cfg


def load_objective_component(
    selection: object, *, base_dir: Path
) -> OrderedPlacementObjectiveCfg:
    """Load a strict ``objective: {component: path}`` relative to its deployment.

    Args:
        selection: Component selection mapping with exactly one path field.
        base_dir: Directory containing the runnable deployment configuration.

    Returns:
        Validated objective declaration loaded from the selected resource.
    """
    data = _fields(selection, {"component"}, "objective selection")
    path = Path(_text(data["component"], "objective.component")).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    with path.open(encoding="utf-8") as stream:
        return decode_objective(yaml.safe_load(stream))
