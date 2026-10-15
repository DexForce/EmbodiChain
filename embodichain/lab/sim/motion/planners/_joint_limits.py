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

"""Resolve shared planner limits in simulator joint order."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from embodichain.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__: list[str] = []

_DEFAULT_VELOCITY = 0.2
_DEFAULT_ACCELERATION = 0.5
_UNBOUNDED_VELOCITY = 1e9


@lru_cache(maxsize=32)
def _read_urdf_limits(
    contents: bytes,
) -> tuple[dict[str, float], dict[str, tuple[str, float]]]:
    """Cache parsed limits by content, including same-size rapid asset edits."""
    joints = ET.fromstring(contents).findall("joint")
    limits: dict[str, float] = {}
    mimics: dict[str, tuple[str, float]] = {}
    for joint in joints:
        name = joint.attrib["name"]
        limit = joint.find("limit")
        if limit is not None and "velocity" in limit.attrib:
            value = float(limit.attrib["velocity"])
            if math.isfinite(value) and 0.0 < value < _UNBOUNDED_VELOCITY:
                limits[name] = value
        mimic = joint.find("mimic")
        if mimic is not None:
            mimics[name] = (
                mimic.attrib["joint"],
                abs(float(mimic.get("multiplier", "1"))),
            )
    return limits, mimics


def _propagate_mimic_limits(
    limits: Mapping[str, float], mimics: Mapping[str, tuple[str, float]]
) -> dict[str, float]:
    """Translate follower caps to leaders for velocity and acceleration alike."""
    coupled = dict(limits)
    for name, value in limits.items():
        visited = {name}
        while name in mimics:
            parent, multiplier = mimics[name]
            if not math.isfinite(multiplier):
                raise ValueError("URDF mimic multipliers must be finite.")
            if parent in visited:
                raise ValueError("URDF mimic joints must not contain a cycle.")
            if multiplier == 0.0:
                break
            value /= multiplier
            coupled[parent] = min(coupled.get(parent, math.inf), value)
            visited.add(parent)
            name = parent
    return coupled


def _asset_joint_limits(
    robot: Robot,
    *,
    urdf_path: str | None = None,
) -> tuple[dict[str, float], dict[str, tuple[str, float]]]:
    """Read source URDF metadata; other assets use effective runtime limits."""
    path = urdf_path if urdf_path is not None else getattr(robot.cfg, "fpath", None)
    if not isinstance(path, str) or Path(path).suffix.lower() != ".urdf":
        return {}, {}
    try:
        contents = Path(path).read_bytes()
    except OSError:
        return {}, {}
    return _read_urdf_limits(contents)


def _asset_velocity_limits(
    robot: Robot, *, include_mimic: bool = True, urdf_path: str | None = None
) -> dict[str, float]:
    """Return authored speeds, optionally including coupled follower caps."""
    source, mimics = _asset_joint_limits(robot, urdf_path=urdf_path)
    return _propagate_mimic_limits(source, mimics) if include_mimic else source


def _plan_joint_ids(robot: Robot, control_part: str | None, dof: int) -> list[int]:
    """Require an unambiguous mapping from waypoint columns to robot joints."""
    if control_part is not None:
        ids = robot.get_joint_ids(control_part)
    elif dof == len(robot.joint_names):
        ids = list(range(dof))
    else:
        candidates = [
            robot.get_joint_ids(part)
            for part in (robot.control_parts or {})
            if len(robot.get_joint_ids(part)) == dof
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Set control_part to map trajectory columns to robot joint limits."
            )
        ids = candidates[0]
    if len(ids) != dof:
        raise ValueError("control_part joint count does not match trajectory DOF.")
    return list(ids)


def _acceleration_caps(
    robot: Robot, mimics: Mapping[str, tuple[str, float]]
) -> torch.Tensor:
    """Resolve optional robot metadata, leaving unconfigured joints unbounded."""
    names = list(robot.joint_names)
    caps = torch.full((len(names),), torch.inf, dtype=torch.float64)
    configured = getattr(robot.cfg, "joint_acceleration_limits", None)
    if configured is None:
        return caps
    if isinstance(configured, Mapping):
        rules = {}
        for pattern, value in configured.items():
            part_names = (robot.control_parts or {}).get(pattern)
            if part_names is None:
                rules[pattern] = value
            else:
                expression = (
                    "(?:" + "|".join(re.escape(name) for name in part_names) + ")"
                )
                rules[expression] = value
        ids, _, values = resolve_matching_names_values(rules, names)
        values = torch.as_tensor(values, dtype=caps.dtype)
    else:
        ids = list(range(len(names)))
        values = torch.full_like(caps, float(configured))
    if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError(
            "joint_acceleration_limits must contain finite positive values."
        )
    caps[ids] = values
    coupled = _propagate_mimic_limits(dict(zip(names, caps.tolist())), mimics)
    return torch.tensor([coupled[name] for name in names], dtype=caps.dtype)


def _merge_limit(value: object, caps: torch.Tensor, fallback: float) -> torch.Tensor:
    """Intersect caller limits with robot caps, retaining signed TOPPRA bounds."""
    if value is None:
        return torch.where(torch.isfinite(caps), caps, fallback)
    requested = torch.as_tensor(value, dtype=caps.dtype, device="cpu")
    if requested.ndim == 0:
        requested = requested.expand_as(caps)
    if not bool(torch.isfinite(requested).all()):
        raise ValueError("Joint limits must contain finite positive values.")
    if requested.shape == (len(caps), 2):
        lower, upper = requested.unbind(dim=-1)
        if bool(((lower > 0) | (upper < 0) | (lower >= upper)).any()):
            raise ValueError(
                "Signed joint limits must contain zero and have lower < upper."
            )
        return torch.stack(
            (torch.maximum(lower, -caps), torch.minimum(upper, caps)), -1
        )
    if requested.shape != caps.shape or bool((requested <= 0).any()):
        raise ValueError(
            f"Joint limits must be positive scalars or per-joint values ({len(caps)},)."
        )
    return torch.minimum(requested, caps)


def _resolve_joint_constraints(
    robot: Robot,
    constraints: Mapping[str, object],
    *,
    joint_ids: Sequence[int] | None = None,
    control_part: str | None = None,
    dof: int | None = None,
    batch_size: int | None = None,
    urdf_path: str | None = None,
) -> dict[str, object]:
    """Resolve asset/runtime caps and requests without changing caller options.

    Backends use shared per-joint constraints, so differing runtime limits are
    reduced to the minimum across participating environments. Missing source
    limits use the explicit request or the compatibility fallback. Robot
    acceleration metadata is independent of physics drive properties.
    """
    if joint_ids is None:
        assert dof is not None
        joint_ids = _plan_joint_ids(robot, control_part, dof)
    ids = list(joint_ids)
    if not ids:
        return {
            **constraints,
            "velocity": torch.empty(0),
            "acceleration": torch.empty(0),
        }
    names = list(robot.joint_names)
    effective = (
        robot.get_qvel_limits(joint_ids=list(range(len(names))))
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )
    if effective.ndim != 2 or effective.shape[1] != len(names):
        raise ValueError("Robot velocity limits must have shape (B, robot DOF).")
    if batch_size is not None:
        if effective.shape[0] < batch_size:
            raise ValueError("Robot velocity limits do not cover the planning batch.")
        effective = effective[:batch_size]
    valid = (
        torch.isfinite(effective) & (effective > 0) & (effective < _UNBOUNDED_VELOCITY)
    )
    runtime = torch.where(valid, effective, torch.inf).amin(dim=0)
    source, mimics = _asset_joint_limits(robot, urdf_path=urdf_path)
    runtime = _propagate_mimic_limits(dict(zip(names, runtime.tolist())), mimics)
    velocity = torch.tensor(
        [runtime[names[index]] for index in ids], dtype=torch.float64
    )
    source = _propagate_mimic_limits(source, mimics)
    asset = torch.tensor(
        [source.get(robot.joint_names[index], math.inf) for index in ids],
        dtype=velocity.dtype,
    )
    velocity = torch.minimum(velocity, asset)
    acceleration = _acceleration_caps(robot, mimics)[ids]
    return {
        **constraints,
        "velocity": _merge_limit(
            constraints.get("velocity"), velocity, _DEFAULT_VELOCITY
        ),
        "acceleration": _merge_limit(
            constraints.get("acceleration"), acceleration, _DEFAULT_ACCELERATION
        ),
    }
