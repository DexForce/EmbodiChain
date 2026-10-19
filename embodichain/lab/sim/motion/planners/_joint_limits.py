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

"""Adapt robot motion caps to shared planner constraints."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__: list[str] = []

_DEFAULT_VELOCITY = 0.2
_DEFAULT_ACCELERATION = 0.5


@lru_cache(maxsize=128)
def _asset_velocity_limits(path: str, modified_ns: int, size: int) -> dict[str, float]:
    """Read positive authored speeds once per assembled URDF revision.

    Zero-valued URDF speeds conventionally represent undeclared gripper limits;
    explicit zero runtime caps retain their separate stationary-joint meaning.
    """
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return {}
    limits = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        try:
            velocity = float(limit.attrib.get("velocity", "nan"))
        except ValueError:
            continue
        if math.isfinite(velocity) and 0 < velocity < 1e9:
            limits[joint.attrib.get("name", "")] = velocity
    return limits


def _fill_asset_velocity_limits(robot: Robot, limits: torch.Tensor) -> torch.Tensor:
    """Fill only absent runtime caps from the assembled robot's authored joints."""
    if bool(torch.isfinite(limits).all()):
        return limits
    path = getattr(getattr(robot, "cfg", None), "fpath", None)
    if not isinstance(path, (str, Path)):
        return limits
    try:
        path = Path(path).expanduser().resolve()
        stat = path.stat()
    except OSError:
        return limits
    authored = _asset_velocity_limits(str(path), stat.st_mtime_ns, stat.st_size)
    caps = torch.tensor(
        [authored.get(name, math.inf) for name in robot.joint_names],
        dtype=limits.dtype,
        device=limits.device,
    )
    return torch.where(torch.isfinite(limits), limits, caps)


def _joint_velocity_limits(robot: Robot) -> torch.Tensor:
    """Read runtime speeds in state order and apply mimic follower caps."""
    limits = (
        robot.get_qvel_limits(joint_ids=list(range(len(robot.joint_names))))
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )
    if limits.ndim != 2 or limits.shape[1] != len(robot.joint_names):
        raise ValueError("Robot velocity limits must have shape (B, robot DOF).")
    limits = torch.where(
        torch.isfinite(limits) & (limits >= 0) & (limits < 1e9), limits, torch.inf
    )
    limits = _fill_asset_velocity_limits(robot, limits)
    mimics = dict(
        zip(
            robot.mimic_ids,
            zip(robot.mimic_parents, robot.mimic_multipliers, strict=True),
            strict=True,
        )
    )
    for child in mimics:
        bound = limits[:, child].clone()
        visited = {child}
        while child in mimics:
            parent, multiplier = mimics[child]
            if not math.isfinite(multiplier) or parent in visited:
                raise ValueError(
                    "Mimic multipliers must be finite and relations acyclic."
                )
            if multiplier == 0:
                break
            bound = bound / abs(multiplier)
            limits[:, parent] = torch.minimum(limits[:, parent], bound)
            visited.add(parent)
            child = parent
    return limits


def _plan_joint_ids(robot: Robot, control_part: str | None, dof: int) -> list[int]:
    """Select an unambiguous mapping from trajectory columns to robot joints."""
    if control_part is None and dof != len(robot.joint_names):
        matches = [
            part
            for part in (robot.control_parts or {})
            if len(robot.get_joint_ids(part)) == dof
        ]
        if len(matches) != 1:
            raise ValueError(
                "Set control_part to map trajectory columns to robot joint limits."
            )
        control_part = matches[0]
    ids = (
        list(range(dof)) if control_part is None else robot.get_joint_ids(control_part)
    )
    if len(ids) != dof:
        raise ValueError("control_part joint count does not match trajectory DOF.")
    return ids


def _merge_limit(
    value: object, caps: torch.Tensor, fallback: float, *, allow_zero: bool = False
) -> torch.Tensor:
    """Intersect requests with robot caps, preserving signed TOPPRA bounds."""
    if value is None:
        return torch.where(torch.isfinite(caps), caps, fallback)
    requested = torch.as_tensor(value, dtype=caps.dtype, device=caps.device)
    if not bool(torch.isfinite(requested).all()):
        raise ValueError("Joint limits must contain finite positive values.")
    if requested.shape == (len(caps), 2):
        lower, upper = requested.unbind(dim=-1)
        if bool(((lower > 0) | (upper < 0) | (lower >= upper)).any()):
            raise ValueError(
                "Signed joint limits must contain zero and have lower < upper."
            )
        return requested.clamp(min=-caps[:, None], max=caps[:, None])
    if requested.ndim == 0:
        requested = requested.expand_as(caps)
    invalid = requested < 0 if allow_zero else requested <= 0
    if requested.shape != caps.shape or bool(invalid.any()):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(
            f"Joint limits must be {qualifier} scalars or per-joint values ({len(caps)},)."
        )
    return torch.minimum(requested, caps)


def _resolve_joint_constraints(
    robot: Robot,
    constraints: Mapping[str, object],
    *,
    dof: int,
    control_part: str | None = None,
    batch_size: int | None = None,
) -> dict[str, object]:
    """Resolve robot speed caps and planner-owned acceleration requests."""
    ids = _plan_joint_ids(robot, control_part, dof)
    limits = _joint_velocity_limits(robot)
    if batch_size is not None and limits.shape[0] < batch_size:
        raise ValueError("Robot velocity limits do not cover the planning batch.")
    resolved = dict(constraints)
    caps = limits[:batch_size, ids].amin(dim=0)
    resolved["velocity"] = _merge_limit(
        constraints.get("velocity"), caps, _DEFAULT_VELOCITY, allow_zero=True
    )
    resolved["acceleration"] = _merge_limit(
        constraints.get("acceleration"),
        torch.full_like(caps, torch.inf),
        _DEFAULT_ACCELERATION,
    )
    return resolved
