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
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__: list[str] = []

_DEFAULT_VELOCITY = 0.2
_DEFAULT_ACCELERATION = 0.5


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


def _merge_limit(value: object, caps: torch.Tensor, fallback: float) -> torch.Tensor:
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
    if requested.shape != caps.shape or bool((requested <= 0).any()):
        raise ValueError(
            f"Joint limits must be positive scalars or per-joint values ({len(caps)},)."
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
        constraints.get("velocity"), caps, _DEFAULT_VELOCITY
    )
    resolved["acceleration"] = _merge_limit(
        constraints.get("acceleration"),
        torch.full_like(caps, torch.inf),
        _DEFAULT_ACCELERATION,
    )
    return resolved
