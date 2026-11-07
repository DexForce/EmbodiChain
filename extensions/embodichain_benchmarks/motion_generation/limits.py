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

"""Shared benchmark motion budgets for planning and independent validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = [
    "resolve_dynamic_limit",
    "resolve_velocity_limits",
    "resolve_robot_velocity_limits",
]

_PLACEHOLDER_LIMIT_RAD_S = 1.0e3


def resolve_dynamic_limit(
    limit: float | Sequence[float] | None,
    dof: int,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    """Expand a finite positive suite budget to the trajectory joint order."""
    if limit is None:
        return None
    if isinstance(limit, bool):
        raise ValueError("Dynamic limits must be finite positive numbers.")
    values = torch.as_tensor(limit, dtype=reference.dtype, device=reference.device)
    if values.ndim == 0:
        values = values.expand(dof)
    if values.shape != (dof,):
        raise ValueError(
            f"Dynamic limit has shape {tuple(values.shape)} but the trajectory has {dof} joints."
        )
    if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError("Dynamic limits must be finite positive numbers.")
    return values


def resolve_velocity_limits(
    asset_limits: torch.Tensor,
    stated_limit: float | Sequence[float] | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    """Intersect valid asset caps with suite requests, filling placeholders.

    Zero is a stationary-joint cap. Unknown, negative, nonfinite and implausibly
    large asset values require a suite budget; they never loosen a known cap.
    Batched inputs resolve to one conservative vector shared by all rows.
    """
    caps = asset_limits.to(dtype=reference.dtype, device=reference.device)
    if caps.ndim not in (1, 2) or caps.shape[-1] == 0:
        raise ValueError("Asset velocity limits must have shape (DOF,) or (B, DOF).")
    valid = torch.isfinite(caps) & (caps >= 0) & (caps < _PLACEHOLDER_LIMIT_RAD_S)
    caps = torch.where(valid, caps, torch.inf)
    requested = resolve_dynamic_limit(stated_limit, caps.shape[-1], reference)
    if requested is not None:
        caps = torch.minimum(caps, requested)
    if caps.ndim == 2:
        caps = caps.amin(dim=0)
    return caps if bool(torch.isfinite(caps).all()) else None


def resolve_robot_velocity_limits(
    robot: Robot,
    control_part: str,
    stated_limit: float | Sequence[float] | None,
    reference: torch.Tensor,
    *,
    dof: int | None = None,
) -> torch.Tensor | None:
    """Resolve effective robot speeds, including mimic caps and batch minima.

    Real robots expose state-order metadata, allowing follower speed limits to
    constrain their leaders before selecting the requested control-part order.
    Lightweight callers without that metadata may provide already-effective
    control-part limits via ``get_qvel_limits(name=...)``.
    """
    names = getattr(robot, "joint_names", None)
    if isinstance(names, (list, tuple)):
        from embodichain.lab.sim.motion.planners._joint_limits import (
            _joint_velocity_limits,
        )

        ids = list(robot.get_joint_ids(control_part))
        caps = _joint_velocity_limits(robot)[:, ids]
    else:
        caps = robot.get_qvel_limits(name=control_part)
    if not isinstance(caps, torch.Tensor):
        raise ValueError("Robot must provide tensor joint velocity limits.")
    if dof is not None and caps.shape[-1] != dof:
        raise ValueError("control_part joint count does not match trajectory DOF.")
    return resolve_velocity_limits(caps, stated_limit, reference)
