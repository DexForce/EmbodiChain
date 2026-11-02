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

"""Public request-level positive motion-limit resolution in controlled order."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple
import torch

from .planners._joint_limits import _joint_velocity_limits, _plan_joint_ids

__all__ = ["ResolvedMotionLimits", "resolve_motion_limits", "parse_motion_limits"]


class ResolvedMotionLimits(NamedTuple):
    """Requested, robot-effective and resolved limits, all in controlled order."""

    requested_velocity: torch.Tensor
    requested_acceleration: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    velocity_reference: torch.Tensor
    acceleration_reference: torch.Tensor
    robot_velocity: torch.Tensor
    robot_acceleration: torch.Tensor

    @property
    def constraints(self) -> dict[str, torch.Tensor]:
        """Return the exact same tensors for timing and validation."""
        return {"velocity": self.velocity, "acceleration": self.acceleration}

    def metadata(self) -> dict[str, object]:
        """Serialize provenance without implying calibrated acceleration caps."""
        result = {
            name: [
                float(x) if torch.isfinite(x) else None for x in value.detach().cpu()
            ]
            for name, value in self._asdict().items()
        }
        result["acceleration_semantics"] = (
            "planning_budget_intersected_with_reliable_caps_if_provided"
        )
        return result


def _positive_vector(
    value: Any, count: int, name: str, device: Any, dtype: torch.dtype
) -> torch.Tensor:
    contains_bool = isinstance(value, bool) or (
        isinstance(value, (list, tuple)) and any(isinstance(x, bool) for x in value)
    )
    if contains_bool or (isinstance(value, torch.Tensor) and value.dtype == torch.bool):
        raise ValueError(f"{name} must be a positive scalar or controlled-joint vector")
    vector = torch.as_tensor(value, device=device, dtype=dtype).detach().clone()
    if vector.numel() == 1 and vector.ndim <= 1:
        vector = vector.reshape(()).expand(count).clone()
    if vector.shape != (count,) or not bool(
        (torch.isfinite(vector) & (vector > 0)).all()
    ):
        raise ValueError(f"{name} must be finite positive with shape ({count},)")
    return vector


def parse_motion_limits(
    limits: Mapping[str, Any],
    *,
    dof: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Validate a complete positive request without reading a robot.

    Args:
        limits: Positive scalar or controlled-order vector for both fields.
        dof: Controlled column count, at least one.
        device: Output device.
        dtype: Output precision.

    Returns:
        Independent velocity and acceleration vectors. This performs request
        validation only; :func:`resolve_motion_limits` also intersects robot caps.
    """
    if (
        not isinstance(limits, Mapping)
        or set(limits) != {"velocity", "acceleration"}
        or isinstance(dof, bool)
        or not isinstance(dof, int)
        or dof < 1
    ):
        raise ValueError("A motion request requires velocity and acceleration only")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Motion limits require float32 or float64 precision")
    return {
        name: _positive_vector(limits[name], dof, name, device, dtype)
        for name in ("velocity", "acceleration")
    }


def resolve_motion_limits(
    robot: Any,
    limits: Mapping[str, Any],
    *,
    dof: int,
    control_part: str | None = None,
    batch_size: int | None = None,
    robot_acceleration_limits: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ResolvedMotionLimits:
    """Resolve a complete positive request against effective robot speed caps.

    Args:
        robot: Runtime robot owning asset and mimic-aware effective limits.
        limits: Both ``velocity`` and ``acceleration``, positive scalars or J-vectors.
        dof: Controlled column count.
        control_part: Explicit robot control group defining column order.
        batch_size: Scene batch whose most restrictive robot caps are shared.
        robot_acceleration_limits: Optional reliable controlled-order hardware caps.
        device: Output device.
        dtype: Output floating precision.

    Returns:
        Immutable record retaining requested/effective values and fixed robot
        normalization references. Missing acceleration caps mean a planning
        budget, never an assertion of calibrated actuator capability. Models
        freeze and serialize their reference scales; this function describes
        only the selected robot batch and does not own model normalization.
    """
    request = parse_motion_limits(limits, dof=dof, device=device, dtype=dtype)
    velocity, acceleration = request["velocity"], request["acceleration"]
    ids = _plan_joint_ids(robot, control_part, dof)
    speeds = _joint_velocity_limits(robot, device=device, dtype=dtype)
    if batch_size is not None and (batch_size < 1 or batch_size > len(speeds)):
        raise ValueError("Robot caps do not cover the requested scene batch")
    caps = speeds[:batch_size, ids].amin(0)
    if bool((caps <= 0).any()):
        raise ValueError(
            "Frozen controlled joints are outside the positive-limit contract"
        )
    acaps = torch.full_like(caps, torch.inf)
    if robot_acceleration_limits is not None:
        acaps = _positive_vector(
            robot_acceleration_limits, dof, "robot acceleration", device, dtype
        )
    return ResolvedMotionLimits(
        velocity,
        acceleration,
        torch.minimum(velocity, caps),
        torch.minimum(acceleration, acaps),
        torch.where(torch.isfinite(caps), caps, 1.0),
        torch.where(torch.isfinite(acaps), acaps, 1.0),
        caps,
        acaps,
    )
