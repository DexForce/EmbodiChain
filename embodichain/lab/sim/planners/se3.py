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
"""Manifold-aware Cartesian line-path utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from embodichain.utils.math import (
    axis_angle_from_quat,
    axis_angle_to_rotation_matrix,
    quat_from_matrix,
)

__all__ = ["SE3LineResult", "plan_se3_line"]


def _validate_transform(transform: torch.Tensor, name: str) -> None:
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4).")
    if not transform.is_floating_point() or not bool(
        torch.isfinite(transform).all().item()
    ):
        raise ValueError(f"{name} must be a finite floating-point tensor.")
    tolerance = 1e-6
    expected_row = transform.new_tensor([0.0, 0.0, 0.0, 1.0])
    rotation = transform[:3, :3]
    identity = torch.eye(3, dtype=transform.dtype, device=transform.device)
    if not bool(torch.allclose(transform[3], expected_row, atol=tolerance, rtol=0.0)):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1].")
    if not bool(
        torch.allclose(
            rotation.transpose(0, 1) @ rotation, identity, atol=tolerance, rtol=0.0
        )
    ) or not bool(
        torch.isclose(
            torch.linalg.det(rotation),
            rotation.new_tensor(1.0),
            atol=tolerance,
            rtol=0.0,
        )
    ):
        raise ValueError(f"{name} rotation must belong to SO(3).")


def _skew(vector: torch.Tensor) -> torch.Tensor:
    matrix = vector.new_zeros((3, 3))
    x, y, z = vector.unbind()
    matrix[0, 1], matrix[0, 2] = -z, y
    matrix[1, 0], matrix[1, 2] = z, -x
    matrix[2, 0], matrix[2, 1] = -y, x
    return matrix


def _se3_log(transform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    omega = axis_angle_from_quat(quat_from_matrix(transform[:3, :3]))
    theta = torch.linalg.vector_norm(omega)
    omega_hat = _skew(omega)
    if bool((theta < 1e-8).item()):
        inverse_v = (
            torch.eye(3, dtype=transform.dtype, device=transform.device)
            - 0.5 * omega_hat
            + omega_hat @ omega_hat / 12.0
        )
    else:
        # Half-angle form remains finite at theta=pi, unlike the equivalent
        # (1 + cos(theta)) / sin(theta) expression which becomes 0/0.
        coefficient = (
            1.0 - 0.5 * theta * torch.cos(0.5 * theta) / torch.sin(0.5 * theta)
        ) / theta.square()
        inverse_v = (
            torch.eye(3, dtype=transform.dtype, device=transform.device)
            - 0.5 * omega_hat
            + coefficient * (omega_hat @ omega_hat)
        )
    return inverse_v @ transform[:3, 3], omega


def _se3_exp(
    rho: torch.Tensor, omega: torch.Tensor, parameter: torch.Tensor
) -> torch.Tensor:
    scaled_omega = parameter[..., None] * omega
    rotation = axis_angle_to_rotation_matrix(scaled_omega)
    theta = torch.linalg.vector_norm(scaled_omega, dim=-1)
    omega_hat = _skew(omega)
    eye = torch.eye(3, dtype=rho.dtype, device=rho.device)
    a = torch.where(
        theta.abs() > 1e-8,
        (1.0 - torch.cos(theta)) / theta.square(),
        0.5 - theta.square() / 24.0,
    )
    b = torch.where(
        theta.abs() > 1e-8,
        (theta - torch.sin(theta)) / theta.pow(3),
        1.0 / 6.0 - theta.square() / 120.0,
    )
    scaled_hat = parameter[..., None, None] * omega_hat
    v = (
        eye
        + a[..., None, None] * scaled_hat
        + b[..., None, None] * (scaled_hat @ scaled_hat)
    )
    result = eye.new_zeros(parameter.shape + (4, 4))
    result[..., :3, :3] = rotation
    result[..., :3, 3] = (v @ (parameter[..., None] * rho)[..., None]).squeeze(-1)
    result[..., 3, 3] = 1.0
    return result


@dataclass(frozen=True, slots=True)
class SE3LineResult:
    """Samples from a time-parameterized Cartesian line trajectory."""

    times: torch.Tensor
    poses: torch.Tensor
    velocities: torch.Tensor
    accelerations: torch.Tensor
    jerks: torch.Tensor
    constraint_report: dict[str, torch.Tensor]

    @property
    def duration(self) -> torch.Tensor:
        """Return the scalar trajectory duration."""
        return self.times[-1]


def se3_line_evaluate(
    start: torch.Tensor, end: torch.Tensor, parameter: torch.Tensor
) -> torch.Tensor:
    """Interpolate translation and shortest-geodesic SO(3) rotation.

    Finite parameters are clamped to ``[0, 1]``. ``start`` and ``end`` must be
    unbatched homogeneous transforms with shape ``(4, 4)``.
    """
    _validate_transform(start, "start")
    _validate_transform(end, "end")
    if start.dtype != end.dtype or start.device != end.device:
        raise ValueError("start and end must share dtype and device.")
    parameter = torch.as_tensor(parameter, dtype=start.dtype, device=start.device)
    if not bool(torch.isfinite(parameter).all().item()):
        raise ValueError("parameter must contain only finite values.")
    t = parameter.clamp(0.0, 1.0)
    relative = torch.linalg.inv(start) @ end
    rho, omega = _se3_log(relative)
    return start @ _se3_exp(rho, omega, t)


def se3_line_state(
    start: torch.Tensor,
    end: torch.Tensor,
    parameter: torch.Tensor,
    parameter_velocity: torch.Tensor,
    parameter_acceleration: torch.Tensor,
    parameter_jerk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose an SE3 line pose and 6D derivatives from a scalar time law.

    The derivative convention is ``[linear_xyz, angular_xyz]``. Angular terms
    are expressed in the start/body frame, matching the constant SO(3)
    logarithm used by the geodesic interpolation.
    """
    parameter = torch.as_tensor(parameter, dtype=start.dtype, device=start.device)
    derivatives = [
        torch.as_tensor(value, dtype=start.dtype, device=start.device)
        for value in (
            parameter_velocity,
            parameter_acceleration,
            parameter_jerk,
        )
    ]
    if any(value.shape != parameter.shape for value in derivatives):
        raise ValueError("Scalar time-law inputs must share the parameter shape.")
    if not all(bool(torch.isfinite(value).all().item()) for value in derivatives):
        raise ValueError("Scalar time-law derivatives must be finite.")
    pose = se3_line_evaluate(start, end, parameter)
    rho, rotation = _se3_log(torch.linalg.inv(start) @ end)
    tangent = torch.cat((rho, rotation))
    velocity, acceleration, jerk = (tangent * value[..., None] for value in derivatives)
    return pose, velocity, acceleration, jerk


def plan_se3_line(
    start: torch.Tensor,
    end: torch.Tensor,
    velocity_limit: torch.Tensor,
    acceleration_limit: torch.Tensor,
    jerk_limit: torch.Tensor,
    *,
    profile: Literal["trapezoidal", "double_s"] = "double_s",
    sample_count: int = 100,
    minimum_duration: float | None = None,
) -> SE3LineResult:
    """Plan a constrained time-parameterized Cartesian line trajectory."""
    _validate_transform(start, "start")
    _validate_transform(end, "end")
    if start.dtype != end.dtype or start.device != end.device:
        raise ValueError("start and end must share dtype and device.")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 2
    ):
        raise ValueError("sample_count must be an integer of at least 2.")
    limits = [
        torch.as_tensor(value, dtype=start.dtype, device=start.device)
        for value in (velocity_limit, acceleration_limit, jerk_limit)
    ]
    if any(value.shape != (6,) for value in limits):
        raise ValueError("Cartesian limits must have shape (6,).")
    if any(
        not bool(torch.isfinite(value).all().item()) or bool((value <= 0).any().item())
        for value in limits
    ):
        raise ValueError("Cartesian limits must contain finite positive values.")
    # Lazy import avoids a module cycle: the joint planner imports Bézier
    # geometry, while this public Cartesian helper reuses its scalar profile.
    from .trapezoidal_planner import (
        _apply_minimum_duration,
        _build_scalar_profile,
        _compose_profile_samples_torch,
        _sample_profile_jerk_torch,
    )

    rho, rotation = _se3_log(torch.linalg.inv(start) @ end)
    tangent = torch.cat((rho, rotation))
    active = tangent.abs() > torch.finfo(start.dtype).eps
    if not bool(active.any().item()):
        raise ValueError("start and end must describe distinct poses.")
    scalar_limits = torch.stack(
        [(limit[active] / tangent[active].abs()).amin() for limit in limits]
    )
    scalar_profile = _build_scalar_profile(
        profile_name=profile,
        velocity_limit=scalar_limits[0].reshape(1, 1),
        acceleration_limit=scalar_limits[1].reshape(1, 1),
        jerk_limit=scalar_limits[2].reshape(1, 1),
        backend="torch",
    )
    scalar_profile = _apply_minimum_duration(scalar_profile, minimum_duration)
    duration = scalar_profile.durations.sum(dim=-1)
    times = torch.linspace(
        0.0, duration.item(), sample_count, dtype=start.dtype, device=start.device
    )[None]
    position, velocity, acceleration = _compose_profile_samples_torch(
        times=times,
        cumulative_duration=duration,
        profile=scalar_profile,
        segment_starts=start.new_zeros((1, 1, 1)),
        segment_deltas=start.new_ones((1, 1, 1)),
    )
    scalar_jerk = _sample_profile_jerk_torch(times, duration, scalar_profile)
    poses, velocities, accelerations, jerks = se3_line_state(
        start,
        end,
        position[0, :, 0],
        velocity[0, :, 0],
        acceleration[0, :, 0],
        scalar_jerk[0],
    )
    peak_velocity = velocities.abs().amax(dim=0)
    peak_acceleration = accelerations.abs().amax(dim=0)
    peak_jerk = jerks.abs().amax(dim=0)
    report = {
        "peak_velocity": peak_velocity,
        "peak_acceleration": peak_acceleration,
        "peak_jerk": peak_jerk,
        "velocity_utilization": peak_velocity / limits[0],
        "acceleration_utilization": peak_acceleration / limits[1],
        "jerk_utilization": peak_jerk / limits[2],
        "maximum_utilization": torch.stack(
            (
                (peak_velocity / limits[0]).amax(),
                (peak_acceleration / limits[1]).amax(),
                (peak_jerk / limits[2]).amax(),
            )
        ).amax(),
        "within_limits": torch.stack(
            (
                (peak_velocity <= limits[0] * (1.0 + 1e-9)).all(),
                (peak_acceleration <= limits[1] * (1.0 + 1e-9)).all(),
                (peak_jerk <= limits[2] * (1.0 + 1e-9)).all(),
            )
        ).all(),
    }
    return SE3LineResult(times[0], poses, velocities, accelerations, jerks, report)
