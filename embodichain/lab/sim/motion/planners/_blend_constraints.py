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
"""Continuous derivative bounds for quintic paths under scalar timing."""

from __future__ import annotations

import torch

from ._scalar_time_law import ScalarTimeLaw

__all__ = ["bound_path_derivatives", "bound_blended_derivatives"]

# Subdivision tightens convex-hull bounds; correctness does not depend on
# hitting a peak or on either subdivision count, even for very short blends.
_SUBDIVISIONS = 16


def _split_controls(
    controls: torch.Tensor, parameter: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split batched Bernstein polynomials by de Casteljau subdivision."""
    left, right = [controls[:, 0]], [controls[:, -1]]
    while controls.shape[1] > 1:
        controls = torch.lerp(controls[:, :-1], controls[:, 1:], parameter)
        left.append(controls[:, 0])
        right.append(controls[:, -1])
    return torch.stack(left, dim=1), torch.stack(right[::-1], dim=1)


def _polynomial_bounds(controls: torch.Tensor) -> torch.Tensor:
    """Bound absolute polynomial values on every equal parameter interval."""
    upper = (
        torch.arange(1, _SUBDIVISIONS + 1, dtype=controls.dtype, device=controls.device)
        / _SUBDIVISIONS
    )
    left, _ = _split_controls(
        controls[None].expand(_SUBDIVISIONS, -1, -1), upper[:, None, None]
    )
    _, restricted = _split_controls(
        left, ((upper - 1.0 / _SUBDIVISIONS) / upper)[:, None, None]
    )
    # A Bernstein polynomial stays inside its control-point convex hull.
    return restricted.abs().amax(dim=1)


def bound_path_derivatives(
    segments: list[torch.Tensor],
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Bound three path derivatives on each segment's parameter intervals.

    Returns float64 bounds shaped (segments, 3, subdivisions, D). Differencing
    the original controls before subdivision avoids cancellation between nearby
    subdivided positions and lets projection and time scaling reuse one result.
    """
    segment_bounds = []
    for controls, segment_length in zip(segments, lengths.double()):
        derivative = controls.double()
        bounds = []
        for _ in range(3):
            degree = derivative.shape[0] - 1
            if degree:
                derivative = degree * torch.diff(derivative, dim=0) / segment_length
                bounds.append(_polynomial_bounds(derivative))
            else:
                derivative = torch.zeros_like(derivative)
                bounds.append(derivative.expand(_SUBDIVISIONS, -1))
        segment_bounds.append(torch.stack(bounds))
    return torch.stack(segment_bounds)


def bound_blended_derivatives(
    lengths: torch.Tensor,
    path_bounds: torch.Tensor,
    profile: ScalarTimeLaw,
    row: int,
) -> torch.Tensor:
    """Bound joint velocity, acceleration and jerk over a complete path.

    The result has shape (3, D). Each constant-jerk phase is subdivided into
    time intervals with exact scalar derivative extrema. Each geometric
    segment is subdivided independently, and its derivative control points
    bound the whole interval. Only overlapping time/path intervals contribute.
    The third-order chain rule and triangle inequality then bound every time,
    including either side of a geometric or time-phase boundary.

    Internal float64 arithmetic and a final margin limit roundoff from short
    geometric segments. These are polynomial bounds, not sampled peak estimates.
    Trapezoidal acceleration jumps have no finite jerk bound; callers must
    enforce only velocity and acceleration for that profile.
    """
    length = lengths.double().sum()
    phase_duration = profile.durations[row, 0].double()[:, None]
    fractions = torch.linspace(
        0.0, 1.0, _SUBDIVISIONS + 1, dtype=torch.float64, device=lengths.device
    )
    lower = phase_duration * fractions[:-1]
    upper = phase_duration * fractions[1:]
    p0, v0, a0, jerk = (
        value[row, 0].double()[:, None]
        for value in (
            profile.positions,
            profile.velocities,
            profile.accelerations,
            profile.jerks,
        )
    )

    def position(time: torch.Tensor) -> torch.Tensor:
        return p0 + v0 * time + 0.5 * a0 * time.square() + jerk * time.pow(3) / 6.0

    def velocity(time: torch.Tensor) -> torch.Tensor:
        return v0 + a0 * time + 0.5 * jerk * time.square()

    critical = -a0 / torch.where(jerk != 0.0, jerk, torch.ones_like(jerk))
    critical = critical.clamp(min=lower, max=upper)
    velocity_bound = torch.stack(
        (velocity(lower).abs(), velocity(upper).abs(), velocity(critical).abs())
    ).amax(dim=0)
    acceleration_bound = torch.maximum(
        (a0 + jerk * lower).abs(), (a0 + jerk * upper).abs()
    )
    active = (phase_duration > 0.0).expand_as(lower).reshape(-1)
    # A small overlap allowance covers floating-point phase integration and
    # cumulative-length rounding at shared endpoints.
    epsilon = 32.0 * torch.finfo(lengths.dtype).eps
    position_lower = position(lower).reshape(-1)[active] - epsilon
    position_upper = position(upper).reshape(-1)[active] + epsilon
    v = (velocity_bound.reshape(-1)[active] * length)[:, None, None]
    a = (acceleration_bound.reshape(-1)[active] * length)[:, None, None]
    j = (jerk.abs().expand_as(lower).reshape(-1)[active] * length)[:, None, None]
    peaks = torch.zeros(
        (3, path_bounds.shape[-1]), device=lengths.device, dtype=torch.float64
    )
    offset = lengths.new_zeros((), dtype=torch.float64)
    for bounds, segment_length in zip(path_bounds, lengths.double()):
        path_lower = (offset + segment_length * fractions[:-1]) / length
        path_upper = (offset + segment_length * fractions[1:]) / length
        overlap = (
            (position_lower[:, None] <= path_upper[None])
            & (position_upper[:, None] >= path_lower[None])
        )[..., None]
        tangent, curvature, third = (bound[None] for bound in bounds)
        composed = torch.stack(
            (
                tangent * v,
                curvature * v.square() + tangent * a,
                third * v.pow(3) + 3.0 * curvature * v * a + tangent * j,
            )
        )
        peaks = torch.maximum(
            peaks, composed.masked_fill(~overlap[None], 0.0).amax(dim=(1, 2))
        )
        offset = offset + segment_length
    margin = 1.0 + max(1e-9, 256.0 * torch.finfo(lengths.dtype).eps)
    return (peaks * margin).to(lengths.dtype)
