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

"""Batched time-domain sampling and differentiation of joint trajectories."""

from __future__ import annotations

import math

import torch

__all__ = [
    "differentiate_positions",
    "resample_in_time",
    "retime_to_control_grid",
]


def _validate_timing(positions: torch.Tensor, dt: torch.Tensor) -> None:
    if positions.ndim != 3 or not positions.is_floating_point():
        raise ValueError("positions must be floating point with shape (B, N, D).")
    if dt.shape != positions.shape[:2] or dt.device != positions.device:
        raise ValueError("dt must match positions batch, samples, and device.")
    if not dt.is_floating_point() or not torch.isfinite(dt).all() or (dt < 0).any():
        raise ValueError("dt must contain finite non-negative floating intervals.")
    if not torch.isfinite(positions).all():
        raise ValueError("positions must contain only finite values.")
    if ((dt[:, 1:] == 0) & (positions.diff(dim=1) != 0).any(dim=-1)).any():
        raise ValueError("A zero time interval cannot change position.")


def differentiate_positions(positions: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Estimate reference velocities using nonuniform three-point differences.

    Interior points use a time-weighted average of neighboring slopes;
    endpoints use one-sided differences. A zero-duration duplicate contributes
    no slope: use the other side if present, otherwise return zero. This
    handles rectangular padding without inventing a small time interval.
    This numerical operation does not impose rest boundaries or motion limits.

    Args:
        positions: Joint samples of shape ``(B, N, D)``.
        dt: Arrival intervals of shape ``(B, N)``. The first interval is an
            arrival offset and does not participate in differentiation.

    Returns:
        Velocities with the positions' shape, floating dtype and device.

    Raises:
        ValueError: For malformed/nonfinite inputs, negative intervals, or
            position changes at identical timestamps.
    """
    _validate_timing(positions, dt)
    velocity = torch.zeros_like(positions)
    if positions.shape[1] < 2:
        return velocity
    h = dt[:, 1:].to(positions.dtype)
    valid = h > 0
    safe_h = torch.where(valid, h, torch.ones_like(h))
    slopes = positions.diff(dim=1) / safe_h.unsqueeze(-1)
    velocity[:, 0] = slopes[:, 0]
    velocity[:, -1] = slopes[:, -1]
    if positions.shape[1] > 2:
        left, right = h[:, :-1], h[:, 1:]
        total = left + right
        both = (left > 0) & (right > 0)
        weighted = (
            right.unsqueeze(-1) * slopes[:, :-1] + left.unsqueeze(-1) * slopes[:, 1:]
        ) / torch.where(total > 0, total, torch.ones_like(total)).unsqueeze(-1)
        one_sided = slopes[:, :-1] + slopes[:, 1:]
        velocity[:, 1:-1] = torch.where(both.unsqueeze(-1), weighted, one_sided)
    return velocity


def resample_in_time(
    positions: torch.Tensor, dt: torch.Tensor, sample_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly sample a timed path on a uniform per-row time grid.

    Preserves the first arrival offset, endpoints and each row's duration.
    Unlike distance resampling, dwell times and the sampled time profile affect
    interpolation. This does not certify continuous velocity/acceleration limits.

    Args:
        positions: Joint samples of shape ``(B, N, D)``, with ``N >= 1``.
        dt: Non-negative arrival intervals of shape ``(B, N)``.
        sample_count: Number of output samples, at least two.

    Returns:
        Positions ``(B, sample_count, D)`` and arrival intervals
        ``(B, sample_count)``, on the original device and with input dtypes.

    Raises:
        ValueError: For invalid timing, empty input paths, or fewer than two
            requested samples.
    """
    _validate_timing(positions, dt)
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 2
    ):
        raise ValueError("sample_count must be an integer of at least two.")
    if positions.shape[1] == 0:
        raise ValueError("Cannot resample an empty trajectory.")
    times = dt.cumsum(dim=1)
    fractions = torch.linspace(0, 1, sample_count, device=dt.device, dtype=dt.dtype)
    query = times[:, :1] + (times[:, -1:] - times[:, :1]) * fractions
    upper = torch.searchsorted(
        times.contiguous(), query.contiguous(), right=True
    ).clamp(max=positions.shape[1] - 1)
    lower = (upper - 1).clamp(min=0)
    t0, t1 = times.gather(1, lower), times.gather(1, upper)
    span = t1 - t0
    weight = torch.where(
        span > 0,
        (query - t0) / torch.where(span > 0, span, torch.ones_like(span)),
        torch.zeros_like(span),
    )
    dims = positions.shape[-1]
    q0 = positions.gather(1, lower.unsqueeze(-1).expand(-1, -1, dims))
    q1 = positions.gather(1, upper.unsqueeze(-1).expand(-1, -1, dims))
    result = torch.lerp(q0, q1, weight.to(positions.dtype).unsqueeze(-1))
    result[:, 0], result[:, -1] = positions[:, 0], positions[:, -1]
    intervals = torch.cat([dt[:, :1], query.diff(dim=1)], dim=1)
    return result, intervals


def retime_to_control_grid(
    positions: torch.Tensor,
    dt: torch.Tensor,
    control_dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retimes a joint path to a fixed command period without speeding it up.

    Each row's source duration ``T`` is quantized to
    ``ceil(T / control_dt) * control_dt``. Positions are sampled at uniformly
    spaced phases of the source path, while returned arrival intervals use the
    fixed destination period. Shorter rows are padded with terminal holds.

    Args:
        positions: Joint samples with shape ``(B, N, D)``.
        dt: Source arrival intervals with shape ``(B, N)``. The first interval
            must be zero.
        control_dt: Positive finite destination command period in seconds.

    Returns:
        Retimed positions, recomputed velocities, destination arrival
        intervals, and valid sample counts. The first and last valid velocity
        of every row are zero.

    Raises:
        ValueError: If the trajectory timing or destination period is invalid.
    """
    _validate_timing(positions, dt)
    if positions.shape[1] == 0:
        raise ValueError("Cannot retime an empty trajectory.")
    if (
        isinstance(control_dt, bool)
        or not isinstance(control_dt, (int, float))
        or not math.isfinite(control_dt)
        or control_dt <= 0
    ):
        raise ValueError("control_dt must be a positive finite number.")
    if (dt[:, 0] != 0).any():
        raise ValueError("The first arrival interval must be zero.")

    destination_dt = float(control_dt)
    durations = dt[:, 1:].sum(dim=1)
    interval_counts: list[int] = []
    for duration in durations.detach().cpu().tolist():
        ratio = duration / destination_dt
        nearest = round(ratio)
        tolerance = max(1.0e-9, abs(ratio) * 1.0e-7)
        interval_counts.append(
            int(
                nearest
                if math.isclose(ratio, nearest, abs_tol=tolerance)
                else math.ceil(ratio)
            )
        )

    valid_counts = torch.tensor(
        [count + 1 for count in interval_counts],
        dtype=torch.long,
        device=positions.device,
    )
    sample_count = int(valid_counts.max().item())
    sample_indices = torch.arange(sample_count, device=dt.device, dtype=dt.dtype)
    count_tensor = valid_counts.sub(1).to(dtype=dt.dtype)
    fractions = sample_indices.unsqueeze(0) / count_tensor.clamp_min(1).unsqueeze(1)
    fractions = fractions.clamp(max=1)

    source_times = dt.cumsum(dim=1)
    query = durations.unsqueeze(1) * fractions
    upper = torch.searchsorted(
        source_times.contiguous(), query.contiguous(), right=True
    ).clamp(max=positions.shape[1] - 1)
    lower = (upper - 1).clamp(min=0)
    t0 = source_times.gather(1, lower)
    t1 = source_times.gather(1, upper)
    span = t1 - t0
    weight = torch.where(
        span > 0,
        (query - t0) / torch.where(span > 0, span, torch.ones_like(span)),
        torch.zeros_like(span),
    )
    dimensions = positions.shape[-1]
    q0 = positions.gather(1, lower.unsqueeze(-1).expand(-1, -1, dimensions))
    q1 = positions.gather(1, upper.unsqueeze(-1).expand(-1, -1, dimensions))
    retimed = torch.lerp(q0, q1, weight.to(positions.dtype).unsqueeze(-1))
    retimed[:, 0] = positions[:, 0]
    terminal_indices = valid_counts.sub(1)
    rows = torch.arange(positions.shape[0], device=positions.device)
    retimed[rows, terminal_indices] = positions[:, -1]

    intervals = torch.zeros(
        positions.shape[0], sample_count, dtype=dt.dtype, device=dt.device
    )
    valid_mask = sample_indices.unsqueeze(0) < valid_counts.unsqueeze(1)
    intervals[:, 1:] = torch.where(
        valid_mask[:, 1:],
        torch.as_tensor(destination_dt, dtype=dt.dtype, device=dt.device),
        0,
    )
    velocities = differentiate_positions(retimed, intervals)
    velocities[:, 0] = 0
    velocities[rows, terminal_indices] = 0
    velocities = torch.where(valid_mask.unsqueeze(-1), velocities, 0)
    return retimed, velocities, intervals, valid_counts
