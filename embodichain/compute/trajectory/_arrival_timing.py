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

"""Batched partition/assembly at identity-bearing ordered trajectory visits."""

from __future__ import annotations

import torch

__all__ = []


def _split_arrival_segments(
    positions: torch.Tensor,
    arrival_indices: torch.Tensor,
    path_lengths: torch.Tensor | None = None,
    dt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Flatten B trajectories into B*(K+1) phases, retaining every source knot."""
    batch, count, dofs = positions.shape
    if count < 1:
        raise ValueError("Arrival partition requires nonempty positions.")
    if (
        arrival_indices.ndim != 2
        or arrival_indices.shape[0] != batch
        or arrival_indices.dtype not in (torch.int32, torch.int64)
        or arrival_indices.device != positions.device
    ):
        raise ValueError("arrival_indices must be integer (B, K) on the path device.")
    lengths = (
        torch.full((batch,), count, device=positions.device, dtype=torch.long)
        if path_lengths is None
        else path_lengths
    )
    if (
        lengths.shape != (batch,)
        or lengths.device != positions.device
        or lengths.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("path_lengths must be integer (B,) on the path device.")
    valid = arrival_indices >= 0
    invalid = ((lengths < 1) | (lengths > count)).any()
    invalid |= ((arrival_indices < -1) | (arrival_indices >= lengths[:, None])).any()
    invalid |= (
        valid[:, 1:]
        & (~valid[:, :-1] | (arrival_indices[:, 1:] <= arrival_indices[:, :-1]))
    ).any()
    if bool(invalid):
        raise ValueError(
            "Arrivals must be a strictly ordered valid prefix followed by -1."
        )
    ends = torch.cat(
        (
            torch.where(valid, arrival_indices, lengths[:, None] - 1),
            lengths[:, None] - 1,
        ),
        dim=1,
    )
    starts = torch.cat((torch.zeros_like(ends[:, :1]), ends[:, :-1]), dim=1)
    spans = ends - starts + 1
    capacity = max(2, int(spans.max()))
    offsets = torch.arange(capacity, device=positions.device)
    indices = torch.minimum(starts[:, :, None] + offsets, ends[:, :, None])
    rows = torch.arange(batch, device=positions.device)[:, None, None]
    segments = positions[rows, indices].reshape(-1, capacity, dofs)
    intervals = None
    if dt is not None:
        if dt.shape != positions.shape[:2] or dt.device != positions.device:
            raise ValueError("dt must match positions.")
        intervals = dt[rows, indices]
        intervals = torch.where(offsets[None, None] < spans[:, :, None], intervals, 0)
        intervals[..., 0] = 0
        intervals = intervals.reshape(-1, capacity)
    # A distinct visit to a stationary configuration still needs a command slot.
    needs_visit = (
        torch.cat((valid, torch.zeros_like(valid[:, :1])), dim=1)
        if valid.shape[1]
        else torch.zeros_like(ends, dtype=torch.bool)
    )
    needs_visit &= ends > starts
    return segments, spans.clamp_min(2).reshape(-1), intervals, needs_visit


def _join_arrival_segments(
    result: dict[str, torch.Tensor],
    arrival_indices: torch.Tensor,
    needs_visit: torch.Tensor,
    minimum_event_dt: float,
) -> dict[str, torch.Tensor]:
    """Compact per-row phases without adding other rows' terminal padding."""
    batch, phases = needs_visit.shape
    q = result["positions"].reshape(batch, phases, -1, result["positions"].shape[-1])
    intervals = result["dt"].reshape(batch, phases, -1).clone()
    if q.shape[2] < 2:
        q = q.expand(-1, -1, 2, -1)
        intervals = intervals.expand(-1, -1, 2).clone()
    stationary_visits = needs_visit & (intervals.sum(dim=-1) == 0)
    intervals[:, :, 1] = torch.where(
        stationary_visits, minimum_event_dt, intervals[:, :, 1]
    )
    steps = (intervals > 0).sum(dim=-1)
    cumulative = steps.cumsum(dim=1)
    previous = torch.cat(
        (torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=1
    )
    lengths = cumulative[:, -1] + 1
    capacity = int(lengths.max())
    samples = torch.arange(capacity, device=q.device)[None].expand(batch, -1)
    query = torch.minimum(samples, (lengths - 1)[:, None]).contiguous()
    phase = torch.searchsorted(cumulative.contiguous(), query, right=False).clamp_max(
        phases - 1
    )
    local = query - previous.gather(1, phase)
    rows = torch.arange(batch, device=q.device)[:, None]
    output = {
        "positions": q[rows, phase, local],
        "dt": intervals[rows, phase, local],
        "valid_lengths": lengths,
        "waypoint_indices": torch.where(arrival_indices >= 0, cumulative[:, :-1], -1),
    }
    output["dt"] = torch.where(samples < lengths[:, None], output["dt"], 0)
    output["dt"][:, 0] = 0
    for name in ("velocities", "accelerations"):
        if name in result:
            values = result[name].reshape(batch, phases, -1, q.shape[-1])
            if values.shape[2] < 2:
                values = values.expand(-1, -1, 2, -1)
            output[name] = torch.where(
                (samples < lengths[:, None])[..., None], values[rows, phase, local], 0
            )
    if "success" in result:
        success = result["success"].reshape(batch, phases).all(dim=1)
        output["success"] = success
        for name in ("positions", "velocities", "accelerations", "dt"):
            if name in output:
                mask = success.reshape(batch, *([1] * (output[name].ndim - 1)))
                output[name] = torch.where(mask, output[name], 0)
        output["waypoint_indices"] = torch.where(
            success[:, None], output["waypoint_indices"], -1
        )
        output["valid_lengths"] = torch.where(success, lengths, 1)
    return output


def _resample_at_arrivals(
    positions: torch.Tensor,
    dt: torch.Tensor,
    control_dt: float,
    arrival_indices: torch.Tensor,
    path_lengths: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Retain visits on the command grid; this alone does not solve dynamics."""
    from .timing import retime_to_control_grid, differentiate_positions
    from ._trace import _record_trajectory

    q, _, intervals, visits = _split_arrival_segments(
        positions, arrival_indices, path_lengths, dt
    )
    resampled, _, command_dt, _ = retime_to_control_grid(q, intervals, control_dt)
    output = _join_arrival_segments(
        {"positions": resampled, "dt": command_dt}, arrival_indices, visits, control_dt
    )
    output["velocities"] = differentiate_positions(output["positions"], output["dt"])
    _record_trajectory(
        "arrival_control_grid",
        source_positions=positions,
        source_dt=dt,
        source_arrival_indices=arrival_indices,
        **output,
    )
    return output
