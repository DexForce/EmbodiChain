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

"""Rest boundaries on a conservative fixed control grid for TOPPRA paths."""

from __future__ import annotations

import torch

from ._toppra_common import _validate_sampling
from ._toppra_warp import _retime_toppra_warp

__all__ = []


def _retime_toppra_stopped(
    waypoints: torch.Tensor,
    velocity: object,
    acceleration: object,
    *,
    sample_dt: float,
    grid_size: int,
) -> dict[str, torch.Tensor]:
    """Solve each adjacent pair at rest, then join valid prefixes on one clock.

    Each pair's uniform native samples are retained. Extending its spacing to
    the requested period can only slow it down; derivatives are scaled with
    that extension. This avoids interpolating a solved path again when an
    action inserts a gripper hold at a required waypoint. Default TOPPRA
    sampling and unconstrained intermediate velocities are handled elsewhere.
    """
    _validate_sampling(None, sample_dt)
    if waypoints.ndim != 3 or min(waypoints.shape) < 1 or waypoints.shape[1] < 2:
        raise ValueError("Stopped paths require shape (B, N, DOF) with N >= 2.")
    parts = [
        _retime_toppra_warp(
            waypoints[:, index : index + 2],
            velocity,
            acceleration,
            sample_dt=sample_dt,
            grid_size=grid_size,
        )
        for index in range(waypoints.shape[1] - 1)
    ]
    edges = torch.stack([(part["dt"] > 0).sum(dim=1) for part in parts])
    total_edges = edges.sum(dim=0)
    count = max(2, int(total_edges.max().item()) + 1)
    batch, _, dofs = waypoints.shape
    merged = {
        name: waypoints.new_zeros(batch, count, dofs)
        for name in ("positions", "velocities", "accelerations")
    }
    offset = torch.zeros_like(total_edges)
    for index, part in enumerate(parts):
        native_index = torch.arange(part["dt"].shape[1], device=waypoints.device)
        valid = native_index[None] <= edges[index, :, None]
        if index:
            valid = valid & (native_index[None] > 0)
        destination = (offset[:, None] + native_index[None]).clamp(max=count - 1)
        destination = destination[..., None].expand(-1, -1, dofs)
        # TIME sampling spaces each row uniformly over its solved duration.
        # Use the same sample count with a longer, exact control period.
        scale = (part["dt"].sum(dim=1) / (edges[index].clamp_min(1) * sample_dt)).clamp(
            max=1
        )
        for name, power in (("positions", 0), ("velocities", 1), ("accelerations", 2)):
            values = part[name] * scale[:, None, None].pow(power)
            merged[name].scatter_add_(
                1, destination, torch.where(valid[..., None], values, 0)
            )
        offset = offset + edges[index]
    success = torch.stack([part["success"] for part in parts]).all(dim=0)
    samples = torch.arange(count, device=waypoints.device)[None]
    tail = samples > total_edges[:, None]
    merged["positions"] = torch.where(
        tail[..., None], parts[-1]["positions"][:, -1:, :], merged["positions"]
    )
    for name in merged:
        merged[name] = torch.where(success[:, None, None], merged[name], 0)
    dt = torch.where(
        (samples > 0) & (samples <= total_edges[:, None]) & success[:, None],
        waypoints.new_tensor(sample_dt),
        waypoints.new_tensor(0),
    )
    return {"success": success, **merged, "dt": dt}
