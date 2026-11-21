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

"""Pure cable routing measurements for declared board geometry.

Geometry measurements adapted from robosimstudio's Apache-2.0 example at
73739abe212f963e8a9f5862b27bca370df091a7; see the repository NOTICE.
https://github.com/robosimstudio/robosimstudio/tree/73739abe212f963e8a9f5862b27bca370df091a7/examples/08_demos/route_the_cable
"""

from __future__ import annotations

import torch
from .cable import cable_endpoints

__all__: list[str] = []

_CABLE_METRICS = (
    "seat_0",
    "seat_1",
    "port_count",
    "end_in_port",
    "strict_port",
    "release_ready",
    "reference_routed",
    "hands_clearance",
    "max_speed",
    "bench_penetration",
    "spool_penetration",
    "port_penetration",
    "max_grasp_slip",
    "end_position",
)


_CABLE_METRIC_UNITS = {
    name: (
        "m/s"
        if name == "max_speed"
        else (
            "m"
            if name
            in {
                "end_position",
                "hands_clearance",
                "bench_penetration",
                "spool_penetration",
                "port_penetration",
                "max_grasp_slip",
            }
            else "unitless"
        )
    )
    for name in _CABLE_METRICS
}


def _measure_cable_route(
    state: torch.Tensor,
    lengths: torch.Tensor,
    hands_w: torch.Tensor,
    origins: torch.Tensor,
    geometry: dict[str, object],
    radius: float,
) -> dict[str, torch.Tensor]:
    """Measure the reference board goal and sampled geometry diagnostics.

    Args:
        state: World cable segment states ``(B,50,13)``.
        lengths: Capsule axis rest lengths ``(50,)``.
        hands_w: World tool centres ``(B,2,3)``.
        origins: World Arena translations ``(B,3)``.

    Returns:
        Per-environment counts, speed, clearance and routing acceptance.
    """
    bench = geometry["bench_height"]
    port_x, port_y = geometry["port_center"]
    p = state[..., :3] - origins[:, None, :]
    starts, ends = cable_endpoints(state, lengths)
    start_local = starts - origins[:, None, :]
    end_local = ends - origins[:, None, :]
    valid = torch.isfinite(state).all(dim=(1, 2)) & (
        state[..., 3:7].norm(dim=-1) > 1e-8
    ).all(-1)
    out = {}
    for i, (x, y, side) in enumerate(geometry["seats"]):
        lo, hi = sorted((x + side * 0.003, x + side * 0.045))
        inside = (
            (p[:, :, 0] >= lo)
            & (p[:, :, 0] <= hi)
            & ((p[:, :, 1] - y).abs() <= 0.035)
            & (p[:, :, 2] >= bench)
            & (p[:, :, 2] <= bench + 0.15)
        )
        out[f"seat_{i}"] = inside.sum(-1)
    in_port = (
        ((p[:, :, 0] - port_x).abs() <= 0.045)
        & ((p[:, :, 1] - port_y).abs() <= 0.045)
        & (p[:, :, 2] >= bench)
        & (p[:, :, 2] <= bench + 0.15)
    )
    end = end_local[:, -1]
    out["end_position"] = end
    out["port_count"] = in_port.sum(-1)
    out["end_in_port"] = (
        ((end[:, :2] - end.new_tensor([port_x, port_y])).abs() <= 0.045).all(-1)
        & (end[:, 2] >= bench)
        & (end[:, 2] <= bench + 0.15)
    )
    out["hands_clearance"] = torch.cdist(hands_w, state[..., :3]).amin(dim=(1, 2))
    out["max_speed"] = torch.linalg.vector_norm(state[..., 7:10], dim=-1).amax(-1)
    out["bench_penetration"] = (
        bench + radius - torch.minimum(start_local[..., 2], end_local[..., 2]).amin(-1)
    ).clamp_min(0)
    # Five points per capsule axis, using the exact capped-cylinder SDF.
    axis_samples = (
        start_local[:, :, None, :]
        + torch.linspace(0, 1, 5, device=state.device)[None, None, :, None]
        * (end_local - start_local)[:, :, None, :]
    )
    penetration = torch.zeros(state.shape[0], device=state.device)
    for x, y, cylinder_radius, z0, z1 in geometry["cylinders"]:
        radius_at_spool = cylinder_radius
        radial = (
            torch.linalg.vector_norm(
                axis_samples[..., :2] - state.new_tensor([x, y]), dim=-1
            )
            - radius_at_spool
        )
        vertical = (axis_samples[..., 2] - (bench + (z0 + z1) / 2)).abs() - (
            z1 - z0
        ) / 2
        q = torch.stack((radial, vertical), -1)
        sdf = torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.amax(-1).clamp_max(0)
        penetration = torch.maximum(
            penetration, (radius - sdf).clamp_min(0).amax(dim=(1, 2))
        )
    out["spool_penetration"] = penetration
    port_penetration = torch.zeros_like(penetration)
    for center, half_size in geometry["boxes"]:
        q = (axis_samples - state.new_tensor(center)).abs() - state.new_tensor(
            half_size
        )
        sdf = torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.amax(-1).clamp_max(0)
        port_penetration = torch.maximum(
            port_penetration, (radius - sdf).clamp_min(0).amax(dim=(1, 2))
        )
    out["port_penetration"] = port_penetration
    strict_lo, strict_hi = (state.new_tensor(v) for v in geometry["strict_region"])
    release_lo, release_hi = (state.new_tensor(v) for v in geometry["release_region"])
    out["strict_port"] = ((end >= strict_lo) & (end <= strict_hi)).all(-1)
    inside_release = ((end >= release_lo) & (end <= release_hi)).all(-1)
    # An endpoint already seated deeply inside the physical channel need not
    # be pushed toward the narrower corridor used for release from above.
    seated_deeply = out["strict_port"] & (end[:, 1] >= release_lo[1])
    out["release_ready"] = valid & (inside_release | seated_deeply)
    out["reference_routed"] = (
        valid
        & (out["seat_0"] >= 2)
        & (out["seat_1"] >= 2)
        & (out["port_count"] >= 3)
        & out["end_in_port"]
        & (out["hands_clearance"] > 0.05)
        & (out["max_speed"] <= 0.05)
    )
    return out
