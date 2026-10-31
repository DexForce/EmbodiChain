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

"""Reproduce joint-limit overshoot without a robot, checkpoint or simulator.

Run from the checkout::

    python -m examples.sim.motion.planners.retime_joint_bounds --device cpu

The legacy control uses the unchanged ordinary TOPPRA geometry. The repaired
public retimer receives exactly the same nodes, dynamic limits and sample_dt.
This synthetic example is not a replay of the NMG benchmark corpus.
"""

from __future__ import annotations

import argparse
import json

import torch

from embodichain.compute.trajectory._toppra_warp import _retime_toppra_warp
from embodichain.lab.sim.motion.planners.toppra_planner import retime_joint_paths
from embodichain.lab.sim.motion.planners.utils import TrajectorySampleMethod


def summarize(result: dict[str, torch.Tensor], nodes: torch.Tensor) -> dict:
    """Check returned samples and independently differentiate their own clock."""
    q, dt = result["positions"][0], result["dt"][0]
    velocity = q.diff(dim=0) / dt[1:, None]
    acceleration = velocity.diff(dim=0) / (0.5 * (dt[1:-1] + dt[2:]))[:, None]
    previous = 0
    ordered = True
    for target in nodes[0, 1:]:
        candidates = torch.nonzero(
            (q[previous:] - target).abs().amax(dim=-1) <= 0.01
        ).flatten()
        if candidates.numel() == 0:
            ordered = False
            break
        previous += int(candidates[0])
    return {
        "solver_success": bool(result["success"][0]),
        "peak_joint_0_rad": float(q[:, 0].max()),
        "joint_limit_valid": bool((q[:, 0] <= 1.0 + 1e-5).all()),
        "start_valid": bool(torch.equal(q[0], nodes[0, 0])),
        "ordered_waypoints_reached": ordered,
        "goal_exact": bool(torch.equal(q[-1], nodes[0, -1])),
        "peak_analytic_velocity": float(result["velocities"].abs().max()),
        "peak_analytic_acceleration": float(result["accelerations"].abs().max()),
        "peak_divided_difference_velocity": float(velocity.abs().max()),
        "peak_divided_difference_acceleration": float(acceleration.abs().max()),
        "dynamic_limits_valid": bool(
            velocity.abs().max() <= 2.0 + 1e-7
            and acceleration.abs().max() <= 4.0 + 1e-7
        ),
        "duration_s": float(dt.sum()),
        "samples": len(dt),
        "max_dt_s": float(dt.max()),
    }


def main() -> None:
    """Print comparable before/after measurements and assert the regression."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    nodes = torch.tensor(
        [
            [
                [0.8, 0.0, 0.1, -2.0, 0.1, 1.0, 0.1],
                [1.0, 0.02, 0.1, -2.0, 0.1, 1.0, 0.1],
                [1.0, 0.04, 0.1, -2.0, 0.1, 1.0, 0.1],
            ]
        ],
        dtype=torch.float64,
        device=args.device,
    )
    before = _retime_toppra_warp(nodes, 2.0, 4.0, sample_dt=0.01)
    repaired = retime_joint_paths(
        nodes,
        constraints={"velocity": 2.0, "acceleration": 4.0},
        sample_method=TrajectorySampleMethod.TIME,
        sample_interval=0.01,
    )
    after = {key: getattr(repaired, key) for key in before}
    report = {
        "scope": "synthetic mechanism reproduction; no full benchmark replay",
        "device": args.device,
        "before": summarize(before, nodes),
        "after": summarize(after, nodes),
    }
    print(json.dumps(report, indent=2))
    assert (
        report["before"]["solver_success"] and not report["before"]["joint_limit_valid"]
    )
    assert all(
        report["after"][key]
        for key in (
            "solver_success",
            "joint_limit_valid",
            "start_valid",
            "ordered_waypoints_reached",
            "goal_exact",
            "dynamic_limits_valid",
        )
    )


if __name__ == "__main__":
    main()
