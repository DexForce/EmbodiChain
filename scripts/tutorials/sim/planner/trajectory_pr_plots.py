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
"""Plot the trajectory PR's main features without starting simulation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("MPLCONFIGDIR", "/tmp/embodichain-matplotlib")

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from embodichain.lab.sim.planners.bezier import BezierPath
from embodichain.lab.sim.planners.se3 import plan_se3_line
from embodichain.lab.sim.planners.trapezoidal_planner import (
    TrapezoidalPlanOptions,
    _plan_linear_profiles,
)
from embodichain.lab.sim.planners.utils import PlanResult

Scenario = Literal[
    "bezier", "trapezoidal", "double-s", "blend", "minimum-duration", "se3"
]


def _plan(
    waypoints: torch.Tensor,
    profile: Literal["trapezoidal", "double_s"],
    *,
    blend_tolerance: float = 0.0,
    minimum_duration: float | None = None,
) -> PlanResult:
    return _plan_linear_profiles(
        waypoints,
        TrapezoidalPlanOptions(
            profile=profile,
            constraints={
                "velocity": [0.8, 0.6],
                "acceleration": [1.5, 1.2],
                "jerk": [4.0, 3.0],
            },
            sample_interval=501,
            blend_tolerance=blend_tolerance,
            minimum_duration=minimum_duration,
            backend="torch",
        ),
    )


def _time(result: PlanResult) -> torch.Tensor:
    return result.dt[0].cumsum(dim=0).cpu()


def _plot_profile(
    result: PlanResult, title: str, *, comparison: PlanResult | None = None
) -> Figure:
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    series = (
        (result.positions[0], "position"),
        (result.velocities[0], "velocity"),
        (result.accelerations[0], "acceleration"),
    )
    time = _time(result)
    for axis, (values, label) in zip(axes[:3], series, strict=True):
        for joint in range(values.shape[1]):
            axis.plot(time, values[:, joint].cpu(), label=f"q{joint}")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    jerk = torch.gradient(result.accelerations[0].cpu(), spacing=(time,), dim=0)[0]
    for joint in range(jerk.shape[1]):
        axes[3].plot(time, jerk[:, joint], label=f"q{joint}")
    axes[3].set_ylabel("sampled jerk")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(alpha=0.25)
    if comparison is not None:
        comparison_time = _time(comparison)
        axes[0].plot(
            comparison_time,
            comparison.positions[0, :, 0].cpu(),
            "--",
            label="q0 normal duration",
        )
    axes[0].legend(ncol=3)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_bezier() -> Figure:
    """Plot quadratic and quintic Bézier paths and their control polygons."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    controls_by_degree = {
        2: [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]],
        5: [
            [0.0, 0.0],
            [0.2, 0.4],
            [0.4, 0.6],
            [0.6, 0.6],
            [0.8, 0.4],
            [1.0, 0.0],
        ],
    }
    parameter = torch.linspace(0.0, 1.0, 501, dtype=torch.float64)
    for axis, (degree, values) in zip(axes, controls_by_degree.items(), strict=True):
        controls = torch.tensor(values, dtype=torch.float64)
        path = BezierPath(controls)
        points = path.evaluate(parameter)
        axis.plot(controls[:, 0], controls[:, 1], "o--", label="control polygon")
        axis.plot(points[:, 0], points[:, 1], linewidth=2, label="curve")
        axis.set_title(f"degree {degree}, length={path.length.item():.6f}")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    return figure


def plot_trapezoidal() -> Figure:
    waypoints = torch.tensor([[[0.0, 0.0], [1.0, -0.5]]], dtype=torch.float64)
    return _plot_profile(_plan(waypoints, "trapezoidal"), "Trapezoidal profile")


def plot_double_s() -> Figure:
    waypoints = torch.tensor([[[0.0, 0.0], [1.0, -0.5]]], dtype=torch.float64)
    return _plot_profile(_plan(waypoints, "double_s"), "Double-S profile")


def plot_blend() -> Figure:
    """Plot blended joint-space geometry and its time derivatives."""
    waypoints = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]],
        dtype=torch.float64,
    )
    result = _plan(waypoints, "double_s", blend_tolerance=0.1)
    figure = _plot_profile(result, "Quintic waypoint blend")
    inset = figure.add_axes((0.62, 0.72, 0.3, 0.2))
    inset.plot(waypoints[0, :, 0], waypoints[0, :, 1], "o--", label="waypoints")
    inset.plot(result.positions[0, :, 0], result.positions[0, :, 1], label="blend")
    inset.set_aspect("equal", adjustable="box")
    inset.grid(alpha=0.25)
    inset.legend(fontsize=8)
    return figure


def plot_minimum_duration() -> Figure:
    waypoints = torch.tensor([[[0.0, 0.0], [1.0, -0.5]]], dtype=torch.float64)
    normal = _plan(waypoints, "double_s")
    slowed = _plan(waypoints, "double_s", minimum_duration=5.0)
    return _plot_profile(slowed, "Double-S scaled to 5 seconds", comparison=normal)


def plot_se3() -> Figure:
    """Plot translation and linear velocity along an SE3 screw path."""
    start = torch.eye(4, dtype=torch.float64)
    end = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    result = plan_se3_line(
        start,
        end,
        torch.tensor([0.4, 0.3, 0.2, 1.0, 1.0, 1.0], dtype=torch.float64),
        torch.full((6,), 0.8, dtype=torch.float64),
        torch.full((6,), 2.0, dtype=torch.float64),
        sample_count=501,
    )
    figure, axes = plt.subplots(2, 1, figsize=(10, 7))
    translation = result.poses[:, :3, 3].cpu()
    for index, label in enumerate("xyz"):
        axes[0].plot(result.times.cpu(), translation[:, index], label=label)
        axes[1].plot(result.times.cpu(), result.velocities[:, index].cpu(), label=label)
    axes[0].set_ylabel("translation [m]")
    axes[1].set_ylabel("linear velocity [m/s]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("SE(3) screw interpolation")
    figure.tight_layout()
    return figure


PLOTS = {
    "bezier": plot_bezier,
    "trapezoidal": plot_trapezoidal,
    "double-s": plot_double_s,
    "blend": plot_blend,
    "minimum-duration": plot_minimum_duration,
    "se3": plot_se3,
}


def main() -> None:
    """Generate and optionally display one diagnostic plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=PLOTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    output = args.output or Path("outputs/trajectory_plots") / f"{args.scenario}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = PLOTS[args.scenario]()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    print(f"[PASS] wrote {output}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
