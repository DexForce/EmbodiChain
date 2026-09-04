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
"""Run the trajectory PR's main features one scenario at a time.

This lightweight diagnostic does not start simulation. Each scenario prints
the quantities that are useful when comparing behavior with HolisticMotion and
raises an error when an invariant is violated.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from typing import Literal

os.environ.setdefault("MPLCONFIGDIR", "/tmp/embodichain-matplotlib")

import torch

from embodichain.lab.sim.planners.bezier import (
    BezierPath,
    quintic_blend_segments,
)
from embodichain.lab.sim.planners.se3 import plan_se3_line
from embodichain.lab.sim.planners.trapezoidal_planner import (
    TrapezoidalPlanOptions,
    _plan_linear_profiles,
)
from embodichain.lab.sim.planners.utils import PlanResult


def _print_tensor(name: str, value: torch.Tensor) -> None:
    """Print a tensor using stable precision and compact formatting."""
    print(f"{name}: {value.detach().cpu().numpy()}")


def _plan(
    waypoints: torch.Tensor,
    *,
    profile: Literal["trapezoidal", "double_s"],
    samples: int = 201,
    blend_tolerance: float = 0.0,
    minimum_duration: float | None = None,
    backend: Literal["auto", "torch", "warp"] = "torch",
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
            sample_interval=samples,
            blend_tolerance=blend_tolerance,
            minimum_duration=minimum_duration,
            backend=backend,
        ),
    )


def run_bezier() -> None:
    """Evaluate quadratic/quintic Bézier values, derivatives, and arc length."""
    parameter = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    for degree, controls in (
        (2, [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]),
        (
            5,
            [[0.0, 0.0], [0.2, 0.4], [0.4, 0.6], [0.6, 0.6], [0.8, 0.4], [1.0, 0.0]],
        ),
    ):
        path = BezierPath(torch.tensor(controls, dtype=torch.float64))
        print(f"\nBezier degree {degree}")
        _print_tensor("values", path.evaluate(parameter))
        _print_tensor("first_derivative", path.derivative(parameter))
        _print_tensor("second_derivative", path.derivative(parameter, order=2))
        print(f"arc_length: {path.length.item():.12f}")


def _run_profile(profile: Literal["trapezoidal", "double_s"]) -> None:
    """Print one scalar-profile diagnostic."""
    waypoints = torch.tensor([[[0.0, 0.0], [1.0, -0.5]]], dtype=torch.float64)
    result = _plan(waypoints, profile=profile)
    report = result.constraint_report
    assert report is not None and bool(report["within_limits"].all().item())
    print(f"duration: {result.duration.item():.12f}")
    _print_tensor("mid_position", result.positions[0, result.positions.shape[1] // 2])
    _print_tensor("peak_velocity_per_joint", report["peak_velocity_per_joint"][0])
    _print_tensor(
        "peak_acceleration_per_joint", report["peak_acceleration_per_joint"][0]
    )
    _print_tensor("peak_jerk_per_joint", report["peak_jerk_per_joint"][0])


def run_trapezoidal() -> None:
    """Run the acceleration-limited trapezoidal time law."""
    _run_profile("trapezoidal")


def run_double_s() -> None:
    """Run the jerk-limited seven-phase Double-S time law."""
    _run_profile("double_s")


def run_blend() -> None:
    """Inspect quintic corner controls and the time-parameterized blend."""
    waypoints = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]],
        dtype=torch.float64,
    )
    segments, lengths = quintic_blend_segments(waypoints[0], 0.1)
    result = _plan(
        waypoints,
        profile="double_s",
        samples=501,
        blend_tolerance=0.1,
    )
    report = result.constraint_report
    assert report is not None and bool(report["within_limits"].all().item())
    print(f"segment_degrees: {[segment.shape[0] - 1 for segment in segments]}")
    _print_tensor("segment_lengths", lengths)
    print(f"duration: {result.duration.item():.12f}")
    _print_tensor("peak_velocity_per_joint", report["peak_velocity_per_joint"][0])
    _print_tensor(
        "peak_acceleration_per_joint", report["peak_acceleration_per_joint"][0]
    )
    _print_tensor("peak_jerk_per_joint", report["peak_jerk_per_joint"][0])


def run_minimum_duration() -> None:
    """Show uniform derivative scaling caused by minimum_duration."""
    waypoints = torch.tensor([[[0.0, 0.0], [1.0, -0.5]]], dtype=torch.float64)
    normal = _plan(waypoints, profile="double_s", samples=501)
    slowed = _plan(waypoints, profile="double_s", samples=501, minimum_duration=5.0)
    print(f"normal_duration: {normal.duration.item():.12f}")
    print(f"scaled_duration: {slowed.duration.item():.12f}")
    for key in (
        "peak_velocity_per_joint",
        "peak_acceleration_per_joint",
        "peak_jerk_per_joint",
    ):
        _print_tensor(f"normal_{key}", normal.constraint_report[key][0])
        _print_tensor(f"scaled_{key}", slowed.constraint_report[key][0])
    torch.testing.assert_close(
        slowed.positions[:, [0, -1]], normal.positions[:, [0, -1]]
    )


def run_batch() -> None:
    """Exercise moving and stationary paths in the same blended batch."""
    waypoints = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.2, -0.1], [0.2, -0.1], [0.2, -0.1]],
        ],
        dtype=torch.float64,
    )
    result = _plan(waypoints, profile="double_s", blend_tolerance=0.1)
    torch.testing.assert_close(
        result.positions[1], waypoints[1, :1].expand_as(result.positions[1])
    )
    assert bool(result.constraint_report["within_limits"].all().item())
    _print_tensor("success", result.success)
    _print_tensor("duration_per_batch", result.duration)
    _print_tensor("within_limits", result.constraint_report["within_limits"])


def run_se3() -> None:
    """Plan a screw-interpolated Cartesian SE(3) trajectory."""
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
        sample_count=201,
    )
    assert bool(result.constraint_report["within_limits"].item())
    print(f"duration: {result.duration.item():.12f}")
    _print_tensor("mid_pose", result.poses[result.poses.shape[0] // 2])
    _print_tensor("peak_velocity", result.constraint_report["peak_velocity"])
    _print_tensor("peak_acceleration", result.constraint_report["peak_acceleration"])
    _print_tensor("peak_jerk", result.constraint_report["peak_jerk"])


def run_backend() -> None:
    """Compare Torch and Warp profile construction/sampling on CPU."""
    waypoints = torch.tensor([[[0.0, 0.0], [1.0, -0.5]]], dtype=torch.float32)
    for profile in ("trapezoidal", "double_s"):
        torch_result = _plan(waypoints, profile=profile, backend="torch")
        warp_result = _plan(waypoints, profile=profile, backend="warp")
        position_error = (torch_result.positions - warp_result.positions).abs().max()
        velocity_error = (torch_result.velocities - warp_result.velocities).abs().max()
        acceleration_error = (
            (torch_result.accelerations - warp_result.accelerations).abs().max()
        )
        print(f"\n{profile}")
        duration_error = (torch_result.duration - warp_result.duration).abs().max()
        print(f"duration_error: {duration_error.item():.9g}")
        print(f"position_error: {position_error.item():.9g}")
        print(f"velocity_error: {velocity_error.item():.9g}")
        print(f"acceleration_error: {acceleration_error.item():.9g}")
        torch.testing.assert_close(
            torch_result.positions, warp_result.positions, atol=2e-5, rtol=2e-5
        )


SCENARIOS: dict[str, Callable[[], None]] = {
    "bezier": run_bezier,
    "trapezoidal": run_trapezoidal,
    "double-s": run_double_s,
    "blend": run_blend,
    "minimum-duration": run_minimum_duration,
    "batch": run_batch,
    "se3": run_se3,
    "backend": run_backend,
}


def main() -> None:
    """Run the selected diagnostic scenario."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=SCENARIOS)
    args = parser.parse_args()
    torch.set_printoptions(precision=9, linewidth=140)
    SCENARIOS[args.scenario]()
    print(f"\n[PASS] {args.scenario}")


if __name__ == "__main__":
    main()
