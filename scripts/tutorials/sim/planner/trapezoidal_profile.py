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

"""Plot a minimal scalar trapezoidal or Double-S trajectory example."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/embodichain-matplotlib")

import matplotlib.pyplot as plt
import torch

from embodichain.lab.sim.planners.trapezoidal_planner import (
    TrapezoidalPlanOptions,
    _plan_linear_profiles,
)


def configure_plot_fonts() -> None:
    """Apply lightweight readable font defaults for this example."""
    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans", "sans-serif"],
            "font.size": 11.0,
            "axes.titlesize": 13.0,
            "legend.fontsize": 9.0,
            "axes.unicode_minus": False,
        }
    )


def positive_float(value: str) -> float:
    """Parse a finite positive floating-point argument."""
    parsed = float(value)
    if not torch.isfinite(torch.tensor(parsed)) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse example options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("velocity_trapezoidal", "acceleration_trapezoidal"),
        default="acceleration_trapezoidal",
    )
    parser.add_argument("--distance", type=positive_float, default=0.1)
    parser.add_argument("--velocity", type=positive_float, default=0.15)
    parser.add_argument("--acceleration", type=positive_float, default=0.3)
    parser.add_argument("--jerk", type=positive_float, default=1.0)
    parser.add_argument("--samples", type=int, default=501)
    parser.add_argument(
        "--show-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the diagnostic figure (default: enabled).",
    )
    return parser.parse_args()


def main() -> None:
    """Plan the scalar move and show its derivatives."""
    args = parse_args()
    configure_plot_fonts()
    if args.samples < 3:
        raise SystemExit("--samples must be at least 3")
    planner_profile = (
        "trapezoidal" if args.profile == "velocity_trapezoidal" else "double_s"
    )
    waypoints = torch.tensor([[[0.0], [args.distance]]], dtype=torch.float64)
    result = _plan_linear_profiles(
        waypoints,
        TrapezoidalPlanOptions(
            profile=planner_profile,
            constraints={
                "velocity": args.velocity,
                "acceleration": args.acceleration,
                "jerk": args.jerk,
            },
            sample_interval=args.samples,
            backend="torch",
        ),
    )
    time = result.dt[0].cumsum(dim=0)
    position = result.positions[0, :, 0]
    velocity = result.velocities[0, :, 0]
    acceleration = result.accelerations[0, :, 0]
    jerk = torch.gradient(acceleration, spacing=(time,), edge_order=2)[0]

    print(
        f"[INFO] profile={args.profile}, duration={result.duration.item():.12f} s, "
        f"max_velocity={velocity.abs().max().item():.6f}, "
        f"max_acceleration={acceleration.abs().max().item():.6f}, "
        f"max_sampled_jerk={jerk.abs().max().item():.6f}"
    )

    figure, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for axis, values, title, ylabel in (
        (axes[0], position, "Position", "q"),
        (axes[1], velocity, "Velocity", "dq/dt"),
        (axes[2], acceleration, "Acceleration", "d²q/dt²"),
        (axes[3], jerk, "Jerk", "d³q/dt³"),
    ):
        axis.plot(time.numpy(), values.numpy(), linewidth=2.0)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    figure.suptitle(args.profile.replace("_", " ").title())
    figure.tight_layout()
    if args.show_plot:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
