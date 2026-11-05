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
"""Plot every S02 measurement repetition and export its plotting data.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/plot_s02_repetitions.py --batch PATH
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["main"]


def main() -> None:
    """Show observed repetition variability, including near-threshold passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    batch = parser.parse_args().batch.resolve()
    attempts = [
        a
        for a in json.loads((batch / "attempts.json").read_text())
        if a["mode"] == "measurement" and "run" in a
    ]
    output = batch / "results"
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_rows = []
    for platform, color in (("embodichain", "tab:green"), ("isaaclab", "tab:orange")):
        selected = [a for a in attempts if a["platform"] == platform]
        for index, attempt in enumerate(selected):
            with np.load(batch / attempt["run"] / "states.npz") as data:
                t, phase = data["time"], data["phase"]
                config = json.loads(
                    (batch / attempt["run"] / "config.json").read_text()
                )
                height = 1000 * (
                    data["particles"][:, :, 2].min(axis=1) - config["scene"]["table_z"]
                )
                hold = phase == 4
                axes[0].plot(
                    t[hold] - t[hold][0],
                    height[hold],
                    color=color,
                    alpha=0.65,
                    label=platform if index == 0 else None,
                )
                plot_rows += [
                    (platform, attempt["repeat"], float(ti), int(pi), float(hi))
                    for ti, pi, hi in zip(t, phase, height)
                ]
        axes[1].plot(
            [a["repeat"] + 1 for a in selected],
            [a["metrics"]["place_center_error_mm"] for a in selected],
            marker="o",
            color=color,
            label=platform,
        )
    axes[0].axhline(80, color="gray", linestyle="--", label="80 mm threshold")
    axes[0].set(
        xlabel="Time since hold began (s)",
        ylabel="Lowest-node height (mm)",
        title="All five no-camera hold trajectories",
    )
    axes[1].axhline(20, color="gray", linestyle="--", label="20 mm threshold")
    axes[1].set(
        xlabel="Independent repetition",
        ylabel="Placement center error (mm)",
        title="All five placement outcomes",
        xticks=range(1, 6),
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "s02_repetitions.png", dpi=160)
    plt.close(fig)
    with (output / "s02_height_curves.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("platform", "repeat", "simulation_time", "phase", "lowest_height_mm")
        )
        writer.writerows(plot_rows)
    keys = (
        "platform",
        "repeat",
        "success",
        "run_id",
        "step_ms",
        "grasp_compression_pct",
        "lift_min_height_mm",
        "hold_drop_mm",
        "place_center_error_mm",
        "place_inside_fraction",
        "grip_slip_mm",
    )
    with (output / "s02_measurements.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**a["metrics"], "success": a["success"]} for a in attempts)


if __name__ == "__main__":
    main()
