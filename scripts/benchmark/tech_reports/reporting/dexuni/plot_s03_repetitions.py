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
"""Plot every S03 measurement repetition and export its plotting data.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/plot_s03_repetitions.py --batch PATH
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


def _performance_artifacts(batch: Path, attempts: list[dict], output: Path) -> None:
    """Export measured costs and every timed step used for the performance plot."""
    keys = (
        "platform",
        "repeat",
        "success",
        "run_id",
        "frames",
        "measured_frames",
        "step_ms",
        "step_median_ms",
        "step_ms_p95",
        "realtime_factor",
        "gpu_peak_mib",
        "init_s",
        "startup_steps_s",
        "readback_record_capture_ms",
        "process_wall_s",
    )
    with (output / "s03_performance.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**a["metrics"], "success": a["success"]} for a in attempts)
    with (output / "s03_step_timings.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "platform",
                "repeat",
                "frame",
                "simulation_time",
                "step_ms",
                "included_in_summary",
            )
        )
        for a in attempts:
            timing = np.atleast_1d(
                np.genfromtxt(
                    batch / a["run"] / "timing.csv", delimiter=",", names=True
                )
            )
            writer.writerows(
                (
                    a["platform"],
                    a["repeat"],
                    int(r["frame"]),
                    float(r["sim_time"]),
                    1000 * float(r["step_seconds"]),
                    i >= 30,
                )
                for i, r in enumerate(timing)
            )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for platform, color in (("embodichain", "tab:green"), ("isaaclab", "tab:orange")):
        selected = [
            a for a in attempts if a["platform"] == platform and a.get("success")
        ]
        x = [a["repeat"] + 1 for a in selected]
        axes[0].plot(
            x,
            [a["metrics"]["step_ms"] for a in selected],
            marker="o",
            color=color,
            label=platform,
        )
        axes[1].plot(
            x,
            [a["metrics"]["gpu_peak_mib"] for a in selected],
            marker="o",
            color=color,
            label=platform,
        )
    axes[0].set(
        ylabel="Synchronized step mean (ms)", title="No-camera platform step cost"
    )
    axes[1].set(
        ylabel="Sampled process framebuffer max (MiB)",
        title="GPU memory including initialization",
    )
    for ax in axes:
        ax.set(xlabel="Independent repetition", xticks=range(1, 6))
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "s03_performance.png", dpi=160)
    plt.close(fig)


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
    performance_only = (
        json.loads((batch / "protocol.json").read_text()).get("scope")
        == "performance_only"
    )
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
                height = (
                    1000 * data["particles"][:, data["corner_ids"], 2].mean(axis=1)
                    - attempt["metrics"]["corner_reference_height_mm"]
                )
                hold = phase == 3
                axes[0].plot(
                    t[hold] - t[hold][0],
                    height[hold],
                    color=color,
                    linestyle="-" if attempt.get("success") else "--",
                    alpha=0.65,
                    label=platform if index == 0 else None,
                )
                plot_rows += [
                    (platform, attempt["repeat"], float(ti), int(pi), float(hi))
                    for ti, pi, hi in zip(t, phase, height)
                ]
        axes[1].plot(
            [a["repeat"] + 1 for a in selected],
            [a["metrics"]["release_corner_height_mm"] for a in selected],
            marker="o",
            color=color,
            label=platform,
        )
    if not performance_only:
        axes[0].axhline(150, color="gray", linestyle="--", label="150 mm threshold")
    axes[0].set(
        xlabel="Time since hold began (s)",
        ylabel="Material corner lift (mm)",
        title=(
            "Descriptive hold response"
            if performance_only
            else "Recorded no-camera holds (dashed = failed task)"
        ),
    )
    if not performance_only:
        axes[1].axhline(25, color="gray", linestyle="--", label="25 mm threshold")
    axes[1].set(
        xlabel="Independent repetition",
        ylabel="Released corner height (mm)",
        title=(
            "Final corner height (not a quality score)"
            if performance_only
            else "Recorded release outcomes"
        ),
        xticks=range(1, 2 + max((a["repeat"] for a in attempts), default=4)),
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "s03_repetitions.png", dpi=160)
    plt.close(fig)
    with (output / "s03_height_curves.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("platform", "repeat", "simulation_time", "phase", "corner_lift_mm")
        )
        writer.writerows(plot_rows)
    if performance_only:
        _performance_artifacts(batch, attempts, output)
    keys = (
        "platform",
        "repeat",
        "success",
        "run_id",
        "step_ms",
        "corner_hold_mean_mm",
        "corner_hold_min_mm",
        "corner_hold_drop_mm",
        "release_corner_height_mm",
        "transport_corner_min_mm",
        "grip_drift_mm",
        "descent_grasp_drift_mm",
        "final_sheet_max_height_mm",
        "final_finger_min_mm",
        "table_penetration_mm",
        "failed_checks",
    )
    with (output / "s03_measurements.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**a["metrics"], "success": a["success"]} for a in attempts)


if __name__ == "__main__":
    main()
