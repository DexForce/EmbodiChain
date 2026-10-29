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
"""Summarize M05 native diagnostic states and save physical plots, without timing.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_m05_media.py --run <directory>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m05_common import (
    BAG_TARGET,
    OUTPUT,
    TABLE_Z,
)

__all__ = ["main"]


def main() -> None:
    """Measure fixed windows and plot their physical trajectories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run = args.run
    manifest = json.loads((run / "manifest.json").read_text())
    timing = dict((name, end) for end, name in manifest["scene"]["phases"])
    duration = timing["observe"]
    reference_range = [timing["unloaded reference"] - 1, timing["unloaded reference"]]
    loaded_range = [duration - 1, duration]
    if manifest["status"] != "completed" or manifest["frames"] != round(duration * 60):
        raise ValueError("Require a full completed native preview")
    rows = json.loads((run / "preview/metrics.json").read_text())
    states = np.load(run / "preview/states.npz")
    mesh = np.load(run / "inputs/bag_mesh.npz")
    central_ids = mesh["top_centre_group"]
    time = states["time"]
    center = states["bag"].mean(1)
    block = states["block_pose"][:, :3, 3]
    height = np.array([r["bag_height_m"] for r in rows])
    top = np.array([r["bag_top_centre_z_m"] for r in rows])
    bottom = np.array([r["block_bottom_z_m"] for r in rows])
    tilt = np.array([r["block_tilt_deg"] for r in rows])
    volume = np.array([r["volume_ratio"] for r in rows])
    table_center, table_size = next(
        (np.asarray(center), np.asarray(size))
        for name, center, size in manifest["scene"]["supports"]
        if name == "table"
    )
    table_covers_bag = bool(
        np.all(np.abs(states["bag"][:, :, :2] - table_center[:2]) <= table_size[:2] / 2)
    )
    reference = (time >= reference_range[0] - 1e-8) & (
        time <= reference_range[1] + 1e-8
    )
    loaded = time >= loaded_range[0] - 1e-8
    first_second = (time >= timing["block release"] - 1e-8) & (
        time <= timing["block release"] + 1 + 1e-8
    )
    carry = (time >= timing["block settle"] - 1e-8) & (
        time <= timing["block carry"] + 1e-8
    )
    offset = block - states["tcp"][:, 0]
    settle = (time >= timing["block settle"] - 0.2 - 1e-8) & (
        time <= timing["block settle"] + 1e-8
    )
    block_offset = np.median(offset[settle], axis=0)
    drift = np.linalg.norm(block[first_second, :2] - block[first_second, :2][0], axis=1)
    summary = {
        "scope": "one native scene diagnostic; no performance samples or success-rate estimate",
        "reference_window_s": reference_range,
        "loaded_window_s": loaded_range,
        "bag_lift_m": float(
            center[(time <= 8), 2].max() - np.median(center[(time <= 2), 2])
        ),
        "bag_target_error_m": float(
            np.linalg.norm(
                np.median(center[reference, :2], axis=0)
                - manifest["scene"]["bag_target_xy"]
            )
        ),
        "reference_height_m": float(np.median(height[reference])),
        "loaded_height_m": float(np.median(height[loaded])),
        "reference_top_centre_above_table_m": float(
            np.median(top[reference]) - TABLE_Z
        ),
        "loaded_top_centre_above_table_m": float(np.median(top[loaded]) - TABLE_Z),
        "central_lowering_m": float(np.median(top[reference]) - np.median(top[loaded])),
        "central_max_material_node_lowering_m": float(
            (
                np.median(states["bag"][reference][:, central_ids, 2], axis=0)
                - np.median(states["bag"][loaded][:, central_ids, 2], axis=0)
            ).max()
        ),
        "central_node_count": int(len(central_ids)),
        "block_bottom_above_table_m": float(np.median(bottom[loaded]) - TABLE_Z),
        "block_to_bag_center_xy_m": float(
            np.linalg.norm(np.median(block[loaded, :2] - center[loaded, :2], axis=0))
        ),
        "block_release_first_second_max_drift_m": float(drift.max()),
        "block_loaded_max_xy_excursion_m": float(
            np.linalg.norm(
                block[loaded, :2] - np.median(block[loaded, :2], axis=0), axis=1
            ).max()
        ),
        "block_final_tilt_deg": float(np.median(tilt[loaded])),
        "block_tilt_definition": "minimum angle between any cube face normal and vertical; cube face changes are not an instability by themselves",
        "block_carry_max_tilt_deg": float(tilt[carry].max()),
        "block_carry_offset_drift_m": float(
            np.linalg.norm(offset[carry] - block_offset, axis=1).max()
        ),
        "volume_ratio_range": [float(volume.min()), float(volume.max())],
        "qpos_max_error": float(np.abs(states["qpos"] - states["target_q"]).max()),
        "max_tcp_error_m": float(max(max(r["tcp_error_m"]) for r in rows)),
        "actual_elapsed_s": float(time[-1]),
        "bag_table_min_surface_clearance_m": float(
            states["bag"][:, :, 2].min() - TABLE_Z
        ),
        "bag_table_max_surface_penetration_m": float(
            max(0.0, TABLE_Z - states["bag"][:, :, 2].min())
        ),
        "bag_footprint_inside_tabletop_all_samples": table_covers_bag,
        "bag_loaded_lowest_surface_clearance_m": float(
            np.median(states["bag"][loaded, :, 2].min(1)) - TABLE_Z
        ),
        "caution": "Before/after central lowering includes continued unloaded settling; it is not a counterfactual load-only dent depth. Block geometry above table does not directly measure contact forces.",
    }
    initial_window = (time >= 0.5 - 1e-8) & (time <= 2 + 1e-8)
    initial_shape = np.median(states["bag"][initial_window], axis=0)
    recovered_shape = np.median(states["bag"][reference], axis=0)
    a = initial_shape - initial_shape.mean(0)
    b = recovered_shape - recovered_shape.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    residual = np.linalg.norm(a @ rotation - b, axis=1)
    summary.update(
        bag_recovery_aligned_rms_m=float(np.sqrt(np.mean(residual**2))),
        bag_recovery_aligned_max_m=float(residual.max()),
        initial_rest_height_m=float(np.median(height[initial_window])),
        block_initial_table_gap_m=float(np.median(bottom[initial_window]) - TABLE_Z),
        plasticity_enabled=manifest["scene"]["bag_plasticity"] is not None,
        gas_law=manifest["scene"]["bag_gas"]["mode"],
    )
    summary["checks"] = {
        "bag_lift_and_placement": summary["bag_lift_m"] >= 0.05
        and summary["bag_target_error_m"] <= 0.02,
        "block_on_bag_projection": summary["block_to_bag_center_xy_m"] <= 0.025,
        "block_not_pressed_to_worktop": summary["block_bottom_above_table_m"] > 0.01,
        "block_first_second_drift_under_10mm": summary[
            "block_release_first_second_max_drift_m"
        ]
        <= 0.01,
        "elastic_recovery_rms_under_1mm": not summary["plasticity_enabled"]
        and summary["bag_recovery_aligned_rms_m"] <= 0.001,
        "block_carry_drift_under_2mm": summary["block_carry_offset_drift_m"] <= 0.002,
        "block_carry_tilt_under_2deg": summary["block_carry_max_tilt_deg"] <= 2.0,
        "block_table_gap_under_2mm": 0 <= summary["block_initial_table_gap_m"] <= 0.002,
        "block_final_window_excursion_under_2mm": summary[
            "block_loaded_max_xy_excursion_m"
        ]
        <= 0.002,
        "bag_table_surface_penetration_under_0_5mm": summary[
            "bag_table_max_surface_penetration_m"
        ]
        <= 0.0005
        and table_covers_bag,
    }
    analysis = run / "analysis"
    analysis.mkdir(exist_ok=True)
    (analysis / "scene_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        analysis / "plot_data.npz",
        time=time,
        bag_height=height,
        top_centre=top,
        block_bottom=bottom,
        block_tilt=tilt,
        volume_ratio=volume,
        bag_center=center,
        block_center=block,
        block_tcp_offset=offset,
    )
    # Track the same upper centreline material nodes; connecting samples does
    # not interpolate the mesh or imply a counterfactual load-only response.
    rest_vertices = mesh["vertices"]
    rest_center = rest_vertices.mean(0)
    section_ids = np.flatnonzero(
        (np.abs(rest_vertices[:, 1] - rest_center[1]) < 1e-5)
        & (rest_vertices[:, 2] > rest_center[2])
    )
    section_ids = section_ids[np.argsort(rest_vertices[section_ids, 0])]
    profiles = [
        np.median(states["bag"][window], axis=0)
        for window in (initial_window, reference, loaded)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, indices, labels in (
        (axes[0], (0, 1), ("Initial settled shape", "After bag placement")),
        (axes[1], (1, 2), ("Before loading", "With rigid load")),
    ):
        for index, label in zip(indices, labels, strict=True):
            profile = profiles[index]
            ax.plot(
                (profile[section_ids, 0] - profile[:, 0].mean()) * 1000,
                (profile[section_ids, 2] - TABLE_Z) * 1000,
                "o-",
                label=label,
                markersize=4,
            )
        ax.set(xlabel="X relative to bag centre (mm)", ylabel="Height above table (mm)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    block_x = np.median(block[loaded, 0] - center[loaded, 0])
    half = manifest["scene"]["block_size_m"] / 2
    axes[1].axvspan(
        (block_x - half) * 1000, (block_x + half) * 1000, color="grey", alpha=0.12
    )
    axes[0].set_title("Elastic recovery after pick and place")
    axes[1].set_title("Local response: sampled upper centreline")
    fig.savefig(analysis / "surface_profiles.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(
        analysis / "surface_profile_data.npz",
        section_ids=section_ids,
        initial_shape=profiles[0],
        recovered_shape=profiles[1],
        loaded_shape=profiles[2],
        reference_window_s=reference_range,
        loaded_window_s=loaded_range,
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(time, (top - TABLE_Z) * 1e3, label="Upper central material nodes")
    axes[0, 0].plot(time, height * 1e3, label="Bag envelope height")
    axes[0, 0].plot(time, (bottom - TABLE_Z) * 1e3, label="Block bottom above worktop")
    axes[0, 0].set_ylabel("Height (mm)")
    axes[0, 1].plot(time, tilt, label="Block tilt")
    axes[0, 1].set_ylabel("Degrees")
    axes[1, 0].plot(
        time,
        np.linalg.norm(block[:, :2] - center[:, :2], axis=1) * 1e3,
        label="Block-to-bag XY distance",
    )
    axes[1, 0].set_ylabel("Distance (mm)")
    axes[1, 1].plot(time, volume * 100, label="Geometric volume / initial rest volume")
    axes[1, 1].set_ylabel("Percent")
    for ax in axes.flat:
        ax.axvspan(*reference_range, color="C2", alpha=0.12)
        ax.axvspan(*loaded_range, color="C1", alpha=0.12)
        ax.axvline(
            timing["block release"], color="grey", ls=":", label="Block released"
        )
        ax.set_xlabel("Actual simulation time (s)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "M05 native scene diagnostic: fixed reference and loaded windows; no performance measurements"
    )
    fig.savefig(analysis / "physical_metrics.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(time, (top - TABLE_Z) * 1000, label="Mean central upper-node height")
    ax.plot(time, height * 1000, label="Bag envelope height")
    ax.axvspan(*reference_range, color="C2", alpha=0.12, label="Unloaded reference")
    ax.axvspan(*loaded_range, color="C1", alpha=0.12, label="Loaded window")
    ax.set_xlim(reference_range[0], duration)
    ax.set_ylim(
        min(
            top[time >= reference_range[0]].min() - TABLE_Z,
            height[time >= reference_range[0]].min(),
        )
        * 1000
        - 1,
        max(
            top[time >= reference_range[0]].max() - TABLE_Z,
            height[time >= reference_range[0]].max(),
        )
        * 1000
        + 1,
    )
    ax.set(
        xlabel="Actual simulation time (s)",
        ylabel="Height (mm)",
        title="M05 central response during the loading stage",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(analysis / "load_response.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
