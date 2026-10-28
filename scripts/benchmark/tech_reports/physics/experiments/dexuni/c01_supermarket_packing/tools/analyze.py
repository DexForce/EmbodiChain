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
"""Audit C01 grip drift, bag motion and packing from saved physical states.

Run: python analyze.py --run <run> [--baseline <old run> ...]
No simulation steps or source/input changes are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["main"]


def _analyze(run: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read one complete recording and compute descriptive physical metrics."""
    manifest = json.loads((run / "manifest.json").read_text())
    with np.load(run / "preview/states.npz") as data:
        states = {k: data[k] for k in data.files}
    with np.load(run / "inputs/goods_meshes.npz") as data:
        mesh = {k: data[k] for k in data.files}
    with np.load(run / "inputs/joint_commands.npz") as data:
        commands = {k: data[k] for k in data.files}
    asset = json.loads((run / "inputs/shopping_bag.json").read_text())
    phases = dict((name, end) for end, name in manifest["scene"]["phases"])
    time = states["time"]
    dt = np.diff(time)
    assert np.allclose(dt, 1 / 60, rtol=0, atol=1e-10)
    velocity = np.diff(states["bag"], axis=0) / dt[:, None, None]
    speed = np.linalg.norm(velocity, axis=-1)
    relative_speed = np.linalg.norm(velocity - velocity.mean(axis=1)[:, None], axis=-1)
    arrays = dict(
        time=time,
        bag_speed_p95=np.r_[0, np.percentile(speed, 95, axis=1)],
        bag_relative_speed_p95=np.r_[0, np.percentile(relative_speed, 95, axis=1)],
    )
    report: dict[str, Any] = dict(
        source_run=str(run.resolve()),
        source_states_sha256=hashlib.sha256(
            (run / "preview/states.npz").read_bytes()
        ).hexdigest(),
        full_finite_states=all(
            len(v) == len(time) and np.isfinite(v).all() for v in states.values()
        ),
        samples=len(time),
        recorded_duration_s=float(time[-1]),
        actual_duration_s=manifest["effective"].get("sim_time"),
        actual_outer_steps=manifest["effective"].get("step_index"),
        graph_status=manifest["effective"]["cuda_graph_status"],
        max_joint_replay_error=(
            float(np.abs(states["qpos"] - commands["targets"]).max())
            if manifest.get("joint_state_order") == commands["joint_names"].tolist()
            else None
        ),
        max_tcp_replay_error_m=float(
            np.linalg.norm(states["tcp"] - commands["tcp_baked"], axis=-1).max()
        ),
        goods={},
        windows={},
    )
    goods = manifest["scene"].get("goods", ["block", "bread", "pouch"])
    sides = {
        name: {"left": 0, "right": 1}[
            manifest["scene"]
            .get("good_arms", {})
            .get(name, "left" if name == "block" else "right")
        ]
        for name in goods
    }
    for name in goods:
        points = states[name] if name != "block" else None
        centres = (
            points.mean(axis=1)
            if points is not None
            else states["block_pose"][:, :3, 3]
        )
        start = round(phases[f"{name} settle"] * 60)
        stop = round(phases[f"{name} lower"] * 60)
        side = sides[name]
        relative = centres - states["tcp"][:, side]
        drift = np.linalg.norm(relative - relative[start], axis=-1)
        arrays[f"{name}_centre"] = centres
        arrays[f"{name}_grip_drift"] = drift
        report["goods"][name] = dict(
            arm="left" if side == 0 else "right",
            grip_window_s=[float(time[start]), float(time[stop])],
            max_centre_to_tcp_drift_during_hold_m=float(drift[start : stop + 1].max()),
            max_centre_height_above_table_m=float(
                centres[start : stop + 1, 2].max() - manifest["scene"]["table_z"]
            ),
            final_centre=centres[-1].tolist(),
        )
    tets = mesh["bread_tetrahedra"]
    rest = mesh["bread_vertices"][tets]
    rest_det = np.einsum(
        "ij,ij->i",
        rest[:, 1] - rest[:, 0],
        np.cross(rest[:, 2] - rest[:, 0], rest[:, 3] - rest[:, 0]),
    )
    assert (abs(rest_det) > 1e-15).all(), "Degenerate rest tetrahedra"
    minimum_ratio, inverted = float("inf"), 0
    for first in range(0, len(time), 64):
        nodes = states["bread"][first : first + 64].astype(np.float64)[:, tets]
        ratio = (
            np.einsum(
                "...i,...i->...",
                nodes[:, :, 1] - nodes[:, :, 0],
                np.cross(
                    nodes[:, :, 2] - nodes[:, :, 0],
                    nodes[:, :, 3] - nodes[:, :, 0],
                ),
            )
            / rest_det
        )
        minimum_ratio = min(minimum_ratio, float(ratio.min()))
        inverted += int((ratio <= 0).sum())
    report["goods"]["bread"].update(
        minimum_signed_tet_volume_ratio=minimum_ratio,
        inverted_tet_pose_count=inverted,
    )
    centred = states["pouch"].astype(np.float64)
    centred -= centred.mean(axis=1)[:, None]
    triangles = centred[:, mesh["pouch_triangles"]]
    volume = (
        abs(
            np.einsum(
                "fij,fij->fi",
                triangles[:, :, 0],
                np.cross(triangles[:, :, 1], triangles[:, :, 2]),
            ).sum(axis=1)
        )
        / 6
    )
    arrays["pouch_volume_ratio"] = volume / float(mesh["pouch_rest_volume"])
    start = round(phases["pouch settle"] * 60)
    stop = round(phases["pouch lower"] * 60)
    report["goods"]["pouch"]["minimum_volume_ratio_during_hold"] = float(
        arrays["pouch_volume_ratio"][start : stop + 1].min()
    )
    report["goods"]["pouch"]["volume_ratio_at_settle"] = float(
        arrays["pouch_volume_ratio"][start]
    )
    if manifest["scene"].get("pouch_grasp_y_offset_m", 0.0):
        authored = mesh["pouch_vertices"]
        positive_edge = manifest["scene"]["pouch_grasp_y_offset_m"] > 0
        edge = (
            authored[:, 1] > authored[:, 1].max() - 1e-6
            if positive_edge
            else authored[:, 1] < authored[:, 1].min() + 1e-6
        )
        marker_ids = np.flatnonzero(
            edge & (abs(authored[:, 0] - authored[:, 0].mean()) < 0.012)
        )
        marker = states["pouch"][:, marker_ids].mean(axis=1)
        relative = marker - states["tcp"][:, sides["pouch"]]
        drift = np.linalg.norm(relative - relative[start], axis=-1)
        arrays["pouch_seam_grip_drift"] = drift
        report["goods"]["pouch"].update(
            seam_marker_node_ids=marker_ids.tolist(),
            seam_marker_definition=f"Central six authored nodes on the {'+Y' if positive_edge else '-Y'} sealed edge; fixed material node IDs, no contact filtering",
            max_seam_marker_to_tcp_drift_during_hold_m=float(
                drift[start : stop + 1].max()
            ),
        )
    handles = np.asarray(asset["vertices"])[:, 2] > 0.30
    arrays["handle_mean_z"] = states["bag"][:, handles, 2].mean(axis=1)
    arrays["bag_lowest_z"] = states["bag"][:, :, 2].min(axis=1)
    release = phases["pouch release"]
    lower = phases["pouch lower"]
    report["pouch_release_end_s"] = release
    for name, a, b in (
        ("after_bread", phases["bread release"], phases["bread release"] + 1.6),
        ("before_pouch", lower - 0.5, lower),
        ("pouch_impact", lower, release + 0.4),
        ("post_pouch", release + 0.4, release + 1.9),
        ("matched_post_release", release + 2.0, min(release + 4.0, float(time[-1]))),
        ("final", float(time[-1]) - 1.0, float(time[-1])),
    ):
        mask = (time[1:] >= a) & (time[1:] < b)
        positions = states["bag"][(time >= a) & (time <= b)]
        report["windows"][name] = dict(
            start_s=a,
            end_s=b,
            bag_node_speed_p95_m_s=float(np.percentile(speed[mask], 95)),
            bag_node_speed_max_m_s=float(speed[mask].max()),
            bag_relative_speed_p95_m_s=float(np.percentile(relative_speed[mask], 95)),
            bag_centre_peak_to_peak_m=np.ptp(positions.mean(axis=1), axis=0).tolist(),
            bag_node_excursion_from_window_start_p95_m=float(
                np.percentile(np.linalg.norm(positions - positions[0], axis=-1), 95)
            ),
        )
    report["bag"] = dict(
        final_handle_mean_z_m=float(arrays["handle_mean_z"][-1]),
        minimum_handle_mean_z_m=float(arrays["handle_mean_z"].min()),
        final_lowest_z_m=float(arrays["bag_lowest_z"][-1]),
        final_speed_p95_m_s=float(arrays["bag_speed_p95"][-1]),
    )
    metrics = json.loads((run / "preview/metrics.json").read_text())
    report["final_centre_in_bag"] = {n: metrics[-1][f"{n}_in_bag"] for n in goods}
    report["packing_scope"] = (
        "Frozen mouth/bottom marker bounds and good centres only; not a full-surface containment or continuous-collision proof"
    )
    return report, arrays


def main() -> None:
    """Write metrics, raw plot arrays and a comparison without changing recordings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, action="append", default=[])
    args = parser.parse_args()
    output = args.run / "analysis"
    output.mkdir(exist_ok=True)
    cases = []
    for index, run in enumerate([*args.baseline, args.run]):
        report, arrays = _analyze(run)
        label = "current" if run == args.run else f"baseline{index+1}"
        (output / f"{label}_metrics.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        np.savez_compressed(output / f"{label}_plot_data.npz", **arrays)
        cases.append((label, report, arrays))
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for label, report, a in cases:
        lo, hi = report["goods"]["bread"]["grip_window_s"]
        held = (a["time"] >= lo) & (a["time"] <= hi)
        axes[0].plot(a["time"][held], 1000 * a["bread_grip_drift"][held], label=label)
        axes[1].plot(a["time"], a["bag_speed_p95"], label=label)
        axes[2].plot(a["time"], a["handle_mean_z"], label=label)
    axes[0].set_ylim(0, 50)
    axes[0].set_ylabel("Bread centre/TCP drift (mm)")
    axes[1].set_ylabel("Bag node speed p95 (m/s)")
    axes[2].set_ylabel("Handle mean height (m)")
    axes[2].set_xlabel("Actual trajectory time (s)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
        ax.axvline(cases[-1][1]["pouch_release_end_s"], color="gray", ls=":")
    fig.tight_layout()
    fig.savefig(output / "repair_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for label, report, a in cases:
        relative_time = a["time"] - report["pouch_release_end_s"]
        axes[0].plot(relative_time, a["bag_speed_p95"], label=label)
        axes[1].plot(relative_time, a["pouch_volume_ratio"], label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-5, 1.0)
    axes[0].set_ylabel("Bag node speed p95 (m/s)")
    axes[1].set_ylabel("Milk volume / rest volume")
    axes[1].set_xlabel("Time since milk release ends (s)")
    release = cases[-1][1]["pouch_release_end_s"]
    axes[1].set_xlim(-2.5, cases[-1][1]["recorded_duration_s"] - release)
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
        ax.axvline(0.0, color="gray", ls=":")
    fig.tight_layout()
    fig.savefig(output / "tail_settling.png", dpi=160)
    plt.close(fig)
    print(json.dumps(cases[-1][1], indent=2))


if __name__ == "__main__":
    main()
