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
"""Measure M04 recorded grip and one-fold geometry without timing results."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[8]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["main"]


def main() -> None:
    """Save task-specific descriptive metrics and their raw plotting arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    r = args.run
    a = r / "analysis"
    a.mkdir(exist_ok=True)
    c = json.loads((r / "inputs/config.json").read_text())
    commands = np.load(r / "inputs/joint_commands.npz")
    s = np.load(r / "preview/states.npz")
    m = np.load(r / "inputs/cloth_mesh.npz")
    t = s["time"]
    p = s["particles"]
    v = s["velocities"]
    tcp = s["tcp"]
    hands = s["hand_poses"]
    hem = m["hem_ids"]
    ends = {name: end for end, name in c["motion"]["phases"]}
    index = lambda name: int(np.argmin(np.abs(t - ends[name])))
    stable = index("pre-roll")
    hold = index("grasp settle")
    release = index(
        "hold folded hem" if "hold folded hem" in ends else "lay folded hem"
    )
    ids = np.array(
        [
            int(np.argmin(np.linalg.norm(p[hold] - target, axis=1)))
            for target in tcp[hold]
        ]
    )
    # Hold-relative material positions measured in the actual gripper frame.
    tracked = p[:, ids, :]
    local = np.einsum(
        "tsji,tsj->tsi",
        hands[:, [0, 3]][:, :, :3, :3],
        tracked - hands[:, [0, 3]][:, :, :3, 3],
    )
    drift = np.linalg.norm(local - local[hold], axis=2)
    center = p[:, hem].mean(1)
    from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_common import (
        supports,
    )

    depth = np.zeros(len(t))
    _, table_center, table_size = supports(c)[0]
    table_lo = np.array(table_center) - np.array(table_size) / 2
    table_hi = np.array(table_center) + np.array(table_size) / 2
    initial_xy_margin = float(
        np.minimum(
            p[stable, :, :2] - table_lo[:2], table_hi[:2] - p[stable, :, :2]
        ).min()
    )
    initial_on_table = bool(
        initial_xy_margin >= 0
        and p[stable, :, 2].min() >= c["support"]["table_z"] - 0.0005
    )
    for _, support_center, size in supports(c):
        lo = np.array(support_center) - np.array(size) / 2
        hi = np.array(support_center) + np.array(size) / 2
        inside_depth = np.minimum(p - lo, hi - p).min(2)
        depth = np.maximum(depth, np.maximum(inside_depth, 0).max(1))
    speed = np.sqrt((v * v).sum(2).mean(1))
    far_end = p[:, m["neckline_ids"]].mean(1)
    triangles = m["triangles"]
    edges = np.unique(
        np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0
    )
    rest_lengths = np.linalg.norm(p[0, edges[:, 0]] - p[0, edges[:, 1]], axis=1)
    edges = edges[rest_lengths > 1e-6]
    rest_lengths = rest_lengths[rest_lengths > 1e-6]
    edge_ratio_p95 = np.array(
        [
            np.percentile(
                np.linalg.norm(row[edges[:, 0]] - row[edges[:, 1]], axis=1)
                / rest_lengths,
                95,
            )
            for row in p
        ]
    )
    lift_end = index("lift hem")
    far_end_slide = float(far_end[hold, 0] - far_end[lift_end, 0])
    lift_edge_ratio = float(edge_ratio_p95[hold : lift_end + 1].max())
    from scipy.signal import butter, sosfiltfilt

    # Separate rapid positional oscillations from the slow folding motion.
    # This is a sampled diagnostic, not a cloth-quality or force criterion.
    sample_dt = float(np.median(np.diff(t)))
    highpass = sosfiltfilt(
        butter(4, 5.0, btype="highpass", fs=1 / sample_dt, output="sos"),
        p,
        axis=0,
    )
    highpass_rms = np.sqrt(np.square(highpass).sum(2).mean(1))
    low_highpass = sosfiltfilt(
        butter(4, 2.0, btype="highpass", fs=1 / sample_dt, output="sos"),
        p,
        axis=0,
    )
    low_highpass_rms = np.sqrt(np.square(low_highpass).sum(2).mean(1))
    jitter = {}
    for label, start, end in (
        ("lift", "grasp settle", "lift hem"),
        ("fold", "lift hem", "lay folded hem"),
        ("hold", "lay folded hem", "hold folded hem"),
        ("released", "vertical retreat", "observe"),
    ):
        mask = (t >= ends[start]) & (t <= ends[end])
        jitter[label] = dict(
            position_above5Hz_rms_mm=float(
                np.sqrt(np.square(highpass_rms[mask]).mean()) * 1000
            ),
            frame_rms_p95_mm=float(np.percentile(highpass_rms[mask], 95) * 1000),
            position_above2Hz_rms_mm=float(
                np.sqrt(np.square(low_highpass_rms[mask]).mean()) * 1000
            ),
            node_above5Hz_amplitude_p95_mm=float(
                np.percentile(np.sqrt(np.square(highpass[mask]).sum(2).mean(0)), 95)
                * 1000
            ),
        )
    jitter_report = dict(
        method="fourth-order zero-phase Butterworth highpass at 5Hz and 2Hz; all-node three-dimensional position RMS; per-node 5Hz amplitude also reported",
        sample_rate_hz=1 / sample_dt,
        scope="rapid sampled position changes; not a universal visual-quality score",
        phases=jitter,
    )
    (a / "jitter_checks.json").write_text(json.dumps(jitter_report, indent=2) + "\n")
    target = np.array(
        [
            c["motion"]["place_x"],
            (
                c["motion"]["place_y"]
                - c["motion"].get("right_grasp_y", c["motion"]["place_y"])
            )
            / 2,
        ]
    )
    final_error = float(np.linalg.norm(center[-1, :2] - target))
    report = dict(
        scope="descriptive scene preview; no formal performance or force estimates",
        max_actual_tcp_error_m=float(
            np.linalg.norm(tcp - commands["tcp_target"][: len(t)], axis=2).max()
        ),
        max_actual_named_qpos_error=float(
            abs(s["qpos"] - commands["targets"][: len(t)]).max()
        ),
        finite=bool(np.isfinite(p).all() and np.isfinite(v).all()),
        hem_ids=hem.tolist(),
        grip_marker_ids=ids.tolist(),
        max_grip_relative_drift_m=drift[hold : release + 1].max(0).tolist(),
        hem_center_initial_m=center[stable].tolist(),
        hem_center_final_m=center[-1].tolist(),
        final_hem_to_release_center_xy_error_m=final_error,
        release_center_xy_m=target.tolist(),
        far_end_initial_m=far_end[hold].tolist(),
        far_end_after_lift_m=far_end[lift_end].tolist(),
        lift_far_end_slide_toward_gripper_m=far_end_slide,
        max_lift_edge_length_ratio_p95=lift_edge_ratio,
        max_lift_far_end_centroid_height_m=float(
            far_end[hold : lift_end + 1, 2].max() - c["support"]["table_z"]
        ),
        candidate_checks=dict(
            initial_on_table=initial_on_table,
            grip=bool(
                drift[hold : release + 1].max()
                <= c["candidate_criteria"]["max_grip_relative_drift_m"]
            ),
            support=bool(
                depth.max() <= c["candidate_criteria"]["max_support_node_penetration_m"]
            ),
            hem_near_release_center=bool(
                final_error <= c["candidate_criteria"]["max_final_hem_xy_error_m"]
            ),
            far_end_slides_on_lift=bool(
                far_end_slide
                >= c["candidate_criteria"].get("min_lift_far_end_slide_m", 0.02)
            ),
            lift_strain=bool(
                lift_edge_ratio
                <= c["candidate_criteria"].get("max_lift_edge_length_ratio_p95", 1.08)
            ),
        ),
        hem_x_displacement_m=float(center[-1, 0] - center[stable, 0]),
        initial_table_projection_margin_m=initial_xy_margin,
        initial_settled_cloth_bounds_m=[
            p[stable].min(0).tolist(),
            p[stable].max(0).tolist(),
        ],
        max_hem_lift_m=float(center[:, 2].max() - center[stable, 2]),
        max_table_node_penetration_m=float(depth.max()),
        support_node_check_method="authored rectangular tabletop and leg AABBs; exact cloth-face mesh intersections checked separately with FCL",
        initial_settled_rms_speed_m_s=float(speed[stable]),
        final_rms_speed_m_s=float(speed[-1]),
    )
    (a / "physical_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        a / "plot_data.npz",
        time=t,
        hem_center=center,
        grip_drift=drift,
        table_depth=depth,
        rms_speed=speed,
        highpass_rms_m=highpass_rms,
        highpass_2Hz_rms_m=low_highpass_rms,
        far_end_center=far_end,
        edge_length_ratio_p95=edge_ratio_p95,
        grip_marker_ids=ids,
    )
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    axs[0, 0].plot(t, center[:, 0], label="hem X")
    axs[0, 0].plot(t, center[:, 2], label="hem Z")
    axs[0, 0].legend()
    axs[0, 1].plot(t, drift * 1000)
    axs[0, 1].set_ylim(0, max(20, float(drift[hold : release + 1].max() * 1100)))
    axs[0, 1].axvspan(t[hold], t[release], alpha=0.12)
    axs[0, 1].set_title("Marker drift in grip frame (mm); shaded=hold")
    axs[1, 0].plot(t, depth * 1000)
    axs[1, 0].set_title("Maximum support-node penetration (mm)")
    axs[1, 1].plot(t, speed * 1000)
    axs[1, 1].set_title("Cloth node RMS speed (mm/s)")
    for ax in axs.flat:
        ax.set_xlabel("Actual simulation time (s)")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(a / "physical_response.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, highpass_rms * 1000)
    ax.axvspan(ends["lift hem"], ends["hold folded hem"], alpha=0.12)
    ax.set(xlabel="Actual simulation time (s)", ylabel="Node RMS above 5Hz (mm)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(a / "jitter_response.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
    axs[0].plot(t, (far_end[hold, 0] - far_end[:, 0]) * 1000)
    axs[0].set_ylabel("Far-end motion toward grippers (mm)")
    axs[1].plot(t, (edge_ratio_p95 - 1) * 100)
    axs[1].set_ylabel("Triangle-edge extension, 95th percentile (%)")
    for ax in axs:
        ax.axvspan(t[hold], t[lift_end], alpha=0.12)
        ax.set_xlabel("Actual simulation time (s)")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(a / "sliding_response.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
