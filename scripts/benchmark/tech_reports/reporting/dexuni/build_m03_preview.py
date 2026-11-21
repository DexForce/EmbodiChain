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
"""Rebuild M03 native preview geometry diagnostics without running simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["main"]
OUTPUT = (
    Path(__file__).resolve().parents[2].parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/m03_four_layer_cloth_pick_place"
)


def write_json(path: Path, value: Any) -> None:
    """Write detached report data."""
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=lambda x: np.asarray(x).tolist(),
        )
        + "\n"
    )


def quality(
    config: dict[str, Any], states: dict[str, np.ndarray], mesh: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Measure stationary flutter and the same material far edge in each layer."""
    time, particles = states["time"], states["particles"]
    timing = config["motion"]["timing"]
    hold = (time >= timing["lift"] + 0.25) & (time <= timing["hold"] - 0.25)
    far = np.flatnonzero(
        mesh["initial_positions"][0, :, 0]
        > mesh["initial_positions"][0, :, 0].max() - 1e-5
    )
    near = np.flatnonzero(
        mesh["initial_positions"][0, :, 0]
        < mesh["initial_positions"][0, :, 0].min() + 1e-5
    )
    final_far = particles[-1][:, far]
    far_center = final_far.mean(1)
    far_distance = np.linalg.norm(
        far_center[:, None, :2] - far_center[None, :, :2], axis=2
    )
    reference = int(np.argmin(np.abs(time - timing["settle"])))
    rest = mesh["initial_positions"][0]
    centerline = np.flatnonzero(np.abs(rest[:, 1]) < 1e-5)
    centerline = centerline[np.argsort(rest[centerline, 0])]
    settle = (time >= timing["settle"] - 0.5) & (time <= timing["settle"])
    return dict(
        scope="single native scene geometry diagnostics; no performance samples",
        hold_window=[timing["lift"] + 0.25, timing["hold"] - 0.25],
        hold_speed_p95_m_s=np.quantile(
            np.linalg.norm(states["velocities"][hold], axis=-1), 0.95, axis=(0, 2)
        ),
        hold_second_difference_p95_m=np.quantile(
            np.linalg.norm(np.diff(particles[hold], n=2, axis=0), axis=-1),
            0.95,
            axis=(0, 2),
        ),
        sample_dt_s=config["solver"]["dt"],
        far_edge_node_ids=far,
        near_edge_node_ids=near,
        far_edge_centroids=far_center,
        far_edge_centroid_xy_spread_m=float(far_distance.max()),
        far_edge_x_range_m=np.ptp(final_far[:, :, 0], axis=1),
        final_layer_z_range_m=np.ptp(particles[-1, :, :, 2], axis=1),
        final_centerline_reversed_segments=(
            np.diff(particles[-1][:, centerline, 0], axis=1) < -0.001
        ).sum(1),
        final_centerline_scope="fixed central material line only; not a full mesh fold detector",
        far_edge_to_bottom_xy_p95_m=np.quantile(
            np.linalg.norm(final_far[1:, :, :2] - final_far[:1, :, :2], axis=2),
            0.95,
            axis=1,
        ),
        settle_speed_p95_m_s=np.quantile(
            np.linalg.norm(states["velocities"][settle], axis=-1), 0.95, axis=(0, 2)
        ),
        settled_same_node_z_gap_min_m=np.diff(
            particles[reference, :, :, 2], axis=0
        ).min(1),
        settled_gap_scope="same rest-node pairs only, not a full triangle-intersection proof",
    )


def plot(
    output: Path,
    config: dict[str, Any],
    states: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> None:
    """Save actual per-layer curves and final near/far material edges."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    time, particles = states["time"], states["particles"]
    colors = config["cloth"]["colors"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    center = particles.mean(2)
    minimum = particles[..., 2].min(2)
    for k, color in enumerate(colors):
        label = f"layer {k+1} ({'bottom' if k == 0 else 'top' if k == 3 else 'middle'})"
        axes[0, 0].plot(
            time,
            (minimum[:, k] - config["support"]["pickup_z"]) * 1000,
            color=color,
            label=label,
        )
        axes[0, 1].plot(time, center[:, k, 0] * 1000, color=color)
        axes[1, 0].plot(
            time,
            particles[:, k, metrics["far_edge_node_ids"]].mean(1)[:, 0] * 1000,
            color=color,
        )
        for group, style in (
            (metrics["near_edge_node_ids"], "--"),
            (metrics["far_edge_node_ids"], "-"),
        ):
            edge = particles[-1, k, group]
            axes[1, 1].plot(
                edge[:, 0] * 1000, edge[:, 1] * 1000, style, color=color, lw=2
            )
        axes[1, 1].scatter(
            center[-1, k, 0] * 1000, center[-1, k, 1] * 1000, color=color, s=25
        )
    axes[0, 0].axhline(50, color="gray", ls="--", label="50 mm candidate clearance")
    axes[0, 0].set_title("Minimum node clearance from pickup surface")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(580, color="gray", ls="--")
    axes[0, 1].axhspan(560, 600, color="gray", alpha=0.1)
    axes[0, 1].set_title("Actual cloth centroid X; target 580 +/-20 mm")
    axes[1, 0].set_title("Far-edge centroid X (fixed rest-node IDs)")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.axvspan(
            config["motion"]["timing"]["lift"],
            config["motion"]["timing"]["hold"],
            color="green",
            alpha=0.08,
        )
        ax.set_xlabel("simulation time (s)")
        ax.set_ylabel("mm")
        ax.grid(alpha=0.2)
    cx, cy = np.asarray(config["support"]["target_center_xy"]) * 1000
    size = config["support"]["target_size"] * 1000
    axes[1, 1].add_patch(
        Rectangle(
            (cx - size / 2, cy - size / 2), size, size, fill=False, edgecolor="teal"
        )
    )
    axes[1, 1].set_title("Final edges: solid far, dashed near; dots = centroids")
    axes[1, 1].set(
        xlabel="X (mm)",
        ylabel="Y (mm)",
        xlim=(390, 780),
        ylim=(-190, 190),
        aspect="equal",
    )
    axes[1, 1].grid(alpha=0.2)
    fig.suptitle("M03 DexSim one-way / surface-fast | native preview only")
    fig.savefig(output / "preview/layer_curves.png", dpi=160)
    plt.close(fig)


def report(
    output: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Write a native-only note without overwriting the cross-platform report."""
    observation = manifest["observations"]
    text = "# M03 原生场景诊断\n\n"
    text += f"[视频](dexsim.mp4) · [实际状态](states.npz) · [几何诊断](quality.json)\n\n实际时长 {config['motion']['duration']:g} s；one_way / kinematic，surface-fast，基础{manifest['effective']['vbd']['iterations']}次、{config['solver']['num_substeps']}子步。无绑定或 pin，未采集性能样本，待用户审阅。\n\n"
    text += f"四层抬升：{observation['all_four_clear']}；放置候选线：{observation['all_four_candidate_placement']}。\n\n"
    text += f"各层高度极差 mm：{[round(float(x)*1000, 2) for x in metrics['final_layer_z_range_m']]}；中央材料线反向段：{metrics['final_centerline_reversed_segments'].tolist()}。这是局部几何描述，不是完整折叠或交叉证明。\n\n跨平台记录见 [REPORT](../REPORT.md)。\n"
    (output / "preview/native_report.md").write_text(text)


def main() -> None:
    """Validate the current completed run and rebuild its local report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    output = args.out_dir.resolve()
    config = json.loads((output / "inputs/config.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "completed" or not manifest["full_trajectory"]:
        raise ValueError("M03 diagnostics need a completed full episode")
    with np.load(output / "preview/states.npz") as data:
        states = {key: data[key] for key in data.files}
    with np.load(output / "inputs/cloth_mesh.npz") as data:
        mesh = {key: data[key] for key in data.files}
    metrics = quality(config, states, mesh)
    write_json(output / "preview/quality.json", metrics)
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "hold_speed_p95_m_s",
                    "hold_second_difference_p95_m",
                    "far_edge_centroid_xy_spread_m",
                    "far_edge_x_range_m",
                )
            },
            default=lambda x: np.asarray(x).tolist(),
            indent=2,
        )
    )
    if args.diagnostics_only:
        return
    plot(output, config, states, metrics)
    from PIL import Image

    times = [
        0,
        config["motion"]["timing"]["settle"],
        config["motion"]["timing"]["close"],
        config["motion"]["timing"]["lift"],
        config["motion"]["timing"]["hold"],
        config["motion"]["timing"]["land"],
        config["motion"]["timing"]["lay"],
        config["motion"]["timing"]["release"],
        config["motion"]["duration"],
    ]
    montage = Image.new("RGB", (1680, 720), "white")
    for i, t in enumerate(times):
        with Image.open(output / f"preview/stage_{t:05.2f}.jpg") as frame:
            montage.paste(frame.resize((560, 240)), ((i % 3) * 560, (i // 3) * 240))
    montage.save(output / "preview/stages.jpg")
    report(output, config, manifest, metrics)
    write_json(
        output / "preview/report_source.json",
        dict(
            source=str(Path(__file__)),
            sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )


if __name__ == "__main__":
    main()
