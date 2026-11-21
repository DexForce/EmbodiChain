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
"""Legacy S01T pilot analysis; not a formal v2 report builder.

Run: DEXUNI_ALLOW_PILOT=1 <numpy-python> scripts/benchmark/tech_reports/reporting/dexuni/s01t/analyze.py
    --root <run-root> --output-root <new-rebuild-root>
    --platforms embodichain,isaaclab

Raw inputs are read-only; generated files go to the separate output root.
Known qualification gaps: latest-run selection, failure denominator,
cold-start averaging and index-based video alignment. See SCRIPT_CONTRACT.md.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO_ROOT))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.paths import (
    PLATFORMS as ADAPTERS,
    input_dir,
    output_dir,
    require_pilot,
)

SUBTASK_DIR = output_dir("s01t")
INPUTS = input_dir("s01t")
RAW = SUBTASK_DIR / "raw_data"
RESULTS = SUBTASK_DIR / "results"
VIDEOS = SUBTASK_DIR / "videos"
PLATFORMS = ("dexsim", "isaaclab")
COLORS = {"dexsim": "tab:blue", "embodichain": "tab:green", "isaaclab": "tab:orange"}
LABELS = {
    "dexsim": "DexSim  (DexUni two-way)",
    "embodichain": "EmbodiChain  (DexUni two-way)",
    "isaaclab": "Isaac Lab 3.0  (MuJoCo-Warp + VBD coupler)",
}
NAMES = {"dexsim": "DexSim", "embodichain": "EmbodiChain", "isaaclab": "Isaac Lab"}
PHASES = ("grip", "rest", "pull", "hold", "return", "release", "recover", "done")
SOFT_LABEL = 6
PRIMARY = 0
PAD_HALF = (0.00875, 0.00925)  # fingertip pad half extents along X and Z


def load_run(path: Path) -> dict:
    run = {
        "path": path,
        "manifest": json.loads((path / "manifest.json").read_text()),
        "config": json.loads((path / "config.json").read_text()),
    }
    timing = np.genfromtxt(
        path / "timing.csv", delimiter=",", names=True, dtype=None, encoding=None
    )
    run["step_seconds"] = np.atleast_1d(np.asarray(timing["step_seconds"], dtype=float))
    # Decompress once: NpzFile re-reads an array on every key access.
    run["states"] = (
        dict(np.load(path / "states.npz")) if (path / "states.npz").exists() else None
    )
    return run


def run_key(run) -> tuple:
    m = run["manifest"]
    return (
        m["platform"],
        m.get("scenario", "task"),
        m["state"]["id"],
        m["variant"],
        m["kind"],
        m["repeat"],
        m["requested"]["vbd_iterations"],
    )


def discover():
    runs, superseded = {}, defaultdict(list)
    for platform in PLATFORMS:
        for path in sorted((RAW / platform).glob("s01t_*")):
            if not (path / "manifest.json").exists():
                continue
            run = load_run(path)
            if run["manifest"].get("status") != "completed":
                superseded["not_completed"].append(path.name)
                continue
            key = run_key(run)
            if key in runs:
                older = runs[key]
                if older["manifest"]["started_utc"] > run["manifest"]["started_utc"]:
                    older, run = run, older
                superseded[str(key)].append(older["path"].name)
            runs[key] = run
    return runs, dict(superseded)


# --------------------------------------------------------------------------
# Task metrics
# --------------------------------------------------------------------------


def timing_row(run) -> dict:
    steps = run["step_seconds"]
    m = run["manifest"]
    return {
        "frames": len(steps),
        "step_ms_mean": 1e3 * steps.mean(),
        "step_ms_p50": 1e3 * np.median(steps),
        "step_ms_p95": 1e3 * np.percentile(steps, 95),
        "realtime_factor": (1.0 / 60.0) / steps.mean(),
        "init_s": m.get("init_seconds"),
        "gpu_peak_mib": m.get("gpu_memory", {}).get("gpu_process_peak_mib"),
    }


def task_metrics(run) -> dict:
    m, config = run["manifest"], run["config"]
    eff = m.get("effective", {})
    row = {
        "run_id": m["run_id"],
        "platform": m["platform"],
        "state": m["state"]["id"],
        "variant": m["variant"],
        "kind": m["kind"],
        "repeat": m["repeat"],
        "num_substeps": eff.get("num_substeps"),
        "cuda_graph": eff.get("cuda_graph_status"),
        "vbd_iterations": m["requested"]["vbd_iterations"],
        "target_reached": m.get("controller", {}).get("target_reached"),
        **timing_row(run),
    }
    data = run["states"]
    if data is None:
        return row
    t, phase = data["time"], data["phase"]
    particles = data["particles"]
    ctrl = m["controller"]
    l0, events = ctrl["l0"], ctrl["events"]
    target = m["target_strain"]
    lengths = data["line_lengths"]
    strain = lengths[:, PRIMARY] / l0 - 1.0
    hold = phase == PHASES.index("hold")
    hold_tail = hold & (t >= t[hold].max() - 0.8 + 1e-9)
    hold_end = int(np.flatnonzero(hold)[-1])
    ref = (t >= config["measurement"]["reference_window"][0] - 1e-9) & (
        t <= config["measurement"]["reference_window"][1] + 1e-9
    )
    ref_lengths = lengths[ref].mean(axis=0)
    line_strain_hold = lengths[hold_tail].mean(axis=0) / ref_lengths - 1.0
    recovery_t = events.get("recover", np.nan) + 2.0
    recovery_i = int(np.argmin(np.abs(t - recovery_t)))
    # End drift. Clamp end: nodes inside the jaw at rest, mean X change from
    # the reference window to the end of hold (clamp is static). Grip end:
    # nodes whose rest position lies inside the pad box around the grip
    # origin (same node set on both platforms), mean position change
    # relative to the TCP (fingertip midpoint).
    rest = data["rest_particles"]
    clamp_ids = np.flatnonzero(rest[:, 0] <= config["clamp"]["x_range"][1])
    ref_i = np.flatnonzero(ref)
    ref_nodes = particles[ref_i].mean(axis=0)
    tcp_ref = data["tcp"][ref_i].mean(axis=0)
    rel = ref_nodes - tcp_ref
    grip = config["grip"]
    grip_ids = np.flatnonzero(
        (np.abs(rest[:, 0] - grip["tcp_x"]) <= PAD_HALF[0])
        & (np.abs(rest[:, 2] - grip["tcp_z"]) <= PAD_HALF[1])
    )
    clamp_drift = (
        particles[hold_end, clamp_ids, 0].mean() - ref_nodes[clamp_ids, 0].mean()
    )
    grip_rel_end = particles[hold_end, grip_ids].mean(axis=0) - data["tcp"][hold_end]
    grip_drift = np.linalg.norm(grip_rel_end - rel[grip_ids].mean(axis=0))
    table_z = config["scene"]["table_z"]
    base_top = config["clamp"]["base_top_z"]
    jaw_bottom = config["clamp"]["jaw_bottom_z"]
    in_clamp = (particles[:, :, 0] >= config["clamp"]["x_range"][0]) & (
        particles[:, :, 0] <= config["clamp"]["x_range"][1]
    )
    z = particles[:, :, 2]
    clamp_pen = max(
        0.0,
        float(np.max(np.where(in_clamp, base_top - z, -1))),
        float(np.max(np.where(in_clamp, z - jaw_bottom, -1))),
    )
    tcp_cmd_x = data["tcp_cmd_x_offset"]
    origin_x = config["grip"]["tcp_x"]
    acc = config["acceptance"]
    row.update(
        {
            "l0_mm": 1e3 * l0,
            "hold_strain_pct": 100 * strain[hold_tail].mean(),
            "hold_strain_min_pct": 100 * strain[hold_tail].min(),
            "hold_strain_max_pct": 100 * strain[hold_tail].max(),
            "strain_error_pp": 100 * (strain[hold_tail].mean() - target),
            "peak_strain_pct": 100 * strain.max(),
            "hold_strain_lines_pct": ";".join(
                f"{100 * v:.3f}" for v in line_strain_hold
            ),
            "hold_strain_lines_spread_pp": 100 * float(np.ptp(line_strain_hold)),
            "pull_duration_s": events.get("hold", np.nan) - events.get("pull", np.nan),
            "tcp_pull_mm": 1e3 * float(tcp_cmd_x[hold_tail].mean()),
            "recovery_time_s": float(t[recovery_i]),
            "recovery_error_pct": 100 * abs(lengths[recovery_i, PRIMARY] - l0) / l0,
            "clamp_drift_mm": 1e3 * abs(float(clamp_drift)),
            "grip_drift_mm": 1e3 * float(grip_drift),
            "grip_nodes": int(len(grip_ids)),
            "clamp_penetration_mm": 1e3 * clamp_pen,
            "table_penetration_mm": 1e3 * max(0.0, table_z - float(z.min())),
            "tcp_tracking_error_hold_mm": 1e3
            * float(
                np.abs(
                    data["tcp"][hold_tail, 0] - origin_x - tcp_cmd_x[hold_tail]
                ).mean()
            ),
            "finger_hold_mean_mm": 1e3 * data["qpos"][hold_tail, 7].mean(),
            "finite": bool(np.isfinite(particles).all()),
        }
    )
    row["pass_strain"] = abs(row["strain_error_pp"]) <= acc["strain_tolerance_pp"]
    row["pass_recovery"] = row["recovery_error_pct"] <= 100 * acc["recovery_error_max"]
    row["pass_drift"] = (
        max(row["clamp_drift_mm"], row["grip_drift_mm"]) <= 1e3 * acc["end_drift_max"]
    )
    row["pass_penetration"] = (
        max(row["clamp_penetration_mm"], row["table_penetration_mm"])
        <= 1e3 * acc["table_penetration_tolerance"]
    )
    checks = (
        "target_reached",
        "finite",
        "pass_strain",
        "pass_recovery",
        "pass_drift",
        "pass_penetration",
    )
    row["success"] = bool(all(row[k] for k in checks))
    row["failed_checks"] = ";".join(k for k in checks if not row[k])
    return row


# --------------------------------------------------------------------------
# Cross-run checks
# --------------------------------------------------------------------------


def controller_identity() -> dict:
    sources = {}
    for platform in PLATFORMS:
        path = ADAPTERS / platform / "s01t.py"
        tree = ast.parse(path.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "StretchController"
        )
        sources[platform] = hashlib.sha256(ast.unparse(node).encode()).hexdigest()
    return {"sha256": sources, "identical": len(set(sources.values())) == 1}


def paired_difference(a, b) -> dict:
    """Frame-wise node difference of two runs of the same state."""
    pa, pb = a["states"]["particles"], b["states"]["particles"]
    n = min(len(pa), len(pb))
    diff = np.linalg.norm(pa[:n] - pb[:n], axis=2)
    la = a["states"]["line_lengths"][:n, PRIMARY]
    lb = b["states"]["line_lengths"][:n, PRIMARY]
    return {
        "frames_compared": int(n),
        "node_rms_mm": 1e3 * float(np.sqrt((diff**2).mean())),
        "node_max_mm": 1e3 * float(diff.max()),
        "line_length_max_diff_mm": 1e3 * float(np.abs(la - lb).max()),
    }


def segmentation_check(run) -> dict | None:
    """Back-project stretch_block pixels and measure distance to the nodes.

    Uses planar depth, K and the camera-to-world pose (OpenGL convention:
    the camera looks along its local -Z). A pixel on the block surface lies
    within half a cell diagonal (3.5 mm) plus the particle radius of a
    node; the check reports the fraction within 6 mm.
    """
    data = run["states"]
    if data is None:
        return None
    frame_time = data["time"]
    out = {}
    for path in sorted(run["path"].glob("images_*.npz")):
        images = dict(np.load(path))
        k, pose = images["K"], images["camera_to_world"]
        fractions, counts = [], []
        for i in range(len(images["frame"])):
            seg = images["seg"][i]
            v, u = np.nonzero(seg == SOFT_LABEL)
            if len(u) == 0:
                counts.append(0)
                continue
            z = images["depth"][i][v, u]
            x = (u + 0.5 - k[0, 2]) / k[0, 0] * z
            y = -(v + 0.5 - k[1, 2]) / k[1, 1] * z
            cam = np.stack([x, y, -z, np.ones_like(z)], axis=1)
            world = (pose @ cam.T).T[:, :3]
            j = int(np.argmin(np.abs(frame_time - images["time"][i])))
            nodes = data["particles"][j]
            pick = world[:: max(1, len(world) // 400)]
            dist = np.linalg.norm(pick[:, None, :] - nodes[None], axis=2)
            fractions.append(float((dist.min(1) < 0.006).mean()))
            counts.append(int(len(u)))
        out[path.stem.removeprefix("images_")] = {
            "frames": int(len(images["frame"])),
            "frames_with_block": int(sum(c > 0 for c in counts)),
            "mean_block_pixels": float(np.mean(counts)),
            "fraction_within_6mm": (float(np.mean(fractions)) if fractions else None),
        }
    return out


# --------------------------------------------------------------------------
# Plots and videos
# --------------------------------------------------------------------------


def _strain(run):
    d = run["states"]
    return d["line_lengths"][:, PRIMARY] / run["manifest"]["controller"]["l0"] - 1.0


def plot_states(task_runs, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = COLORS
    for (platform, variant, state), run in sorted(task_runs.items()):
        if variant != "nominal" or run["states"] is None:
            continue
        ax = axes[PLATFORMS.index(platform)]
        ax.plot(
            run["states"]["time"],
            100 * _strain(run),
            color=colors[platform],
            lw=1.4 if state == "s00" else 0.6,
            alpha=1.0 if state == "s00" else 0.5,
        )
    for ax, platform in zip(axes, PLATFORMS):
        ax.axhline(5.0, color="k", lw=0.6, ls=":")
        ax.set_title(f"{platform}: 10 initial states (s00 bold)")
        ax.set_ylabel("centre-line strain L/L0-1 [%]")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("simulation time [s]")
    fig.tight_layout()
    fig.savefig(out / "strain_initial_states.png", dpi=120)
    plt.close(fig)


def plot_variants(task_runs, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    styles = {"substeps5": ":", "nominal": "-", "substeps20": "--", "direct": "-."}
    for (platform, variant, state), run in sorted(task_runs.items()):
        if state != "s00" or run["states"] is None:
            continue
        eff = run["manifest"]["effective"]
        axes[PLATFORMS.index(platform)].plot(
            run["states"]["time"],
            100 * _strain(run),
            ls=styles.get(variant, "-"),
            label=f"{variant} ({eff['num_substeps']} substeps, graph "
            f"{eff['cuda_graph_status']})",
        )
    for ax, platform in zip(axes, PLATFORMS):
        ax.axhline(5.0, color="k", lw=0.6, ls=":")
        ax.set_title(f"{platform}: s00 substep / CUDA-Graph variants")
        ax.set_ylabel("strain [%]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("simulation time [s]")
    fig.tight_layout()
    fig.savefig(out / "strain_variants.png", dpi=120)
    plt.close(fig)


def plot_nominal(task_runs, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for platform in PLATFORMS:
        run = task_runs.get((platform, "nominal", "s00"))
        if run is None or run["states"] is None:
            continue
        d = run["states"]
        t = d["time"]
        axes[0].plot(t, 100 * _strain(run), label=platform)
        origin = run["config"]["grip"]["tcp_x"]
        (line,) = axes[1].plot(
            t, 1e3 * (d["tcp"][:, 0] - origin), label=f"{platform} actual"
        )
        axes[1].plot(
            t,
            1e3 * d["tcp_cmd_x_offset"],
            "--",
            color=line.get_color(),
            label=f"{platform} command",
        )
        axes[2].plot(t, 1e3 * d["qpos"][:, 7], label=platform)
    axes[0].axhline(5.0, color="k", lw=0.6, ls=":")
    axes[0].set_ylabel("strain [%]")
    axes[1].set_ylabel("TCP x - grip origin [mm]")
    axes[2].set_ylabel("finger q [mm]")
    axes[2].set_xlabel("simulation time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "strain_timeline_s00.png", dpi=120)
    plt.close(fig)


def depth_seg_sheet(run, out: Path):
    """One keyframe per camera: RGB, depth and segmentation side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    m = run["manifest"]
    names = sorted(
        p.stem.removeprefix("images_") for p in run["path"].glob("images_*.npz")
    )
    if not names:
        return
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 3.4 * len(names)))
    axes = np.atleast_2d(axes)
    for row, name in zip(axes, names):
        images = dict(np.load(run["path"] / f"images_{name}.npz"))
        hold = m["controller"]["events"]["hold"]
        i = int(np.argmin(np.abs(images["time"] - hold - 0.5)))
        source = run["path"] / f"rgb_{name}.mp4"
        if source.exists():  # full-image run: RGB lives in the source video
            frame_index = int(round(images["time"][i] * 30))
            rgb = imageio.get_reader(str(source)).get_data(frame_index)
        else:
            rgb = images["rgb"][i]
        row[0].imshow(rgb)
        depth = images["depth"][i]
        # No-hit pixels: DexSim ~1e16, Isaac Lab 0; mask both for display.
        im = row[1].imshow(
            np.where((depth > 0) & (depth < 10.0), depth, np.nan), cmap="viridis"
        )
        fig.colorbar(im, ax=row[1], fraction=0.035, label="m")
        row[2].imshow(
            images["seg"][i], cmap="tab10", vmin=0, vmax=9, interpolation="nearest"
        )
        for ax, title in zip(row, ("rgb", "depth (image plane)", "segmentation label")):
            ax.set_title(
                f"{m['platform']} {name} t={images['time'][i]:.2f}s " f"{title}",
                fontsize=8,
            )
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"images_{m['platform']}_s00.png", dpi=100)
    plt.close(fig)


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
VIEWS = ("oblique", "side45")
TILE = (960, 720)


def rolling_rate(
    step_seconds: np.ndarray, sim_frame: int, window: int = 30
) -> tuple[float, float]:
    """Mean physics step [s] over the last ``window`` 60 Hz frames."""
    lo = max(0, sim_frame - window)
    steps = step_seconds[lo : max(sim_frame, 1)]
    return float(steps.mean()), 1.0 / float(steps.mean())


def comparison_video(left, right, out_path: Path) -> dict | None:
    """The single exported video: 2x2 grid, columns = platform, rows =
    oblique / side45, with each platform's real-time physics rate.

    Sources are the rgb_<view>.mp4 files of the two s00 full runs. Frames are
    aligned by simulation time from t=0 at 30 Hz; the frame rate shown is the
    mean measured physics step (synchronised ``update``/``step`` wall time,
    excluding capture) over the preceding 0.5 s of simulation.
    """
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    runs = {PLATFORMS[0]: left, PLATFORMS[1]: right}
    labels = LABELS
    frames = {}
    for platform, run in runs.items():
        views = []
        for view in VIEWS:
            path = run["path"] / f"rgb_{view}.mp4"
            if not path.exists():
                return None
            views.append([f for f in imageio.get_reader(str(path))])
        frames[platform] = views
    big = ImageFont.truetype(FONT, 26)
    small = ImageFont.truetype(FONT, 20)
    count = max(len(v[0]) for v in frames.values())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path),
        fps=30,
        codec="libx264",
        quality=9,
        macro_block_size=16,
        pixelformat="yuv420p",
    )
    phases = PHASES
    for i in range(count):
        columns = []
        for platform in PLATFORMS:
            run = runs[platform]
            steps = run["step_seconds"]
            sim_frame = min(2 * i, len(steps))
            step, rate = rolling_rate(steps, sim_frame)
            phase = phases[
                int(
                    run["states"]["phase"][
                        min(sim_frame, len(run["states"]["phase"]) - 1)
                    ]
                )
            ]
            tiles = []
            for row, seq in enumerate(frames[platform]):
                image = Image.fromarray(seq[min(i, len(seq) - 1)]).resize(
                    TILE, Image.LANCZOS
                )
                draw = ImageDraw.Draw(image)
                if row == 0:
                    lines = [
                        (labels[platform], big),
                        (f"t = {i / 30:5.2f} s   phase: {phase}", small),
                        (
                            f"physics {rate:5.1f} steps/s   "
                            f"{1e3 * step:5.1f} ms/step   "
                            f"{rate / 60.0:4.2f}x real time",
                            small,
                        ),
                    ]
                    draw.rectangle((0, 0, TILE[0], 108), fill=(0, 0, 0))
                    y = 8
                    for text, font in lines:
                        draw.text((14, y), text, font=font, fill=(255, 255, 255))
                        y += 34
                draw.rectangle((0, TILE[1] - 34, 150, TILE[1]), fill=(0, 0, 0))
                draw.text(
                    (10, TILE[1] - 30), VIEWS[row], font=small, fill=(255, 255, 255)
                )
                tiles.append(np.asarray(image))
            columns.append(np.concatenate(tiles, axis=0))
        writer.append_data(np.concatenate(columns, axis=1))
    writer.close()
    info = {
        "left": left["manifest"]["run_id"],
        "right": right["manifest"]["run_id"],
        "layout": f"2x2, 1920x1440: columns = platform (left "
        f"{NAMES[PLATFORMS[0]]}, right {NAMES[PLATFORMS[1]]}); "
        "rows = oblique, side45; each tile "
        "is the 1280x960 render scaled to 960x720",
        "frame_rate_overlay": "physics steps per second = 1 / mean "
        "step_seconds (timing.csv) over the last "
        "30 simulation frames (0.5 s); excludes "
        "rendering and data capture",
        "time_mapping": "frame-aligned from simulation t=0 at 30 Hz; the "
        "shorter clip holds its last frame",
        "tool": "tools/analyze.py comparison_video (imageio + PIL)",
    }
    out_path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    return info


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return value


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})


def main():
    import argparse

    global RAW, RESULTS, VIDEOS, PLATFORMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--platforms", default=",".join(PLATFORMS))
    args = parser.parse_args()
    require_pilot()
    root = args.root.expanduser().resolve()
    rebuilt = output_dir("s01t", args.output_root)
    if rebuilt == root or root in rebuilt.parents:
        raise ValueError("Rebuild output must be outside the raw run root")
    rebuilt.mkdir(parents=True, exist_ok=False)
    RAW, RESULTS, VIDEOS = root / "raw_data", rebuilt / "results", rebuilt / "videos"
    PLATFORMS = tuple(args.platforms.split(","))
    if len(PLATFORMS) != 2:
        raise SystemExit("--platforms takes exactly two names")
    runs, superseded = discover()
    RESULTS.mkdir(parents=True, exist_ok=True)
    plots = RESULTS / "plots"
    plots.mkdir(exist_ok=True)
    task_full, timing_runs = {}, defaultdict(list)
    task_rows = []
    for key, run in sorted(runs.items()):
        platform, scenario, state, variant, kind, repeat, iterations = key
        task_rows.append(task_metrics(run))
        if kind == "full":
            task_full[(platform, variant, state)] = run
        elif variant == "nominal" and state == "s00":
            timing_runs[platform].append(run)
    write_csv(RESULTS / "task_metrics.csv", task_rows)
    by_id = {r["run_id"]: r for r in task_rows}

    summary = {
        "controller_source": controller_identity(),
        "superseded_runs": superseded,
    }
    states = {}
    for platform in PLATFORMS:
        rows = [
            by_id[r["manifest"]["run_id"]]
            for (p, v, s), r in sorted(task_full.items())
            if p == platform and v == "nominal"
        ]
        if not rows:
            continue
        strain = np.array([r["hold_strain_pct"] for r in rows])

        def span(key):
            return [float(min(r[key] for r in rows)), float(max(r[key] for r in rows))]

        states[platform] = {
            "states": [r["state"] for r in rows],
            "successes": int(sum(r["success"] for r in rows)),
            "denominator": len(rows),
            "failures": {
                r["state"]: r["failed_checks"] for r in rows if not r["success"]
            },
            "hold_strain_pct_mean": float(strain.mean()),
            "hold_strain_pct_std": (
                float(strain.std(ddof=1)) if len(strain) > 1 else None
            ),
            "hold_strain_pct_range": [float(strain.min()), float(strain.max())],
            "peak_strain_pct_range": span("peak_strain_pct"),
            "hold_strain_lines_spread_pp_max": span("hold_strain_lines_spread_pp")[1],
            "recovery_error_pct_max": span("recovery_error_pct")[1],
            "clamp_drift_mm_max": span("clamp_drift_mm")[1],
            "grip_drift_mm_max": span("grip_drift_mm")[1],
            "clamp_penetration_mm_max": span("clamp_penetration_mm")[1],
            "table_penetration_mm_max": span("table_penetration_mm")[1],
            "pull_duration_s_range": span("pull_duration_s"),
            "tcp_pull_mm_range": span("tcp_pull_mm"),
        }
    summary["initial_states"] = states

    variants = {}
    for platform in PLATFORMS:
        base = task_full.get((platform, "nominal", "s00"))
        if base is None:
            continue
        entry = {}
        for variant in ("substeps5", "substeps20", "direct"):
            other = task_full.get((platform, variant, "s00"))
            if other is None:
                continue
            a, b = by_id[base["manifest"]["run_id"]], by_id[other["manifest"]["run_id"]]
            entry[variant] = {
                "effective_substeps": b["num_substeps"],
                "cuda_graph": b["cuda_graph"],
                "hold_strain_pct": b["hold_strain_pct"],
                "peak_strain_pct": b["peak_strain_pct"],
                "pull_duration_s": b["pull_duration_s"],
                "recovery_error_pct": b["recovery_error_pct"],
                "success": b["success"],
                "failed_checks": b["failed_checks"],
                "step_ms_mean": b["step_ms_mean"],
                "vs_nominal": paired_difference(base, other),
                "nominal_hold_strain_pct": a["hold_strain_pct"],
            }
        variants[platform] = entry
    summary["variants_s00"] = variants

    timing = {}
    for platform, items in timing_runs.items():
        means = np.array([1e3 * r["step_seconds"].mean() for r in items])
        timing[platform] = {
            "runs": [r["manifest"]["run_id"] for r in items],
            "step_ms_mean_per_run": means.tolist(),
            "step_ms_mean": float(means.mean()),
            "step_ms_run_spread": float(np.ptp(means)),
            "gpu_process_peak_mib": [
                r["manifest"].get("gpu_memory", {}).get("gpu_process_peak_mib")
                for r in items
            ],
            "init_s": [r["manifest"].get("init_seconds") for r in items],
        }
    summary["timing_s00_nominal"] = timing

    seg = {}
    for platform in PLATFORMS:
        run = task_full.get((platform, "nominal", "s00"))
        if run is not None:
            seg[platform] = segmentation_check(run)
    summary["segmentation_check_s00"] = seg
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, default=fmt) + "\n"
    )

    plot_states(task_full, plots)
    plot_variants(task_full, plots)
    plot_nominal(task_full, plots)
    for platform in PLATFORMS:
        run = task_full.get((platform, "nominal", "s00"))
        if run is not None:
            depth_seg_sheet(run, plots)
    left = task_full.get((PLATFORMS[0], "nominal", "s00"))
    right = task_full.get((PLATFORMS[1], "nominal", "s00"))
    if left is not None and right is not None:
        comparison_video(left, right, VIDEOS / "s01t_comparison.mp4")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("controller_source", "initial_states", "timing_s00_nominal")
            },
            indent=1,
            default=fmt,
        )
    )


if __name__ == "__main__":
    main()
