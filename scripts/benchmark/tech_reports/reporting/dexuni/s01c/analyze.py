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
"""Legacy S01C pilot analysis; not a formal v2 report builder.

Run: DEXUNI_ALLOW_PILOT=1 <numpy-python> scripts/benchmark/tech_reports/reporting/dexuni/s01c/analyze.py
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

SUBTASK_DIR = output_dir("s01c")
INPUTS = input_dir("s01c")
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
PHASES = ("rest", "approach", "press", "hold", "unload", "recover", "done")
SOFT_LABEL = 6
GRAVITY = 9.81


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
        for path in sorted((RAW / platform).glob("s01c_*")):
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
    h = np.ptp(particles[:, :, 2], axis=1)
    ctrl = m["controller"]
    h0, events = ctrl["h0"], ctrl["events"]
    target = m["target_compression"]
    hold = phase == PHASES.index("hold")
    hold_tail = hold & (t >= t[hold].max() - 0.8 + 1e-9)
    compression = 1.0 - h[hold_tail] / h0
    recovery_t = events.get("recover", np.nan) + 2.0
    recovery_i = int(np.argmin(np.abs(t - recovery_t)))
    plate, tcp = data["plate_pose"], data["tcp"]
    grip = tcp[:, 2] - plate[:, 2, 3]
    ref = (t >= config["measurement"]["reference_window"][0]) & (
        t <= config["measurement"]["reference_window"][1]
    )
    tilt = np.degrees(np.arccos(np.clip(plate[:, 2, 2], -1, 1)))
    fingers = data["qpos"][:, 7]
    contact_z = float(np.load(INPUTS / "s01c_ik_table.npz")["contact_tcp_z"])
    tcp_cmd = contact_z + data["tcp_cmd_z_offset"]
    table_z = config["scene"]["table_z"]
    acc = config["acceptance"]
    row.update(
        {
            "h0_mm": 1e3 * h0,
            "rest_mesh_h_mm": 1e3 * np.ptp(data["rest_particles"][:, 2]),
            "hold_compression_pct": 100 * compression.mean(),
            "hold_compression_min_pct": 100 * compression.min(),
            "hold_compression_max_pct": 100 * compression.max(),
            "compression_error_pp": 100 * (compression.mean() - target),
            "peak_compression_pct": 100 * (1 - h.min() / h0),
            "press_duration_s": events.get("hold", np.nan)
            - events.get("press", np.nan),
            "recovery_time_s": float(t[recovery_i]),
            "recovery_error_pct": 100 * abs(h[recovery_i] - h0) / h0,
            "table_penetration_mm": 1e3
            * max(0.0, table_z - float(particles[:, :, 2].min())),
            "grip_slip_mm": 1e3 * abs(grip[hold_tail].mean() - grip[ref].mean()),
            "grip_slip_episode_max_mm": 1e3
            * float(np.abs(grip - grip[ref].mean()).max()),
            "plate_tilt_max_deg": float(tilt.max()),
            "block_xy_drift_mm": 1e3
            * float(
                np.linalg.norm(
                    particles[recovery_i, :, :2].mean(0)
                    - data["rest_particles"][:, :2].mean(0)
                )
            ),
            "tcp_tracking_error_hold_mm": 1e3
            * float(np.abs(tcp[hold_tail, 2] - tcp_cmd[hold_tail]).mean()),
            "finger_hold_mean_mm": 1e3 * fingers[hold_tail].mean(),
            "finger_hold_p2p_mm": 1e3 * np.ptp(fingers[hold_tail]),
            "finite": bool(np.isfinite(particles).all()),
        }
    )
    row["pass_compression"] = (
        abs(row["compression_error_pp"]) <= acc["compression_tolerance_pp"]
    )
    row["pass_recovery"] = row["recovery_error_pct"] <= 100 * acc["recovery_error_max"]
    row["pass_table"] = (
        row["table_penetration_mm"] <= 1e3 * acc["table_penetration_tolerance"]
    )
    # Whole episode: the plate must stay seated in the hand, also after
    # unloading (hold-only slip missed a late drop in a pilot run).
    row["pass_grip"] = row["grip_slip_episode_max_mm"] <= 1e3 * acc["grip_slip_max"]
    row["success"] = bool(
        row["target_reached"]
        and row["finite"]
        and all(
            row[k]
            for k in ("pass_compression", "pass_recovery", "pass_table", "pass_grip")
        )
    )
    row["failed_checks"] = ";".join(
        k
        for k in (
            "target_reached",
            "finite",
            "pass_compression",
            "pass_recovery",
            "pass_table",
            "pass_grip",
        )
        if not row[k]
    )
    return row


def bench_metrics(run) -> dict:
    m, config = run["manifest"], run["config"]
    data = run["states"]
    bench = config["bench"]
    t = data["time"]
    p = data["particles"]
    h = np.ptp(p[:, :, 2], axis=1)
    lo, hi = bench["measure_window"]
    window = (t >= lo) & (t <= hi)
    h_rest = float(np.ptp(data["rest_particles"][:, 2]))
    block = config["soft_block"]
    size = block["size"]
    area = size[0] * size[1]
    load = bench["bench_mass"] * GRAVITY / area
    self_weight = block["density"] * GRAVITY * size[2] / 2.0
    estimate = size[2] * (load + self_weight) / block["youngs_modulus"]
    plate = data["plate_pose"]
    tilt = np.degrees(np.arccos(np.clip(plate[:, 2, 2], -1, 1)))
    dh = h_rest - h[window]
    return {
        "run_id": m["run_id"],
        "platform": m["platform"],
        "vbd_iterations": m["requested"]["vbd_iterations"],
        "num_substeps": m.get("effective", {}).get("num_substeps"),
        "bench_mass_kg": bench["bench_mass"],
        "estimate_dh_mm": 1e3 * estimate,
        "measured_dh_mm": 1e3 * dh.mean(),
        "measured_dh_p2p_mm": 1e3 * np.ptp(dh),
        "ratio_measured_over_estimate": dh.mean() / estimate,
        "weight_tilt_max_deg": float(tilt.max()),
        "weight_on_block_at_end": bool(
            abs(plate[-1, 0, 3] - config["trajectory"]["block_xy"][0]) < 0.02
            and abs(plate[-1, 1, 3] - config["trajectory"]["block_xy"][1]) < 0.02
            and tilt[-1] < 10.0
        ),
        **timing_row(run),
    }


# --------------------------------------------------------------------------
# Cross-run checks
# --------------------------------------------------------------------------


def controller_identity() -> dict:
    sources = {}
    for platform in PLATFORMS:
        path = ADAPTERS / platform / "s01c.py"
        tree = ast.parse(path.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "CompressionController"
        )
        text = ast.unparse(node)
        sources[platform] = hashlib.sha256(text.encode()).hexdigest()
    return {"sha256": sources, "identical": len(set(sources.values())) == 1}


def paired_difference(a, b) -> dict:
    """Frame-wise node difference of two runs of the same state."""
    pa, pb = a["states"]["particles"], b["states"]["particles"]
    n = min(len(pa), len(pb))
    diff = np.linalg.norm(pa[:n] - pb[:n], axis=2)
    ha = np.ptp(pa[:n, :, 2], axis=1)
    hb = np.ptp(pb[:n, :, 2], axis=1)
    return {
        "frames_compared": int(n),
        "node_rms_mm": 1e3 * float(np.sqrt((diff**2).mean())),
        "node_max_mm": 1e3 * float(diff.max()),
        "height_max_diff_mm": 1e3 * float(np.abs(ha - hb).max()),
    }


def segmentation_check(run) -> dict | None:
    """Back-project soft_block pixels and measure distance to the nodes.

    Uses planar depth, K and the camera-to-world pose (OpenGL convention:
    the camera looks along its local -Z). A pixel on the block surface lies
    within half the node spacing (3.75 mm) plus the particle radius of a
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


def plot_states(task_runs, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = COLORS
    for (platform, variant, state), run in sorted(task_runs.items()):
        if variant != "nominal" or run["states"] is None:
            continue
        d = run["states"]
        h = np.ptp(d["particles"][:, :, 2], axis=1)
        h0 = run["manifest"]["controller"]["h0"]
        ax = axes[PLATFORMS.index(platform)]
        ax.plot(
            d["time"],
            100 * (1 - h / h0),
            color=colors[platform],
            lw=1.4 if state == "s00" else 0.6,
            alpha=1.0 if state == "s00" else 0.5,
            label=state if state in ("s00", "s01") else None,
        )
    for ax, platform in zip(axes, PLATFORMS):
        ax.axhline(5.0, color="k", lw=0.6, ls=":")
        ax.set_title(f"{platform}: 10 initial states (s00 bold)")
        ax.set_ylabel("compression 1-h/h0 [%]")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("simulation time [s]")
    fig.tight_layout()
    fig.savefig(out / "compression_initial_states.png", dpi=120)
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
        d = run["states"]
        h = np.ptp(d["particles"][:, :, 2], axis=1)
        h0 = run["manifest"]["controller"]["h0"]
        sub = run["manifest"]["effective"]["num_substeps"]
        graph = run["manifest"]["effective"]["cuda_graph_status"]
        axes[PLATFORMS.index(platform)].plot(
            d["time"],
            100 * (1 - h / h0),
            ls=styles.get(variant, "-"),
            label=f"{variant} ({sub} substeps, graph {graph})",
        )
    for ax, platform in zip(axes, PLATFORMS):
        ax.axhline(5.0, color="k", lw=0.6, ls=":")
        ax.set_title(f"{platform}: s00 substep / CUDA-Graph variants")
        ax.set_ylabel("compression [%]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("simulation time [s]")
    fig.tight_layout()
    fig.savefig(out / "compression_variants.png", dpi=120)
    plt.close(fig)


def plot_nominal(task_runs, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contact_z = float(np.load(INPUTS / "s01c_ik_table.npz")["contact_tcp_z"])
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for platform in PLATFORMS:
        run = task_runs.get((platform, "nominal", "s00"))
        if run is None or run["states"] is None:
            continue
        d = run["states"]
        t = d["time"]
        h = np.ptp(d["particles"][:, :, 2], axis=1)
        h0 = run["manifest"]["controller"]["h0"]
        axes[0].plot(t, 100 * (1 - h / h0), label=platform)
        (line,) = axes[1].plot(
            t, 1e3 * (d["tcp"][:, 2] - contact_z), label=f"{platform} actual"
        )
        axes[1].plot(
            t,
            1e3 * d["tcp_cmd_z_offset"],
            "--",
            color=line.get_color(),
            label=f"{platform} command",
        )
        axes[2].plot(t, 1e3 * d["qpos"][:, 7], label=platform)
    axes[0].axhline(5.0, color="k", lw=0.6, ls=":")
    axes[0].set_ylabel("compression [%]")
    axes[1].set_ylabel("TCP z - contact [mm]")
    axes[2].set_ylabel("finger q [mm]")
    axes[2].set_xlabel("simulation time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "compression_timeline_s00.png", dpi=120)
    plt.close(fig)


def plot_bench(bench_runs, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    estimate = None
    for (platform, iterations), run in sorted(bench_runs.items()):
        d = run["states"]
        h_rest = np.ptp(d["rest_particles"][:, 2])
        dh = h_rest - np.ptp(d["particles"][:, :, 2], axis=1)
        ax.plot(
            d["time"],
            1e3 * dh,
            ls="-" if platform == PLATFORMS[0] else "--",
            label=f"{platform} {iterations} it",
        )
        estimate = bench_metrics(run)["estimate_dh_mm"]
    if estimate is not None:
        ax.axhline(estimate, color="k", lw=0.8, ls=":", label="linear-elastic estimate")
    ax.set_xlabel("simulation time [s]")
    ax.set_ylabel("block height loss h_rest - h [mm]")
    ax.set_title("robot-free bench: 0.5 kg flat weight on block A")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "bench_compression.png", dpi=120)
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
VIEWS = ("oblique", "front45")
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
    oblique / front45, with each platform's real-time physics rate.

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
    phases = ("rest", "approach", "press", "hold", "unload", "recover", "done")
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
        "rows = oblique, front45; each tile "
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
    rebuilt = output_dir("s01c", args.output_root)
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
    task_full, timing_runs, bench_runs = {}, defaultdict(list), {}
    task_rows, bench_rows = [], []
    for key, run in sorted(runs.items()):
        platform, scenario, state, variant, kind, repeat, iterations = key
        if scenario == "bench":
            bench_runs[(platform, iterations)] = run
            bench_rows.append(bench_metrics(run))
            continue
        task_rows.append(task_metrics(run))
        if kind == "full":
            task_full[(platform, variant, state)] = run
        elif variant == "nominal" and state == "s00":
            timing_runs[platform].append(run)
    write_csv(RESULTS / "task_metrics.csv", task_rows)
    write_csv(RESULTS / "bench_metrics.csv", bench_rows)
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
        comp = np.array([r["hold_compression_pct"] for r in rows])
        states[platform] = {
            "states": [r["state"] for r in rows],
            "successes": int(sum(r["success"] for r in rows)),
            "denominator": len(rows),
            "failures": {
                r["state"]: r["failed_checks"] for r in rows if not r["success"]
            },
            "hold_compression_pct_mean": float(comp.mean()),
            "hold_compression_pct_std": (
                float(comp.std(ddof=1)) if len(comp) > 1 else None
            ),
            "hold_compression_pct_range": [float(comp.min()), float(comp.max())],
            "recovery_error_pct_max": float(max(r["recovery_error_pct"] for r in rows)),
            "grip_slip_mm_max": float(max(r["grip_slip_mm"] for r in rows)),
            "grip_slip_episode_max_mm": float(
                max(r["grip_slip_episode_max_mm"] for r in rows)
            ),
            "block_xy_drift_mm_range": [
                float(min(r["block_xy_drift_mm"] for r in rows)),
                float(max(r["block_xy_drift_mm"] for r in rows)),
            ],
            "table_penetration_mm_max": float(
                max(r["table_penetration_mm"] for r in rows)
            ),
            "press_duration_s_range": [
                float(min(r["press_duration_s"] for r in rows)),
                float(max(r["press_duration_s"] for r in rows)),
            ],
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
                "hold_compression_pct": b["hold_compression_pct"],
                "peak_compression_pct": b["peak_compression_pct"],
                "press_duration_s": b["press_duration_s"],
                "recovery_error_pct": b["recovery_error_pct"],
                "success": b["success"],
                "step_ms_mean": b["step_ms_mean"],
                "vs_nominal": paired_difference(base, other),
                "nominal_hold_compression_pct": a["hold_compression_pct"],
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
    summary["bench"] = bench_rows
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, default=fmt) + "\n"
    )

    plot_states(task_full, plots)
    plot_variants(task_full, plots)
    plot_nominal(task_full, plots)
    if bench_runs:
        plot_bench(bench_runs, plots)
    for platform in PLATFORMS:
        run = task_full.get((platform, "nominal", "s00"))
        if run is not None:
            depth_seg_sheet(run, plots)
    left = task_full.get((PLATFORMS[0], "nominal", "s00"))
    right = task_full.get((PLATFORMS[1], "nominal", "s00"))
    if left is not None and right is not None:
        comparison_video(left, right, VIDEOS / "s01c_comparison.mp4")
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
