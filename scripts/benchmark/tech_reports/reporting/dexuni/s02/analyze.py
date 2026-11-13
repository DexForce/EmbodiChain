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
"""Legacy S02 pilot analysis; not a formal v2 report builder.

Run: DEXUNI_ALLOW_PILOT=1 <numpy-python> scripts/benchmark/tech_reports/reporting/dexuni/s02/analyze.py
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

TASK_DIR = output_dir("s02")
INPUTS = input_dir("s02")
RAW = TASK_DIR / "raw_data"
RESULTS = TASK_DIR / "results"
VIDEOS = TASK_DIR / "videos"
PLATFORMS = ("embodichain", "isaaclab")
COLORS = {"dexsim": "tab:blue", "embodichain": "tab:green", "isaaclab": "tab:orange"}
LABELS = {
    "dexsim": "DexSim  (DexUni two-way)",
    "embodichain": "EmbodiChain  (DexUni two-way)",
    "isaaclab": "Isaac Lab 3.0  (MuJoCo-Warp + VBD coupler)",
}
NAMES = {"dexsim": "DexSim", "embodichain": "EmbodiChain", "isaaclab": "Isaac Lab"}
PHASES = (
    "settle",
    "squeeze",
    "squeeze_settle",
    "lift",
    "hold",
    "transport",
    "lower",
    "release",
    "retreat",
    "observe",
    "done",
)
SOFT_LABEL = 5


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
    return (m["platform"], m["state"]["id"], m["variant"], m["kind"], m["repeat"])


def discover():
    runs, superseded = {}, defaultdict(list)
    for platform in PLATFORMS:
        for path in sorted((RAW / platform).glob("s02_*")):
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


_PATH = None


def commanded_tcp(s: np.ndarray) -> np.ndarray:
    """TCP position of the shared path at arc length s (inputs/)."""
    global _PATH
    if _PATH is None:
        _PATH = dict(np.load(INPUTS / "s02_ik_path.npz"))
    return np.stack(
        [np.interp(s, _PATH["s"], _PATH["tcp"][:, k]) for k in range(3)], axis=-1
    )


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


def frame_at(t: np.ndarray, time: float) -> int:
    return int(np.argmin(np.abs(t - time)))


def task_metrics(run) -> dict:
    m, config = run["manifest"], run["config"]
    eff = m.get("effective", {})
    ctrl = m.get("controller", {})
    row = {
        "run_id": m["run_id"],
        "platform": m["platform"],
        "state": m["state"]["id"],
        "variant": m["variant"],
        "kind": m["kind"],
        "repeat": m["repeat"],
        "num_substeps": eff.get("num_substeps"),
        "cuda_graph": eff.get("cuda_graph_status"),
        "episode_s": ctrl.get("events", {}).get("done"),
        **timing_row(run),
    }
    data = run["states"]
    if data is None:
        return row
    acc, grip = config["acceptance"], config["grip"]
    table_z = config["scene"]["table_z"]
    t, phase = data["time"], data["phase"]
    p = data["particles"]
    events, w0 = ctrl["events"], ctrl["w0"]
    width = data["width"]
    comp = 1.0 - width / w0
    z = p[:, :, 2]
    zmin = z.min(axis=1) - table_z
    centroid = p.mean(axis=1)

    def mask(name):
        return phase == PHASES.index(name)

    squeeze_end = int(np.flatnonzero(mask("squeeze_settle"))[-1])
    hold = mask("hold")
    hold_i = np.flatnonzero(hold)
    hold_end = int(hold_i[-1])
    tail = hold & (t >= t[hold_end] - acc["hold_drop_window"] - 1e-9)
    tail_start = int(np.flatnonzero(tail)[0])
    transport_end = int(np.flatnonzero(mask("transport"))[-1])
    lower_end = int(np.flatnonzero(mask("lower"))[-1])
    # The fingers are fully open when "retreat" starts.
    place_i = frame_at(t, events["retreat"] + acc["place_check_delay"])
    end = len(t) - 1
    target = np.asarray(config["target_area"]["center"])
    half = 0.5 * np.asarray(config["target_area"]["size"])
    xy = p[place_i, :, :2]
    inside = np.all(np.abs(xy - target) <= half, axis=1)
    # Grip slip: block centroid relative to the TCP, from the end of the
    # squeeze to the end of the lowering (nothing but the grip holds it).
    rel = centroid - data["tcp"]
    slip = np.linalg.norm(rel[lower_end] - rel[squeeze_end])
    tcp_cmd = commanded_tcp(data["path_s_cmd"])
    track = np.linalg.norm(data["tcp"] - tcp_cmd, axis=1)
    moving = mask("lift") | mask("transport") | mask("lower")
    # Squeeze rebound: the block briefly springs back against the closing
    # fingers (a contact event of random size); largest rise of the width
    # above its running minimum during the squeeze and its settle.
    squeezing = np.flatnonzero(mask("squeeze") | mask("squeeze_settle"))
    w_sq = width[squeezing]
    rebound = float((w_sq - np.minimum.accumulate(w_sq)).max())
    rebound_i = int(squeezing[np.argmax(w_sq - np.minimum.accumulate(w_sq))])
    lo = config["grip"]["compression_band"][0]
    hi = config["grip"]["compression_band"][1]
    stop = ctrl["squeeze_stop"]
    row.update(
        {
            "w0_mm": 1e3 * w0,
            "squeeze_stop_reason": stop["reason"],
            "squeeze_duration_s": stop["time"] - events["squeeze"],
            "compression_at_stop_pct": 100 * stop["compression"],
            "squeeze_rebound_mm": 1e3 * rebound,
            "squeeze_rebound_time_s": float(t[rebound_i]),
            "grasp_compression_pct": 100 * comp[squeeze_end],
            "hold_compression_pct": 100 * comp[hold].mean(),
            "finger_target_closed_mm": 1e3 * ctrl["finger_closed"],
            "finger_hold_mm": 1e3 * data["qpos"][hold, 7].mean(),
            "finger_lag_hold_mm": 1e3
            * (data["qpos"][hold, 7].mean() - ctrl["finger_closed"]),
            "lift_min_height_mm": 1e3 * float(zmin[hold].min()),
            "hold_drop_mm": 1e3
            * float(centroid[tail_start, 2] - centroid[hold_end, 2]),
            "transport_end_height_mm": 1e3 * float(zmin[transport_end]),
            "grip_slip_mm": 1e3 * float(slip),
            "place_time_s": float(t[place_i]),
            "place_center_error_mm": 1e3
            * float(np.linalg.norm(centroid[place_i, :2] - target)),
            "place_inside_fraction": float(inside.mean()),
            "place_rest_height_mm": 1e3 * float(zmin[place_i]),
            "width_end_mm": 1e3 * float(width[end]),
            "width_recovery_error_pct": 100 * abs(width[end] / w0 - 1.0),
            "table_penetration_mm": 1e3 * max(0.0, -float(zmin.min())),
            "tcp_tracking_error_move_mm": 1e3 * float(track[moving].mean()),
            "tcp_tracking_error_max_mm": 1e3 * float(track[moving].max()),
            "finite": bool(np.isfinite(p).all()),
        }
    )
    row["pass_grasp"] = lo <= row["grasp_compression_pct"] / 100 <= hi
    row["pass_lift"] = row["lift_min_height_mm"] >= 1e3 * acc["lift_min_height"]
    row["pass_hold"] = row["hold_drop_mm"] <= 1e3 * acc["hold_drop_max"]
    row["pass_transport"] = (
        row["transport_end_height_mm"] >= 1e3 * acc["lift_min_height"]
    )
    row["pass_place"] = (
        row["place_center_error_mm"] <= 1e3 * acc["place_center_error_max"]
        and row["place_inside_fraction"] >= acc["place_inside_fraction_min"]
        and row["place_rest_height_mm"] <= 1e3 * acc["place_rest_height_max"]
    )
    row["pass_penetration"] = (
        row["table_penetration_mm"] <= 1e3 * acc["table_penetration_tolerance"]
    )
    checks = (
        "finite",
        "pass_grasp",
        "pass_lift",
        "pass_hold",
        "pass_transport",
        "pass_place",
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
    for platform in ("dexsim", "embodichain", "isaaclab"):
        path = ADAPTERS / platform / "s02.py"
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "PickPlaceController"
        )
        sources[platform] = hashlib.sha256(ast.unparse(node).encode()).hexdigest()
    return {"sha256": sources, "identical": len(set(sources.values())) == 1}


def paired_difference(a, b) -> dict:
    """Frame-wise node difference of two runs of the same state."""
    pa, pb = a["states"]["particles"], b["states"]["particles"]
    n = min(len(pa), len(pb))
    diff = np.linalg.norm(pa[:n] - pb[:n], axis=2)
    wa, wb = a["states"]["width"][:n], b["states"]["width"][:n]
    ca, cb = pa[:n].mean(axis=1), pb[:n].mean(axis=1)
    return {
        "frames_compared": int(n),
        "node_rms_mm": 1e3 * float(np.sqrt((diff**2).mean())),
        "node_max_mm": 1e3 * float(diff.max()),
        "centroid_max_mm": 1e3 * float(np.linalg.norm(ca - cb, axis=1).max()),
        "width_max_diff_mm": 1e3 * float(np.abs(wa - wb).max()),
    }


def segmentation_check(run) -> dict | None:
    """Back-project soft_block pixels and measure distance to the nodes.

    Uses planar depth, K and the camera-to-world pose (OpenGL convention:
    the camera looks along its local -Z). A pixel on the block surface lies
    within half a cell diagonal (4.3 mm) plus the particle radius of a node;
    the check reports the fraction within 7 mm.
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
            fractions.append(float((dist.min(1) < 0.007).mean()))
            counts.append(int(len(u)))
        out[path.stem.removeprefix("images_")] = {
            "frames": int(len(images["frame"])),
            "frames_with_block": int(sum(c > 0 for c in counts)),
            "mean_block_pixels": float(np.mean(counts)),
            "fraction_within_7mm": (float(np.mean(fractions)) if fractions else None),
        }
    return out


# --------------------------------------------------------------------------
# Plots and videos
# --------------------------------------------------------------------------


def _series(run):
    d, m, config = run["states"], run["manifest"], run["config"]
    table_z = config["scene"]["table_z"]
    target = np.asarray(config["target_area"]["center"])
    centroid = d["particles"].mean(axis=1)
    return {
        "t": d["time"],
        "compression": 100 * (1 - d["width"] / m["controller"]["w0"]),
        "zmin": 1e3 * (d["particles"][:, :, 2].min(axis=1) - table_z),
        "target_dist": 1e3 * np.linalg.norm(centroid[:, :2] - target, axis=1),
    }


def plot_states(task_runs, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for (platform, variant, state), run in sorted(task_runs.items()):
        if variant != "nominal" or run["states"] is None:
            continue
        col = PLATFORMS.index(platform)
        s = _series(run)
        style = dict(
            color=COLORS[platform],
            lw=1.4 if state == "s00" else 0.6,
            alpha=1.0 if state == "s00" else 0.5,
        )
        axes[0, col].plot(s["t"], s["compression"], **style)
        axes[1, col].plot(s["t"], s["zmin"], **style)
        axes[2, col].plot(s["t"], s["target_dist"], **style)
    for col, platform in enumerate(PLATFORMS):
        axes[0, col].set_title(f"{NAMES[platform]}: 10 initial states " "(s00 bold)")
        axes[0, col].axhspan(30, 40, color="k", alpha=0.08)
        axes[1, col].axhline(80, color="k", lw=0.6, ls=":")
        axes[2, col].axhline(20, color="k", lw=0.6, ls=":")
        axes[2, col].set_xlabel("simulation time [s]")
    axes[0, 0].set_ylabel("width compression [%]")
    axes[1, 0].set_ylabel("lowest node above table [mm]")
    axes[2, 0].set_ylabel("centroid XY to target [mm]")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pick_place_initial_states.png", dpi=120)
    plt.close(fig)


def plot_variants(task_runs, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    styles = {"substeps5": ":", "nominal": "-", "substeps20": "--", "direct": "-."}
    for (platform, variant, state), run in sorted(task_runs.items()):
        if state != "s00" or run["states"] is None:
            continue
        eff = run["manifest"]["effective"]
        s = _series(run)
        col = PLATFORMS.index(platform)
        label = (
            f"{variant} ({eff['num_substeps']} substeps, graph "
            f"{eff['cuda_graph_status']})"
        )
        axes[0, col].plot(
            s["t"], s["compression"], ls=styles.get(variant, "-"), label=label
        )
        axes[1, col].plot(s["t"], s["zmin"], ls=styles.get(variant, "-"), label=label)
    for col, platform in enumerate(PLATFORMS):
        axes[0, col].set_title(
            f"{NAMES[platform]}: s00 substep / " "CUDA-Graph variants"
        )
        axes[0, col].legend(fontsize=7)
        axes[1, col].set_xlabel("simulation time [s]")
    axes[0, 0].set_ylabel("width compression [%]")
    axes[1, 0].set_ylabel("lowest node above table [mm]")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pick_place_variants.png", dpi=120)
    plt.close(fig)


def plot_nominal(task_runs, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    for platform in PLATFORMS:
        run = task_runs.get((platform, "nominal", "s00"))
        if run is None or run["states"] is None:
            continue
        d = run["states"]
        s = _series(run)
        t = s["t"]
        axes[0].plot(t, s["compression"], label=NAMES[platform], color=COLORS[platform])
        axes[1].plot(
            t,
            1e3 * d["qpos"][:, 7],
            color=COLORS[platform],
            label=f"{NAMES[platform]} actual",
        )
        axes[1].plot(
            t,
            1e3 * d["finger_target"],
            "--",
            color=COLORS[platform],
            label=f"{NAMES[platform]} target",
        )
        axes[2].plot(t, s["zmin"], color=COLORS[platform], label=NAMES[platform])
        track = np.linalg.norm(d["tcp"] - commanded_tcp(d["path_s_cmd"]), axis=1)
        axes[3].plot(t, 1e3 * track, color=COLORS[platform], label=NAMES[platform])
    axes[0].axhspan(30, 40, color="k", alpha=0.08)
    axes[0].set_ylabel("width compression [%]")
    axes[1].set_ylabel("finger q [mm]")
    axes[2].axhline(80, color="k", lw=0.6, ls=":")
    axes[2].set_ylabel("lowest node above table [mm]")
    axes[3].set_ylabel("TCP tracking error [mm]")
    axes[3].set_xlabel("simulation time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pick_place_timeline_s00.png", dpi=120)
    plt.close(fig)


def depth_seg_sheet(run, out: Path):
    """One keyframe per camera (hold): RGB, depth and segmentation."""
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
        i = int(np.argmin(np.abs(images["time"] - hold - 1.0)))
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
VIEWS = ("oblique", "front")
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
    oblique / front, with each platform's real-time physics rate.

    Sources are the rgb_<view>.mp4 files of the two s00 full runs. Frames are
    aligned by simulation time from t=0 at 30 Hz; the frame rate shown is the
    mean measured physics step (synchronised ``update``/``step`` wall time,
    excluding capture) over the preceding 0.5 s of simulation.
    """
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    runs = {PLATFORMS[0]: left, PLATFORMS[1]: right}
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
    for i in range(count):
        columns = []
        for platform in PLATFORMS:
            run = runs[platform]
            steps = run["step_seconds"]
            sim_frame = min(2 * i, len(steps))
            step, rate = rolling_rate(steps, sim_frame)
            phase = PHASES[
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
                        (LABELS[platform], big),
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
        "rows = oblique, front; each tile is the 1280x960 "
        "render scaled to 960x720",
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


STAGES = ("pass_grasp", "pass_lift", "pass_hold", "pass_transport", "pass_place")


def main():
    import argparse

    global RAW, RESULTS, VIDEOS, PLATFORMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--platforms", default=",".join(PLATFORMS))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    require_pilot()
    root = args.root.expanduser().resolve()
    rebuilt = output_dir("s02", args.output_root)
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
        platform, state, variant, kind, repeat = key
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

        def span(key):
            return [float(min(r[key] for r in rows)), float(max(r[key] for r in rows))]

        states[platform] = {
            "states": [r["state"] for r in rows],
            "successes": int(sum(r["success"] for r in rows)),
            "denominator": len(rows),
            "stage_successes": {k: int(sum(r[k] for r in rows)) for k in STAGES},
            "failures": {
                r["state"]: r["failed_checks"] for r in rows if not r["success"]
            },
            **{
                f"{k}_range": span(k)
                for k in (
                    "squeeze_duration_s",
                    "compression_at_stop_pct",
                    "squeeze_rebound_mm",
                    "grasp_compression_pct",
                    "hold_compression_pct",
                    "finger_lag_hold_mm",
                    "lift_min_height_mm",
                    "hold_drop_mm",
                    "transport_end_height_mm",
                    "grip_slip_mm",
                    "place_center_error_mm",
                    "place_inside_fraction",
                    "place_rest_height_mm",
                    "width_recovery_error_pct",
                    "table_penetration_mm",
                    "tcp_tracking_error_move_mm",
                    "tcp_tracking_error_max_mm",
                    "episode_s",
                )
            },
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
            b = by_id[other["manifest"]["run_id"]]
            entry[variant] = {
                "effective_substeps": b["num_substeps"],
                "cuda_graph": b["cuda_graph"],
                **{
                    k: b[k]
                    for k in (
                        "grasp_compression_pct",
                        "hold_compression_pct",
                        "lift_min_height_mm",
                        "grip_slip_mm",
                        "place_center_error_mm",
                        "place_inside_fraction",
                        "success",
                        "failed_checks",
                        "step_ms_mean",
                    )
                },
                "vs_nominal": paired_difference(base, other),
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
    if left is not None and right is not None and not args.no_video:
        comparison_video(left, right, VIDEOS / "s02_comparison.mp4")
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
