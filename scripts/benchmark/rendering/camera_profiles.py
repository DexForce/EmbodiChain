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

"""Compare default, throughput and retained-gutter FastRT camera settings.

All profiles render the same static scene, one RGB-D camera per Arena, and
one native group. This measures configuration tradeoffs, not equal image quality.
Run: python -m scripts.benchmark.rendering.camera_profiles
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.benchmark.rendering.camera_grouping import _other_gpu_processes
from scripts.benchmark.rendering.offscreen_denoising import (
    BenchmarkCfg,
    _build_benchmark_scene,
    _camera_pose,
    _gpu_description,
    _memory_snapshot,
    compute_quality_metrics,
    write_markdown_report,
)

__all__ = ["main"]

_DEFAULT_CONFIG = Path(__file__).parent / "settings/fast_rt_high_throughput.yaml"
_QUALITY_CONFIG = Path(__file__).parent / "settings/fast_rt_quality_preserving.yaml"
_PROFILES = ("default", "high-throughput", "quality-preserving")
_COMPARISONS = (
    ("default", "high-throughput"),
    ("default", "quality-preserving"),
    ("high-throughput", "quality-preserving"),
)


def _settings(path: Path) -> tuple[BenchmarkCfg, dict[str, Any], list[int], int]:
    """Decode workload separately from the production rendering configuration."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    render = raw.pop("render_cfg")
    quality_render = raw.pop("quality_render_cfg", None)
    counts = raw.pop("env_sweep")
    repeats = raw.pop("repeats")
    if raw.pop("enable_depth") is not True or raw.get("camera_count") != 1:
        raise ValueError("The profile comparison requires one RGB-D camera per Arena.")
    if not counts or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError("env_sweep must contain positive integers.")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be positive.")
    for rendering in (render, quality_render):
        if rendering is not None and rendering.get("renderer") != "fast-rt":
            raise ValueError("All profiles must use fast-rt.")
    cfg = BenchmarkCfg.from_mapping(
        dict(
            raw,
            renderer="fast-rt",
            denoising_modes=["dlss"],
            reference_mode="dlss",
            quality_frames=1,
        )
    )
    return cfg, render, counts, repeats


def _worker(path: Path, profile: str, num_envs: int, result: Path) -> None:
    """Measure completed static render and GPU observation calls for one profile."""
    import dexsim
    import torch

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg
    from embodichain.lab.sim.sensors import CameraCfg

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261117)
    np.random.seed(20261117)
    cfg, render, _, _ = _settings(path)
    if profile == "quality-preserving":
        render = yaml.safe_load(path.read_text(encoding="utf-8"))["quality_render_cfg"]
    if torch.device(cfg.device) != torch.device("cuda", cfg.gpu_id):
        raise ValueError(
            "The comparison requires observations on the selected CUDA GPU."
        )
    render_cfg = (
        RenderCfg(renderer="fast-rt")
        if profile == "default"
        else RenderCfg.from_dict(render)
    )
    native = dexsim.WorldConfig()
    render_cfg.apply_to_dexsim_config(native)
    effective = {
        "dlss": {
            name: getattr(native.dlss_config, name, None)
            for name in (
                "dlss_quality",
                "tiled_enabled",
                "tiled_gutter_pixels",
                "tiled_overscan_pixels",
                "frame_time_delta_ms",
            )
        },
        "min_bounces": native.raytrace_config.min_bounces,
        "max_bounces": native.raytrace_config.max_bounces,
        "render_iterations_per_frame": native.raytrace_config.render_iterations_per_frame,
        "tone_mapping_enabled": native.postprocess_config.tone_mapping_enabled,
        "tone_mapping_type": str(native.postprocess_config.tone_mapping_type),
        "tone_mapping_exposure": native.postprocess_config.tone_mapping_exposure,
    }
    torch.cuda.set_device(cfg.gpu_id)
    torch.cuda.synchronize(cfg.gpu_id)
    before = _memory_snapshot()
    start_setup = time.perf_counter()
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device=cfg.device,
            gpu_id=cfg.gpu_id,
            num_envs=num_envs,
            enable_entity_gizmo=False,
            robot_ik_gizmo=None,
            startup_summary="off",
            dexsim_startup_info=False,
            render_cfg=render_cfg,
        )
    )
    try:
        _build_benchmark_scene(sim, cfg.scene)
        sim.prepare()
        sim.get_world().enable_physics(False)
        camera = sim.add_sensor(
            CameraCfg(
                uid="benchmark_camera",
                width=cfg.width,
                height=cfg.height,
                intrinsics=(
                    cfg.width * 0.78,
                    cfg.width * 0.78,
                    cfg.width / 2,
                    cfg.height / 2,
                ),
                near=0.05,
                far=20.0,
                enable_color=True,
                enable_depth=True,
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=_camera_pose(0, 1),
                    target=(0.0, 0.35, 0.55),
                    up=(0.0, 0.0, 1.0),
                ),
            )
        )
        group_ids = list(camera.group_ids)
        if len(group_ids) != 1:
            raise RuntimeError("The FastRT comparison requires one camera group.")
        setup_s = time.perf_counter() - start_setup

        def frame() -> tuple[float, float]:
            """Time one render, then the complete GPU observation without CPU copies."""
            torch.cuda.synchronize(cfg.gpu_id)
            started = time.perf_counter()
            sim.render_camera_group(group_ids)
            torch.cuda.synchronize(cfg.gpu_id)
            rendered = time.perf_counter()
            camera.update(fetch_only=True)
            torch.cuda.synchronize(cfg.gpu_id)
            completed = time.perf_counter()
            return (rendered - started) * 1000, (completed - started) * 1000

        prepared = _memory_snapshot()
        for _ in range(max(1, cfg.warmup_frames)):
            frame()
        samples = [frame() for _ in range(cfg.measure_frames)]
        after = _memory_snapshot()
        # Validation, memory instrumentation, encoding and CPU copies are untimed.
        data = camera.get_data()
        rgb = data["color"][..., :3].cpu().numpy().copy()
        depth = data["depth"].cpu().numpy().copy()
        if rgb.shape != (num_envs, cfg.height, cfg.width, 3):
            raise RuntimeError(f"Invalid RGB output shape: {rgb.shape}")
        valid_depth = np.isfinite(depth) & (depth > 0)
        valid_views = (rgb.std(axis=(1, 2, 3)) > 1) & valid_depth.any(axis=(1, 2))
        selected = [0, num_envs - 1]
        np.savez_compressed(
            result.with_suffix(".npz"),
            rgb=rgb[selected],
            depth=depth[selected],
            selected_envs=selected,
        )
        payload = {
            "profile": profile,
            "num_envs": num_envs,
            "group_ids": group_ids,
            "render_ms": [s[0] for s in samples],
            "complete_ms": [s[1] for s in samples],
            "setup_s": setup_s,
            "valid_views": int(valid_views.sum()),
            "success_rate": float(valid_views.mean() * 100),
            "depth_valid_fraction": float(valid_depth.mean()),
            "cpu_delta_mb": after["cpu_mb"] - before["cpu_mb"],
            "gpu_delta_mb": after["gpu_mb"] - before["gpu_mb"],
            "peak_gpu_mb": max(prepared["gpu_mb"], after["gpu_mb"]),
            "render_cfg": render_cfg.to_dict(),
            "effective_native": effective,
            "sdk": importlib.metadata.version("dexsim_engine"),
            "sdk_path": dexsim.__file__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "cpu_threads": torch.get_num_threads(),
        }
        result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if not valid_views.all():
            raise RuntimeError(f"Only {valid_views.sum()}/{num_envs} views are valid.")
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _gain_interval(
    baseline: list[float], candidate: list[float]
) -> tuple[float, float, float]:
    """Use independent process pairs for a Student-t interval of log throughput ratios."""
    from scipy.stats import t

    ratios = [math.log(a / b) for a, b in zip(baseline, candidate, strict=True)]
    mean = statistics.mean(ratios)
    half = (
        float(t.ppf(0.975, len(ratios) - 1))
        * statistics.stdev(ratios)
        / math.sqrt(len(ratios))
        if len(ratios) > 1
        else 0.0
    )
    return tuple(100 * math.expm1(x) for x in (mean, mean - half, mean + half))


def _report(root: Path, counts: list[int], repeats: int) -> Path:
    """Aggregate matched process blocks, render three tables and export a plot."""
    import cv2

    identity = {
        profile: json.loads((root / f"{counts[0]}_0_{profile}.json").read_text())
        for profile in _PROFILES
    }
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["runtime"] = {
        key: identity["default"][key]
        for key in ("sdk", "sdk_path", "torch", "python", "cpu_threads")
    }
    manifest["effective_native"] = {
        profile: run["effective_native"] for profile, run in identity.items()
    }
    manifest["report_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    perf, quality, gains = [], [], []
    for count in counts:
        per_profile = {}
        for profile in _PROFILES:
            runs = [
                json.loads((root / f"{count}_{r}_{profile}.json").read_text())
                for r in range(repeats)
            ]
            per_profile[profile] = runs
            render = [statistics.mean(run["render_ms"]) for run in runs]
            complete = [statistics.mean(run["complete_ms"]) for run in runs]
            perf.append(
                {
                    "arenas": count,
                    "profile": profile,
                    "processes": repeats,
                    "render_ms": round(statistics.mean(render), 3),
                    "render_median_ms": round(
                        statistics.median(
                            statistics.median(run["render_ms"]) for run in runs
                        ),
                        3,
                    ),
                    "cost_time_ms": round(statistics.mean(complete), 3),
                    "complete_median_ms": round(
                        statistics.median(
                            statistics.median(run["complete_ms"]) for run in runs
                        ),
                        3,
                    ),
                    "complete_p95_ms": round(
                        statistics.median(
                            float(np.percentile(run["complete_ms"], 95)) for run in runs
                        ),
                        3,
                    ),
                    "render_images_per_s": round(
                        1000 * count / statistics.mean(render), 1
                    ),
                    "complete_images_per_s": round(
                        1000 * count / statistics.mean(complete), 1
                    ),
                    "cpu_delta_mb": round(
                        statistics.median(run["cpu_delta_mb"] for run in runs), 1
                    ),
                    "gpu_delta_mb": round(
                        statistics.median(run["gpu_delta_mb"] for run in runs), 1
                    ),
                    "peak_gpu_mb": round(max(run["peak_gpu_mb"] for run in runs), 1),
                    "success_rate": min(run["success_rate"] for run in runs),
                }
            )
        for reference, candidate in _COMPARISONS:
            gain = _gain_interval(
                *[
                    [statistics.mean(run["render_ms"]) for run in per_profile[p]]
                    for p in (reference, candidate)
                ]
            )
            gains.append(
                {
                    "arenas": count,
                    "reference": reference,
                    "candidate": candidate,
                    "gain_pct": gain[0],
                    "lower95": gain[1],
                    "upper95": gain[2],
                }
            )
            with (
                np.load(root / f"{count}_0_{reference}.npz") as a,
                np.load(root / f"{count}_0_{candidate}.npz") as b,
            ):
                mask_a = np.isfinite(a["depth"]) & (a["depth"] > 0)
                mask_b = np.isfinite(b["depth"]) & (b["depth"] > 0)
                common = mask_a & mask_b
                quality.append(
                    {
                        "arenas": count,
                        "reference": reference,
                        "candidate": candidate,
                        "success_rate": min(
                            run["success_rate"]
                            for p in (reference, candidate)
                            for run in per_profile[p]
                        ),
                        "render_gain_pct": round(gain[0], 2),
                        "paired_95_interval_pct": (
                            f"[{gain[1]:.2f}, {gain[2]:.2f}]"
                            if repeats > 1
                            else "N/A (one pair)"
                        ),
                        "depth_mask_iou": float(common.sum() / (mask_a | mask_b).sum()),
                        "depth_max_abs_m": (
                            float(
                                np.max(np.abs(a["depth"][common] - b["depth"][common]))
                            )
                            if common.any()
                            else None
                        ),
                        "rgb_ssim": round(
                            compute_quality_metrics(a["rgb"][0], b["rgb"][0])["ssim"], 4
                        ),
                        "rgb_ssim_last": round(
                            compute_quality_metrics(a["rgb"][-1], b["rgb"][-1])["ssim"],
                            4,
                        ),
                    }
                )
        if count == counts[-1]:
            previews = []
            for profile in _PROFILES:
                with np.load(root / f"{count}_0_{profile}.npz") as frames:
                    previews.append(frames["rgb"][0])
            cv2.imwrite(
                str(root / "comparison.png"),
                cv2.cvtColor(np.concatenate(previews, axis=1), cv2.COLOR_RGB2BGR),
            )
    leaderboard = []
    for profile in _PROFILES:
        rows = [r for r in perf if r["profile"] == profile]
        leaderboard.append(
            {
                "profile": profile,
                "success_rate": min(r["success_rate"] for r in rows),
                "geomean_render_images_per_s": round(
                    statistics.geometric_mean(r["render_images_per_s"] for r in rows), 1
                ),
            }
        )
    leaderboard.sort(
        key=lambda r: (-r["success_rate"], -r["geomean_render_images_per_s"])
    )
    summary = {
        "performance": perf,
        "quality": quality,
        "gains": gains,
        "leaderboard": leaderboard,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(root, perf, gains)
    return write_markdown_report(
        output_path=root / "report.md",
        perf_rows=perf,
        metric_rows=quality,
        leaderboard_rows=leaderboard,
        settings=manifest,
        notes=[
            "Static fixed_reflective scene; one 640x480 RGB-D camera/Arena; no physics stepping.",
            "render_ms and cost_time_ms aggregate process means; median columns aggregate process medians. Throughput uses the total completed images divided by total measured time.",
            "Render time includes state publication, ordered group render and CUDA completion; complete time additionally includes GPU sensor fetch.",
            "CPU readback, validation, image encoding and memory queries are outside timing. Memory is sampled before warmup and after measurement, with no query between warmup and timing. peak_gpu_mb is sampled process VRAM, not an allocator peak.",
            "Each profile/repeat uses a fresh process. Profile order rotates and reverses across matched blocks. Confidence intervals pair process means within each block, not individual frames; they are not simultaneous intervals.",
            "The high-throughput profile changes path termination, border reconstruction and tone mapping: this is a settings tradeoff, not equal-quality acceleration.",
            "Rendering parameters match MR !1629; scene assets differ, and Arena counts/resolution follow this comparison. A released SDK without overscan already has it disabled; null native overscan means unavailable.",
            "The quality-preserving profile retains gutter=16; its other render settings match the high-throughput profile. It preserves a guard band, not equality to an independent image-quality reference.",
            "RGB preview: comparison.png (default, high-throughput, quality-preserving). success_rate denotes final-frame valid views across all Arenas, not per-frame image validation. Depth and RGB comparisons sample the first/last Arena in the first process block; rgb_ssim retains its first-view meaning and rgb_ssim_last covers the last view.",
        ],
    )


def _plot(root: Path, perf: list[dict[str, Any]], gains: list[dict[str, Any]]) -> None:
    """Export measured throughput and process-paired uncertainty as standalone figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for profile in _PROFILES:
        rows = [r for r in perf if r["profile"] == profile]
        axes[0].plot(
            [r["arenas"] for r in rows],
            [r["render_images_per_s"] for r in rows],
            "o-",
            label=profile,
        )
    axes[0].set(
        xlabel="Arenas",
        ylabel="Completed render images / s",
        title="FastRT · 640 × 480 · one RGB-D camera / Arena",
    )
    axes[0].legend()
    for profile in _PROFILES[1:]:
        rows = [
            r
            for r in gains
            if r["reference"] == "default" and r["candidate"] == profile
        ]
        axes[1].errorbar(
            [r["arenas"] for r in rows],
            [r["gain_pct"] for r in rows],
            yerr=[
                [r["gain_pct"] - r["lower95"] for r in rows],
                [r["upper95"] - r["gain_pct"] for r in rows],
            ],
            fmt="o-",
            capsize=4,
            label=profile,
        )
    axes[1].legend()
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].set(
        xlabel="Arenas",
        ylabel="Render throughput gain (%)",
        title=(
            "Profiles vs default · paired 95% CI"
            if perf[0]["processes"] > 1
            else "Profiles vs default · one block, no CI"
        ),
    )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(
            sorted({r["arenas"] for r in gains}),
            labels=[str(n) for n in sorted({r["arenas"] for r in gains})],
        )
        ax.grid(alpha=0.25)
    fig.savefig(root / "throughput.png", dpi=180)
    fig.savefig(root / "throughput.svg")
    plt.close(fig)


def main() -> None:
    """Run serial matched blocks or one isolated benchmark sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--quality-config", type=Path, default=_QUALITY_CONFIG)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--env-sweep", type=int, nargs="+")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup-frames", type=int)
    parser.add_argument("--measure-frames", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--profile", choices=_PROFILES)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.profile is None or args.num_envs is None or args.result is None:
            parser.error("worker requires profile, num-envs and result")
        _worker(args.config, args.profile, args.num_envs, args.result)
        return
    raw = yaml.safe_load(args.config.read_text())
    raw["quality_render_cfg"] = yaml.safe_load(args.quality_config.read_text())[
        "render_cfg"
    ]
    for name in ("env_sweep", "repeats", "warmup_frames", "measure_frames"):
        if (value := getattr(args, name)) is not None:
            raw[name] = value
    root = args.output_root / f"camera_profiles_{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True)
    settings = root / "settings.json"
    settings.write_text(json.dumps(raw, indent=2) + "\n")
    cfg, _, counts, repeats = _settings(settings)
    import torch

    if not torch.cuda.is_available():
        print("Skipped: the FastRT comparison requires a CUDA GPU.")
        return
    manifest = {
        "gpu": _gpu_description(cfg.gpu_id),
        "settings": raw,
        "python": sys.executable,
        "platform": platform.platform(),
        "embodichain_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path("embodichain/lab/sim/cfg/rendering.py"),
                args.config,
                args.quality_config,
            )
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        EMBODICHAIN_SIM_EXIT_PROCESS="0",
    )
    for count in counts:
        for repeat in range(repeats):
            shift = repeat % len(_PROFILES)
            order = _PROFILES[shift:] + _PROFILES[:shift]
            if (repeat // len(_PROFILES)) % 2:
                order = order[::-1]
            for profile in order:
                if other := _other_gpu_processes():
                    raise RuntimeError(
                        f"GPU is shared by compute processes {other}; raw samples retained in {root}."
                    )
                result = root / f"{count}_{repeat}_{profile}.json"
                print(
                    f"Arena={count} block={repeat + 1}/{repeats} profile={profile}",
                    flush=True,
                )
                with result.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "scripts.benchmark.rendering.camera_profiles",
                            "--worker",
                            "--config",
                            str(settings),
                            "--profile",
                            profile,
                            "--num-envs",
                            str(count),
                            "--result",
                            str(result),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                if other := _other_gpu_processes():
                    raise RuntimeError(
                        f"GPU became shared by {other}; raw samples retained in {root}."
                    )
    print(f"Markdown report saved: {_report(root, counts, repeats)}", flush=True)


if __name__ == "__main__":
    main()
