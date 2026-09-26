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

"""Benchmark offscreen DLSS and NRD rendering against OptiX at one spp.

Run: python -m scripts.benchmark.rendering.offscreen_denoising
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import psutil

RenderMode = Literal["optix", "dlss", "nrd"]
BENCHMARK_MODES: tuple[RenderMode, ...] = ("optix", "dlss", "nrd")


@dataclass(frozen=True)
class BenchmarkCfg:
    """Runtime settings shared by every rendering mode worker."""

    width: int
    height: int
    warmup_frames: int
    measure_frames: int
    quality_frames: int
    dlss_quality: int
    gpu_id: int
    device: str


def compute_quality_metrics(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    """Return image similarity metrics for one RGB frame pair."""
    reference_rgb = np.asarray(reference)[..., :3].astype(np.float64) / 255.0
    candidate_rgb = np.asarray(candidate)[..., :3].astype(np.float64) / 255.0
    if reference_rgb.shape != candidate_rgb.shape:
        raise ValueError(
            "Reference and candidate images must have identical RGB shapes."
        )

    delta = reference_rgb - candidate_rgb
    mse = float(np.mean(np.square(delta)))
    mae = float(np.mean(np.abs(delta)))
    psnr_db = math.inf if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))

    c1 = 0.01**2
    c2 = 0.03**2
    channel_ssim: list[float] = []
    for channel in range(3):
        ref_channel = reference_rgb[..., channel]
        candidate_channel = candidate_rgb[..., channel]
        ref_mean = float(np.mean(ref_channel))
        candidate_mean = float(np.mean(candidate_channel))
        ref_variance = float(np.mean(np.square(ref_channel - ref_mean)))
        candidate_variance = float(
            np.mean(np.square(candidate_channel - candidate_mean))
        )
        covariance = float(
            np.mean((ref_channel - ref_mean) * (candidate_channel - candidate_mean))
        )
        numerator = (2.0 * ref_mean * candidate_mean + c1) * (2.0 * covariance + c2)
        denominator = (ref_mean**2 + candidate_mean**2 + c1) * (
            ref_variance + candidate_variance + c2
        )
        channel_ssim.append(numerator / denominator)

    reference_gray = np.mean(reference_rgb, axis=-1)
    candidate_gray = np.mean(candidate_rgb, axis=-1)
    reference_gradient = np.hypot(*np.gradient(reference_gray))
    candidate_gradient = np.hypot(*np.gradient(candidate_gray))
    edge_mae = float(np.mean(np.abs(reference_gradient - candidate_gradient)))
    return {
        "psnr_db": psnr_db,
        "ssim": float(np.mean(channel_ssim)),
        "mae": mae,
        "edge_mae": edge_mae,
    }


def build_leaderboard_rows(
    metric_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Rank algorithms by valid captures, similarity, and throughput."""
    ordered = sorted(
        metric_rows,
        key=lambda row: (
            float(row["success_rate"]),
            float(row["ssim"]),
            float(row.get("fps", 0.0)),
        ),
        reverse=True,
    )
    return [
        {
            "rank": rank,
            "algorithm": row["mode"],
            "overall_success_rate": row["success_rate"],
            "ssim": row["ssim"],
            "fps": row.get("fps", 0.0),
        }
        for rank, row in enumerate(ordered, start=1)
    ]


def _markdown_table(rows: list[dict[str, object]]) -> list[str]:
    """Render one Markdown table from rows sharing the same keys."""
    if not rows:
        return ["No rows were produced."]
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[header]) for header in headers) + " |" for row in rows
    )
    return lines


def write_markdown_report(
    *,
    output_path: Path,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    leaderboard_rows: list[dict[str, object]],
    notes: list[str] | None = None,
) -> Path:
    """Write one benchmark report containing exactly three Markdown tables."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Offscreen Denoising Benchmark Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Time & Memory",
        "",
        *_markdown_table(perf_rows),
        "",
        "## Success & Other Metrics",
        "",
        *_markdown_table(metric_rows),
        "",
        "## Leaderboard",
        "",
        *_markdown_table(leaderboard_rows),
    ]
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _process_gpu_memory_mb() -> float:
    """Return this process's GPU allocation reported by nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0.0
    current_pid = os.getpid()
    memory_mb = 0.0
    for line in result.stdout.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if len(columns) != 2:
            continue
        try:
            if int(columns[0]) == current_pid:
                memory_mb += float(columns[1])
        except ValueError:
            continue
    return memory_mb


def _gpu_description(gpu_id: int) -> str:
    """Return the selected GPU name, driver, and total memory when available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
                "--id",
                str(gpu_id),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return f"GPU {gpu_id} (metadata unavailable)"
    return result.stdout.strip() or f"GPU {gpu_id} (metadata unavailable)"


def _memory_snapshot() -> dict[str, float]:
    """Return CPU RSS, process GPU VRAM, and PyTorch GPU allocation in MB."""
    import torch

    mib = 1024**2
    return {
        "cpu_mb": psutil.Process(os.getpid()).memory_info().rss / mib,
        "gpu_mb": _process_gpu_memory_mb(),
        "torch_gpu_mb": (
            torch.cuda.memory_allocated() / mib if torch.cuda.is_available() else 0.0
        ),
    }


def _material(
    uid: str,
    color: tuple[float, float, float],
    *,
    roughness: float,
    metallic: float = 0.0,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Create one PBR material without importing simulation modules globally."""
    from embodichain.lab.sim.material import VisualMaterialCfg

    return VisualMaterialCfg(
        uid=uid,
        base_color=[*color, 1.0],
        roughness=roughness,
        metallic=metallic,
        emissive=list(emissive),
    )


def _add_scene(sim):
    """Build a deterministic scene stressing edges, roughness, and reflections."""
    from embodichain.lab.sim.cfg import LightCfg, RigidObjectCfg
    from embodichain.lab.sim.shapes import CubeCfg, SphereCfg

    def add_cube(
        uid: str,
        size: tuple[float, float, float],
        position: tuple[float, float, float],
        color: tuple[float, float, float],
        roughness: float,
        metallic: float = 0.0,
        *,
        body_type: str = "static",
        emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        return sim.add_rigid_object(
            RigidObjectCfg(
                uid=uid,
                shape=CubeCfg(
                    size=list(size),
                    visual_material=_material(
                        f"{uid}_material",
                        color,
                        roughness=roughness,
                        metallic=metallic,
                        emissive=emissive,
                    ),
                ),
                body_type=body_type,
                init_pos=position,
            )
        )

    add_cube(
        "floor",
        (5.0, 5.0, 0.08),
        (0.0, 0.0, -0.04),
        (0.32, 0.34, 0.38),
        0.72,
    )
    add_cube(
        "back_wall",
        (5.0, 0.08, 2.8),
        (0.0, 1.65, 1.4),
        (0.62, 0.65, 0.70),
        0.82,
    )
    add_cube(
        "left_wall",
        (0.08, 3.4, 2.8),
        (-2.1, 0.0, 1.4),
        (0.25, 0.30, 0.42),
        0.78,
    )

    sphere_specs = (
        ("chrome", (-1.15, 0.30, 0.42), (0.72, 0.76, 0.82), 0.05, 1.0),
        ("brushed", (-0.35, 0.45, 0.36), (0.75, 0.28, 0.08), 0.28, 1.0),
        ("ceramic", (0.45, 0.40, 0.32), (0.15, 0.55, 0.92), 0.18, 0.0),
        ("matte", (1.20, 0.28, 0.28), (0.75, 0.72, 0.16), 0.85, 0.0),
    )
    for uid, position, color, roughness, metallic in sphere_specs:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid=uid,
                shape=SphereCfg(
                    radius=position[2],
                    resolution=64,
                    visual_material=_material(
                        f"{uid}_material",
                        color,
                        roughness=roughness,
                        metallic=metallic,
                    ),
                ),
                body_type="static",
                init_pos=(position[0], position[1], position[2]),
            )
        )

    for index in range(11):
        x_position = -1.35 + index * 0.27
        add_cube(
            f"slat_{index}",
            (0.035, 0.16, 0.86 - index * 0.035),
            (x_position, 1.20, 0.43 - index * 0.0175),
            (0.92 if index % 2 == 0 else 0.08,) * 3,
            0.45,
        )

    add_cube(
        "emissive_panel",
        (0.85, 0.04, 0.28),
        (1.35, 1.52, 1.30),
        (0.08, 0.30, 0.95),
        0.3,
        emissive=(0.08, 0.30, 0.95),
    )
    moving_object = add_cube(
        "moving_cube",
        (0.34, 0.34, 0.34),
        (-1.0, -0.55, 0.22),
        (0.95, 0.12, 0.10),
        0.12,
        metallic=0.35,
        body_type="static",
    )

    sim.add_light(
        LightCfg(
            uid="key_light",
            light_type="rect",
            init_pos=(0.2, -0.7, 2.6),
            direction=(0.0, 0.35, -1.0),
            color=(1.0, 0.90, 0.78),
            intensity=180.0,
            rect_width=1.4,
            rect_height=1.0,
        )
    )
    sim.add_light(
        LightCfg(
            uid="fill_light",
            light_type="point",
            init_pos=(-1.5, -0.8, 1.5),
            color=(0.35, 0.55, 1.0),
            intensity=55.0,
            radius=5.0,
        )
    )
    return moving_object


def _build_simulation(mode: RenderMode, cfg: BenchmarkCfg):
    """Create one isolated simulation and its offscreen camera."""
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import DenoisingCfg, DLSSCfg, NRDCfg, RenderCfg
    from embodichain.lab.sim.sensors import CameraCfg

    sim_cfg = SimulationManagerCfg(
        width=cfg.width,
        height=cfg.height,
        headless=True,
        device=cfg.device,
        gpu_id=cfg.gpu_id,
        num_envs=1,
        enable_entity_gizmo=False,
        robot_ik_gizmo=None,
        startup_summary="off",
        dexsim_startup_info=False,
        render_cfg=RenderCfg(
            renderer="fast-rt",
            spp=1,
            denoising=DenoisingCfg(window="off", offscreen=mode),
            dlss=DLSSCfg(
                dlss_quality=cfg.dlss_quality,
                tiled_enabled=False,
                frame_time_delta_ms=1000.0 / 30.0,
            ),
            nrd=NRDCfg(taa_tone_mapping_enabled=False),
        ),
    )
    sim = SimulationManager(sim_cfg)
    moving_object = _add_scene(sim)
    sim.prepare()
    camera = sim.add_sensor(
        CameraCfg(
            uid="benchmark_camera",
            width=cfg.width,
            height=cfg.height,
            intrinsics=(
                cfg.width * 0.78,
                cfg.width * 0.78,
                cfg.width / 2.0,
                cfg.height / 2.0,
            ),
            near=0.05,
            far=20.0,
            enable_color=True,
            extrinsics=CameraCfg.ExtrinsicsCfg(
                eye=(3.0, -3.1, 1.85),
                target=(0.0, 0.35, 0.55),
                up=(0.0, 0.0, 1.0),
            ),
        )
    )
    return sim, camera, moving_object


def _set_frame_state(sim, camera, moving_object, frame_index: int) -> None:
    """Apply the deterministic camera orbit and moving-object trajectory."""
    import torch

    phase = frame_index * 0.085
    object_pose = torch.tensor(
        [
            [
                -0.95 + 1.9 * (0.5 + 0.5 * math.sin(phase)),
                -0.62 + 0.12 * math.cos(phase * 0.7),
                0.22,
                0.0,
                0.0,
                math.sin(phase * 0.5),
                math.cos(phase * 0.5),
            ]
        ],
        dtype=torch.float32,
        device=sim.device,
    )
    moving_object.set_local_pose(object_pose)

    eye = torch.tensor(
        [
            [
                3.0 + 0.16 * math.sin(phase * 0.5),
                -3.1 + 0.12 * math.cos(phase * 0.5),
                1.85 + 0.05 * math.sin(phase * 0.35),
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([[0.0, 0.35, 0.55]], dtype=torch.float32)
    camera.look_at(eye, target)
    sim.sync_render_state()


def _capture_frame(camera) -> np.ndarray:
    """Render one frame and return the first camera's RGBA image on CPU."""
    import torch

    camera.update()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return camera.get_data()["color"][0].detach().cpu().numpy().copy()


def _run_mode_worker(mode: RenderMode, cfg: BenchmarkCfg, run_dir: Path) -> None:
    """Render one mode in an isolated process and persist raw measurements."""
    import torch

    sim = None
    camera = None
    try:
        sim, camera, moving_object = _build_simulation(mode, cfg)
        _set_frame_state(sim, camera, moving_object, 0)
        for _ in range(cfg.warmup_frames):
            _capture_frame(camera)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        before = _memory_snapshot()
        peak_gpu_mb = before["gpu_mb"]
        elapsed_ms: list[float] = []
        frames: list[np.ndarray] = []
        for frame_index in range(cfg.measure_frames):
            _set_frame_state(sim, camera, moving_object, frame_index + 1)
            start = time.perf_counter()
            frame = _capture_frame(camera)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)
            if frame_index < cfg.quality_frames:
                frames.append(frame)
            peak_gpu_mb = max(peak_gpu_mb, _process_gpu_memory_mb())
        after = _memory_snapshot()
        torch_peak_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        )
        np.savez_compressed(run_dir / f"{mode}_frames.npz", frames=np.stack(frames))
        metrics = {
            "mode": mode,
            "captured_frames": len(elapsed_ms),
            "expected_frames": cfg.measure_frames,
            "mean_ms": float(np.mean(elapsed_ms)),
            "median_ms": float(np.median(elapsed_ms)),
            "p95_ms": float(np.percentile(elapsed_ms, 95)),
            "fps": float(1000.0 / np.median(elapsed_ms)),
            "cpu_delta_mb": after["cpu_mb"] - before["cpu_mb"],
            "gpu_delta_mb": after["gpu_mb"] - before["gpu_mb"],
            "peak_gpu_mb": peak_gpu_mb,
            "torch_peak_gpu_mb": torch_peak_mb,
        }
        (run_dir / f"{mode}_performance.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{mode:>6}: median={metrics['median_ms']:.2f} ms  "
            f"p95={metrics['p95_ms']:.2f} ms  fps={metrics['fps']:.2f}"
        )
    finally:
        if sim is not None:
            sim.destroy(exit_process=False)
            del camera
            del sim


def _temporal_delta_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return the mean error between consecutive-frame RGB changes."""
    if len(reference) < 2:
        return 0.0
    reference_float = reference[..., :3].astype(np.float32) / 255.0
    candidate_float = candidate[..., :3].astype(np.float32) / 255.0
    reference_delta = np.diff(reference_float, axis=0)
    candidate_delta = np.diff(candidate_float, axis=0)
    return float(np.mean(np.abs(reference_delta - candidate_delta)))


def _aggregate_quality(
    mode: RenderMode, reference: np.ndarray, candidate: np.ndarray, fps: float
) -> dict[str, object]:
    """Aggregate quality metrics across corresponding sequence frames."""
    frame_metrics = [
        compute_quality_metrics(ref_frame, candidate_frame)
        for ref_frame, candidate_frame in zip(reference, candidate, strict=True)
    ]
    psnr_values = [float(row["psnr_db"]) for row in frame_metrics]
    finite_psnr = [value for value in psnr_values if math.isfinite(value)]
    psnr_mean = math.inf if not finite_psnr else float(np.mean(finite_psnr))
    return {
        "mode": mode,
        "success_rate": 100.0 * len(candidate) / len(reference),
        "psnr_db": round(psnr_mean, 4) if math.isfinite(psnr_mean) else "inf",
        "ssim": round(float(np.mean([row["ssim"] for row in frame_metrics])), 6),
        "mae": round(float(np.mean([row["mae"] for row in frame_metrics])), 6),
        "edge_mae": round(
            float(np.mean([row["edge_mae"] for row in frame_metrics])), 6
        ),
        "temporal_delta_mae": round(_temporal_delta_error(reference, candidate), 6),
        "fps": round(fps, 3),
    }


def _write_comparison_image(run_dir: Path, frame_index: int = 0) -> Path:
    """Write an OptiX/DLSS/NRD frame row and amplified difference row."""
    frames = {
        mode: np.load(run_dir / f"{mode}_frames.npz")["frames"][frame_index]
        for mode in BENCHMARK_MODES
    }
    reference = frames["optix"][..., :3]
    top_row: list[np.ndarray] = []
    bottom_row: list[np.ndarray] = []
    for mode in BENCHMARK_MODES:
        rgb = frames[mode][..., :3]
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            image,
            mode,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        top_row.append(image)
        difference = np.clip(
            np.abs(rgb.astype(np.int16) - reference.astype(np.int16)) * 4,
            0,
            255,
        ).astype(np.uint8)
        difference = cv2.cvtColor(difference, cv2.COLOR_RGB2BGR)
        cv2.putText(
            difference,
            "|delta| x4",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        bottom_row.append(difference)
    comparison = np.concatenate(
        (np.concatenate(top_row, axis=1), np.concatenate(bottom_row, axis=1)),
        axis=0,
    )
    path = run_dir / "comparison.png"
    cv2.imwrite(str(path), comparison)
    return path


def _worker_command(mode: RenderMode, cfg: BenchmarkCfg, run_dir: Path) -> list[str]:
    """Build the isolated worker command for one render mode."""
    return [
        sys.executable,
        "-m",
        "scripts.benchmark.rendering.offscreen_denoising",
        "--worker-mode",
        mode,
        "--run-dir",
        str(run_dir),
        "--width",
        str(cfg.width),
        "--height",
        str(cfg.height),
        "--warmup-frames",
        str(cfg.warmup_frames),
        "--measure-frames",
        str(cfg.measure_frames),
        "--quality-frames",
        str(cfg.quality_frames),
        "--dlss-quality",
        str(cfg.dlss_quality),
        "--gpu-id",
        str(cfg.gpu_id),
        "--device",
        cfg.device,
    ]


def run_all_benchmarks(cfg: BenchmarkCfg, output_root: Path) -> Path:
    """Run isolated mode workers and write images plus one Markdown report."""
    import dexsim

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"offscreen_denoising_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 60)
    print("Offscreen Denoising Quality and Efficiency Benchmark")
    print("=" * 60)
    print(
        f"Resolution={cfg.width}x{cfg.height}, spp=1, "
        f"reference=optix, quality frames={cfg.quality_frames}"
    )
    for mode in BENCHMARK_MODES:
        print(f"\n=== {mode.upper()} offscreen worker ===")
        subprocess.run(_worker_command(mode, cfg, run_dir), check=True)

    performance = {
        mode: json.loads(
            (run_dir / f"{mode}_performance.json").read_text(encoding="utf-8")
        )
        for mode in BENCHMARK_MODES
    }
    frames = {
        mode: np.load(run_dir / f"{mode}_frames.npz")["frames"]
        for mode in BENCHMARK_MODES
    }
    reference = frames["optix"]
    metric_rows = [
        _aggregate_quality(
            mode,
            reference,
            frames[mode],
            float(performance[mode]["fps"]),
        )
        for mode in BENCHMARK_MODES
    ]
    perf_rows = [
        {
            "mode": mode,
            "cost_time_ms": round(float(performance[mode]["median_ms"]), 3),
            "mean_ms": round(float(performance[mode]["mean_ms"]), 3),
            "p95_ms": round(float(performance[mode]["p95_ms"]), 3),
            "fps": round(float(performance[mode]["fps"]), 3),
            "cpu_delta_mb": round(float(performance[mode]["cpu_delta_mb"]), 3),
            "gpu_delta_mb": round(float(performance[mode]["gpu_delta_mb"]), 3),
            "peak_gpu_mb": round(float(performance[mode]["peak_gpu_mb"]), 3),
            "torch_peak_gpu_mb": round(
                float(performance[mode]["torch_peak_gpu_mb"]), 3
            ),
        }
        for mode in BENCHMARK_MODES
    ]
    leaderboard_rows = build_leaderboard_rows(metric_rows)
    comparison_path = _write_comparison_image(
        run_dir, frame_index=min(cfg.quality_frames // 2, cfg.quality_frames - 1)
    )
    report_path = write_markdown_report(
        output_path=run_dir / "report.md",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            "Quality values are consistency metrics relative to OptiX denoising at 1 spp, not absolute ground-truth fidelity.",
            "DLSS and NRD use the same 1 spp input, output resolution, camera path, and DLSS quality preset.",
            f"Run settings: {cfg.width}x{cfg.height}, warmup={cfg.warmup_frames}, measured={cfg.measure_frames}, quality frames={cfg.quality_frames}, DLSS quality={cfg.dlss_quality}.",
            f"GPU {cfg.gpu_id}: {_gpu_description(cfg.gpu_id)}; DexSim {getattr(dexsim, '__version__', 'unknown')}.",
            "SSIM is computed over the complete RGB frame; temporal_delta_mae compares corresponding consecutive-frame changes.",
            "Timing excludes scene/world setup and includes offscreen camera rendering plus GPU completion.",
            "peak_gpu_mb is process VRAM from nvidia-smi; torch_peak_gpu_mb covers PyTorch-managed allocations only.",
            f"Comparison image: {comparison_path.name}",
        ],
    )
    print("\n" + "=" * 60)
    print(f"Markdown report saved: {report_path}")
    print(f"Comparison image saved: {comparison_path}")
    print("=" * 60)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for parent and worker execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--warmup-frames", type=int, default=12)
    parser.add_argument("--measure-frames", type=int, default=40)
    parser.add_argument("--quality-frames", type=int, default=12)
    parser.add_argument("--dlss-quality", type=int, default=3, choices=range(-1, 6))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-mode", choices=BENCHMARK_MODES, default=None, help=argparse.SUPPRESS
    )
    return parser


def main() -> None:
    """Run the benchmark parent or one isolated rendering worker."""
    args = build_parser().parse_args()
    if args.measure_frames < 1:
        raise ValueError("--measure-frames must be positive.")
    if not 1 <= args.quality_frames <= args.measure_frames:
        raise ValueError("--quality-frames must be within --measure-frames.")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative.")
    cfg = BenchmarkCfg(
        width=args.width,
        height=args.height,
        warmup_frames=args.warmup_frames,
        measure_frames=args.measure_frames,
        quality_frames=args.quality_frames,
        dlss_quality=args.dlss_quality,
        gpu_id=args.gpu_id,
        device=args.device,
    )
    if args.worker_mode is not None:
        if args.run_dir is None:
            raise ValueError("--run-dir is required for worker mode.")
        _run_mode_worker(args.worker_mode, cfg, args.run_dir)
        return
    run_all_benchmarks(cfg, args.output_root)


if __name__ == "__main__":
    main()
