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

"""Compare separate and shared native targets for three equal-resolution RGB cameras.

Run: python -m scripts.benchmark.rendering.camera_grouping
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.benchmark.rendering.offscreen_denoising import (
    BenchmarkCfg,
    _build_benchmark_scene,
    _camera_pose,
    _gpu_description,
    _set_frame_state,
    compute_quality_metrics,
)

__all__ = ["main"]

_DEFAULT_CONFIG = Path(__file__).parent / "settings" / "three_rgb_640x480.yaml"


def _other_gpu_processes() -> list[int]:
    """Return unrelated CUDA process IDs before a paired sample starts."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    return [
        int(pid)
        for line in result.stdout.splitlines()
        if (pid := line.strip()).isdigit() and int(pid) != os.getpid()
    ]


def _settings(path: Path) -> tuple[BenchmarkCfg, tuple[int, ...], int]:
    """Load the shared scene settings and the additional grouping sweep."""
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    counts = tuple(raw.pop("env_sweep"))
    repeats = int(raw.pop("repeats"))
    if (raw.pop("enable_color", None), raw.pop("enable_depth", None)) != (
        True,
        False,
    ):
        raise ValueError("The grouping comparison requires RGB without depth.")
    cfg = BenchmarkCfg.from_mapping(raw)
    if cfg.camera_count != 3 or cfg.renderer != "fast-rt":
        raise ValueError("The grouping comparison requires three FastRT cameras.")
    if any(type(count) is not int or count < 1 for count in counts) or not counts:
        raise ValueError("env_sweep must contain positive integers.")
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    return cfg, counts, repeats


def _worker(cfg: BenchmarkCfg, grouping: str, output: Path) -> None:
    """Measure one group layout in an isolated native World."""
    import psutil
    import torch

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import DenoisingCfg, DLSSCfg, RenderCfg
    from embodichain.lab.sim.sensors import CameraCfg

    if grouping not in {"separate", "shared"}:
        raise ValueError("grouping must be separate or shared.")
    torch.manual_seed(123)
    np.random.seed(123)
    sim_cfg = SimulationManagerCfg(
        width=cfg.width,
        height=cfg.height,
        headless=True,
        device=cfg.device,
        gpu_id=cfg.gpu_id,
        num_envs=cfg.num_envs,
        enable_entity_gizmo=False,
        robot_ik_gizmo=None,
        startup_summary="off",
        dexsim_startup_info=False,
        render_cfg=RenderCfg(
            renderer="fast-rt",
            spp=cfg.spp,
            denoising=DenoisingCfg(window="off", offscreen="dlss"),
            dlss=DLSSCfg(dlss_quality=cfg.dlss_quality, tiled_enabled=True),
        ),
    )
    sim = SimulationManager(sim_cfg)
    try:
        moving_object = _build_benchmark_scene(sim, cfg.scene)
        sim.prepare()
        camera_cfgs = [
            CameraCfg(
                uid=f"benchmark_camera_{index}",
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
                enable_depth=False,
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=_camera_pose(index, 3),
                    target=(0.0, 0.35, 0.55),
                    up=(0.0, 0.0, 1.0),
                ),
            )
            for index in range(3)
        ]
        if grouping == "shared":
            sim._plan_camera_groups(camera_cfgs)
        cameras = [sim.add_sensor(camera_cfg) for camera_cfg in camera_cfgs]
        group_ids = list(dict.fromkeys(gid for c in cameras for gid in c.group_ids))

        def capture() -> None:
            """Render each target once and fetch each sensor's RGB view."""
            sim.render_camera_group(group_ids)
            for camera in cameras:
                camera.update(fetch_only=True)
            torch.cuda.synchronize(cfg.gpu_id)

        for index in range(cfg.warmup_frames):
            _set_frame_state(sim, cameras, moving_object, index)
            capture()

        render_ms: list[float] = []
        complete_ms: list[float] = []
        sample: np.ndarray | None = None
        free_min = torch.cuda.mem_get_info(cfg.gpu_id)[0]
        total_gpu = torch.cuda.mem_get_info(cfg.gpu_id)[1]
        used_before = total_gpu - free_min
        rss_before = psutil.Process().memory_info().rss
        for index in range(cfg.measure_frames):
            _set_frame_state(sim, cameras, moving_object, index + cfg.warmup_frames)
            torch.cuda.synchronize(cfg.gpu_id)
            start = time.perf_counter()
            sim.render_camera_group(group_ids)
            torch.cuda.synchronize(cfg.gpu_id)
            rendered = time.perf_counter()
            for camera in cameras:
                camera.update(fetch_only=True)
            torch.cuda.synchronize(cfg.gpu_id)
            completed = time.perf_counter()
            render_ms.append((rendered - start) * 1000)
            complete_ms.append((completed - start) * 1000)
            free_min = min(free_min, torch.cuda.mem_get_info(cfg.gpu_id)[0])
            if index == 0:
                sample = np.stack(
                    [
                        camera.get_data()["color"][0].cpu().numpy().copy()
                        for camera in cameras
                    ]
                )
        assert sample is not None
        if sample.shape != (3, cfg.height, cfg.width, 4):
            raise RuntimeError(f"Unexpected RGB sample shape {sample.shape}.")
        if np.any(sample.std(axis=(1, 2, 3)) < 1):
            raise RuntimeError("A camera returned a blank image.")
        sample_path = output.with_suffix(".npz")
        np.savez_compressed(sample_path, rgb=sample)
        output.write_text(
            json.dumps(
                {
                    "grouping": grouping,
                    "num_envs": cfg.num_envs,
                    "camera_count": 3,
                    "resolution": [cfg.width, cfg.height],
                    "channels": ["color"],
                    "group_ids": {c.uid: list(c.group_ids) for c in cameras},
                    "render_ms": render_ms,
                    "complete_ms": complete_ms,
                    "images_per_s": 3000
                    * cfg.num_envs
                    * len(complete_ms)
                    / sum(complete_ms),
                    "cpu_delta_mb": (psutil.Process().memory_info().rss - rss_before)
                    / 1048576,
                    "gpu_delta_mb": (
                        total_gpu - torch.cuda.mem_get_info(cfg.gpu_id)[0] - used_before
                    )
                    / 1048576,
                    "peak_gpu_mb": (total_gpu - free_min) / 1048576,
                    "gpu_free_min_mb": free_min / 1048576,
                    "sample": sample_path.name,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _aggregate(paths: list[Path], num_envs: int, grouping: str) -> dict[str, object]:
    """Aggregate complete frames and per-frame latency across repetitions."""
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    render = [sample for run in runs for sample in run["render_ms"]]
    complete = [sample for run in runs for sample in run["complete_ms"]]
    gpu_deltas = [run.get("gpu_delta_mb") for run in runs]
    return {
        "num_envs": num_envs,
        "grouping": grouping,
        "groups": len(set(gid for ids in runs[0]["group_ids"].values() for gid in ids)),
        "images_per_s": 3000 * num_envs * len(complete) / sum(complete),
        "render_median_ms": statistics.median(render),
        "complete_median_ms": statistics.median(complete),
        "complete_p95_ms": sorted(complete)[int(0.95 * (len(complete) - 1))],
        "cpu_delta_mb": statistics.median(run["cpu_delta_mb"] for run in runs),
        "gpu_delta_mb": (
            statistics.median(gpu_deltas)
            if all(value is not None for value in gpu_deltas)
            else None
        ),
        "peak_gpu_mb": max(run["peak_gpu_mb"] for run in runs),
        "gpu_free_min_mb": min(run["gpu_free_min_mb"] for run in runs),
        "runs": [str(path) for path in paths],
    }


def _paired_gain(
    separate: dict[str, object], shared: dict[str, object]
) -> tuple[float, float, float]:
    """Return the paired mean gain and a seeded bootstrap 95% interval."""
    gains = [
        json.loads(Path(str(b)).read_text(encoding="utf-8"))["images_per_s"]
        / json.loads(Path(str(a)).read_text(encoding="utf-8"))["images_per_s"]
        - 1
        for a, b in zip(separate["runs"], shared["runs"], strict=True)
    ]
    rng = random.Random(1729 + int(separate["num_envs"]))
    boot = sorted(
        statistics.mean(rng.choices(gains, k=len(gains))) for _ in range(10000)
    )
    return statistics.mean(gains), boot[250], boot[9749]


def _report(root: Path, rows: list[dict[str, object]]) -> Path:
    """Write throughput, image-quality and ranking tables."""
    by_key = {(int(row["num_envs"]), str(row["grouping"])): row for row in rows}
    lines = [
        "# Three 640×480 RGB camera groups",
        "",
        "Each camera contributes one RGB image per environment per completed frame.",
        "Timing excludes scene updates and includes GPU completion. GPU memory is a",
        "sampled steady-state whole-device peak and delta; CPU memory is a retained RSS delta.",
        "See settings.json for the runtime, hardware and seed.",
        "",
        "## Time & Memory",
        "",
        "| envs | grouping | groups | camera_images_per_s | render_ms | cost_time_ms | p95_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb | free_gpu_mb |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gpu_delta = row["gpu_delta_mb"]
        gpu_delta_text = "n/a" if gpu_delta is None else f"{float(gpu_delta):+.1f}"
        lines.append(
            f"| {row['num_envs']} | {row['grouping']} | {row['groups']} | "
            f"{float(row['images_per_s']):.1f} | {float(row['render_median_ms']):.2f} | "
            f"{float(row['complete_median_ms']):.2f} | {float(row['complete_p95_ms']):.2f} | "
            f"{float(row['cpu_delta_mb']):+.1f} | {gpu_delta_text} | "
            f"{float(row['peak_gpu_mb']):.0f} | "
            f"{float(row['gpu_free_min_mb']):.0f} |"
        )
    lines.extend(
        [
            "",
            "## Success & Other Metrics",
            "",
            "| envs | success_rate | paired_gain | gain_95pct_ci | first_frame_min_ssim |",
            "|---:|---:|---:|---|---:|",
        ]
    )
    acceptance = True
    for n in sorted({int(row["num_envs"]) for row in rows}):
        separate = by_key[n, "separate"]
        shared = by_key[n, "shared"]
        sample_a = np.load(Path(str(separate["runs"][0])).with_suffix(".npz"))["rgb"]
        sample_b = np.load(Path(str(shared["runs"][0])).with_suffix(".npz"))["rgb"]
        similarity = [
            [
                compute_quality_metrics(sample_a[i], sample_b[j])["ssim"]
                for j in range(3)
            ]
            for i in range(3)
        ]
        ssim = min(similarity[i][i] for i in range(3))
        if any(
            similarity[i][i] <= max(similarity[i][j] for j in range(3) if j != i)
            for i in range(3)
        ):
            raise RuntimeError(f"Camera image order changed at num_envs={n}.")
        gain, low, high = _paired_gain(separate, shared)
        acceptance &= low > 0 and float(shared["gpu_free_min_mb"]) >= 2048
        lines.append(
            f"| {n} | 1.00 | {gain:+.1%} | [{low:+.1%}, {high:+.1%}] | {ssim:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| rank | algorithm | overall_success_rate | mean_camera_images_per_s |",
            "|---:|---|---:|---:|",
        ]
    )
    ranked = sorted(
        ("separate", "shared"),
        key=lambda name: statistics.mean(
            float(row["images_per_s"]) for row in rows if row["grouping"] == name
        ),
        reverse=True,
    )
    for rank, name in enumerate(ranked, 1):
        mean = statistics.mean(
            float(row["images_per_s"]) for row in rows if row["grouping"] == name
        )
        lines.append(f"| {rank} | {name} | 1.00 | {mean:.1f} |")
    lines.extend(
        [
            "",
            "Shared-by-default gate: "
            + ("PASS" if acceptance else "FAIL")
            + "; all paired gain intervals must be positive and shared runs must leave at least 2 GiB free GPU memory.",
        ]
    )
    path = root / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    """Run isolated paired workers or one internal worker invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--grouping", choices=("separate", "shared"))
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args()
    cfg, counts, repeats = _settings(args.config)
    if args.worker:
        if args.grouping is None or args.num_envs is None or args.result is None:
            parser.error("worker requires grouping, num-envs and result")
        _worker(cfg.with_overrides(num_envs=args.num_envs), args.grouping, args.result)
        return
    if not args.allow_shared_gpu and (others := _other_gpu_processes()):
        raise RuntimeError(f"GPU is in use by other processes: {others}.")
    root = args.output_root / f"camera_grouping_{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=False)
    import dexsim
    import torch

    (root / "settings.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "settings": cfg.to_dict(),
                "env_sweep": counts,
                "repeats": repeats,
                "seed": 123,
                "python": sys.version,
                "dexsim": dexsim.__version__,
                "torch": torch.__version__,
                "gpu_name": torch.cuda.get_device_name(cfg.gpu_id),
                "gpu_driver": _gpu_description(cfg.gpu_id),
                "torch_cpu_threads": torch.get_num_threads(),
                "sim_cpu_threads": 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, object]] = []
    environment = os.environ.copy()
    repository = Path(__file__).resolve().parents[3]
    environment["PYTHONPATH"] = (
        f"{repository / 'extensions'}:{repository}:" + environment.get("PYTHONPATH", "")
    )
    for num_envs in counts:
        paths: dict[str, list[Path]] = {"separate": [], "shared": []}
        for repetition in range(repeats):
            order = (
                ("separate", "shared")
                if repetition % 2 == 0
                else ("shared", "separate")
            )
            for grouping in order:
                if not args.allow_shared_gpu and (others := _other_gpu_processes()):
                    raise RuntimeError(f"GPU became shared during the sweep: {others}.")
                path = root / f"env{num_envs}_{grouping}_{repetition}.json"
                command = [
                    sys.executable,
                    "-m",
                    "scripts.benchmark.rendering.camera_grouping",
                    "--worker",
                    "--config",
                    str(args.config.resolve()),
                    "--grouping",
                    grouping,
                    "--num-envs",
                    str(num_envs),
                    "--result",
                    str(path),
                ]
                print(
                    f"Running envs={num_envs} grouping={grouping} repetition={repetition}",
                    flush=True,
                )
                with (root / f"env{num_envs}_{grouping}_{repetition}.log").open(
                    "w"
                ) as log:
                    subprocess.run(
                        command,
                        check=True,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                if not args.allow_shared_gpu and (others := _other_gpu_processes()):
                    raise RuntimeError(f"GPU was shared during a sample: {others}.")
                paths[grouping].append(path)
        rows.extend(
            _aggregate(paths[name], num_envs, name) for name in ("separate", "shared")
        )
    (root / "results.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Markdown report saved: {_report(root, rows)}")


if __name__ == "__main__":
    main()
