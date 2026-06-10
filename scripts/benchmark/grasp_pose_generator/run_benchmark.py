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

"""Benchmark the grasp pose generator used by the grasp tutorial.

Measures how ``grasp_cfg.antipodal_sampler_cfg.n_sample`` and
``grasp_cfg.n_top_grasps`` affect antipodal sampling, grasp-pose selection
latency, and memory usage for the CoffeeCup mesh used in
``scripts/tutorials/grasp/grasp_generator.py``.
Run: python -m scripts.benchmark.grasp_pose_generator.run_benchmark
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
import trimesh

from embodichain.data import get_data_path
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GraspGenerator,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler import (
    AntipodalSamplerCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)

DEFAULT_N_SAMPLES = [5000, 10000, 20000]
DEFAULT_N_TOP_GRASPS = [30, 50, 100]
DEFAULT_MESH_SCALE = 4.0


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for grasp generator benchmarks."""
    parser = argparse.ArgumentParser(
        description="Benchmark CoffeeCup grasp pose generation parameter sweeps."
    )
    parser.add_argument(
        "--n-samples",
        nargs="+",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Values for grasp_cfg.antipodal_sampler_cfg.n_sample.",
    )
    parser.add_argument(
        "--n-top-grasps",
        nargs="+",
        type=int,
        default=DEFAULT_N_TOP_GRASPS,
        help="Values for grasp_cfg.n_top_grasps.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for tensors. Auto uses CUDA when available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used before each timed sampling/pose-selection call.",
    )
    parser.add_argument(
        "--skip-pose-selection",
        action="store_true",
        help="Only benchmark antipodal point generation.",
    )
    return parser.parse_args()


def _resolve_device(device_name: str) -> torch.device:
    """Resolve requested device name to a torch device."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _sync_cuda() -> None:
    """Synchronize CUDA stream when available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_gpu_memory() -> None:
    """Reset PyTorch peak GPU memory stats when CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_memory_mb() -> float:
    """Return peak GPU memory allocated by PyTorch in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def _memory_snapshot() -> dict[str, float]:
    """Return current process memory usage snapshot in MB."""
    process = psutil.Process(os.getpid())
    cpu_mb = process.memory_info().rss / 1024**2
    gpu_mb = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    return {"cpu_mb": cpu_mb, "gpu_mb": gpu_mb}


def _format_markdown_table(rows: list[dict[str, object]]) -> list[str]:
    """Format rows into a markdown table."""
    if not rows:
        return ["No data."]

    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return lines


def _write_markdown_report(
    benchmark_name: str,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    leaderboard_rows: list[dict[str, object]],
    notes: list[str] | None = None,
) -> Path:
    """Write benchmark results to a markdown report with three tables."""
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{benchmark_name}_{timestamp}.md"

    lines: list[str] = [
        f"# {benchmark_name} Benchmark Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Time & Memory",
        "",
    ]
    lines.extend(_format_markdown_table(perf_rows))
    lines.extend(["", "## Success & Other Metrics", ""])
    lines.extend(_format_markdown_table(metric_rows))
    lines.extend(["", "## Leaderboard", ""])
    lines.extend(_format_markdown_table(leaderboard_rows))

    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in notes])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _load_tutorial_mesh(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and scale the CoffeeCup mesh used by the tutorial."""
    mesh = trimesh.load_mesh(get_data_path("CoffeeCup/cup.ply"), process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * DEFAULT_MESH_SCALE
    triangles = np.asarray(mesh.faces, dtype=np.int64)
    return (
        torch.as_tensor(vertices, dtype=torch.float32, device=device),
        torch.as_tensor(triangles, dtype=torch.int64, device=device),
    )


def _set_seed(seed: int) -> None:
    """Set random seeds used by numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _remove_antipodal_cache_file(generator: GraspGenerator) -> None:
    """Remove the whole-mesh antipodal cache file used by GraspGenerator.generate."""
    cache_path = Path(generator._get_cache_dir(generator.vertices, generator.triangles))
    if cache_path.exists():
        cache_path.unlink()


def _clear_antipodal_cache(generator: GraspGenerator) -> None:
    """Remove cached antipodal pairs from disk and memory."""
    _remove_antipodal_cache_file(generator)
    generator._hit_point_pairs = None


def _make_grasp_generator(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
    n_sample: int,
    n_top_grasps: int,
) -> GraspGenerator:
    """Create the tutorial-style GraspGenerator for one parameter setting."""
    grasp_cfg = GraspGeneratorCfg(
        antipodal_sampler_cfg=AntipodalSamplerCfg(
            n_sample=n_sample,
            max_length=0.088,
            min_length=0.003,
        ),
        is_partial_annotate=False,
        is_filter_ground_collision=True,
        n_top_grasps=n_top_grasps,
    )
    gripper_collision_cfg = GripperCollisionCfg(
        max_open_length=0.088,
        finger_length=0.078,
        point_sample_dense=0.012,
    )
    return GraspGenerator(
        vertices=vertices,
        triangles=triangles,
        cfg=grasp_cfg,
        gripper_collision_cfg=gripper_collision_cfg,
    )


def _timed_generate(
    generator: GraspGenerator,
    seed: int,
) -> tuple[float, dict[str, float], float, torch.Tensor]:
    """Run a timed antipodal point generation call."""
    _clear_antipodal_cache(generator)
    _set_seed(seed)
    _reset_peak_gpu_memory()
    mem_before = _memory_snapshot()
    _sync_cuda()

    start = time.perf_counter()
    hit_point_pairs = generator.generate()
    _sync_cuda()
    elapsed = time.perf_counter() - start
    _remove_antipodal_cache_file(generator)

    mem_after = _memory_snapshot()
    deltas = {
        "cpu_mb": mem_after["cpu_mb"] - mem_before["cpu_mb"],
        "gpu_mb": mem_after["gpu_mb"] - mem_before["gpu_mb"],
    }
    return elapsed, deltas, _peak_gpu_memory_mb(), hit_point_pairs


def _timed_pose_selection(
    generator: GraspGenerator,
    n_top_grasps: int,
    seed: int,
) -> tuple[
    float, dict[str, float], float, bool, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Run a timed grasp-pose selection call."""
    generator.cfg.n_top_grasps = n_top_grasps
    object_pose = torch.eye(4, dtype=torch.float32, device=generator.device)
    object_pose[:3, 3] = torch.tensor(
        [0.55, 0.0, 0.01], dtype=torch.float32, device=generator.device
    )
    approach_direction = torch.tensor(
        [0.0, 0.0, -1.0], dtype=torch.float32, device=generator.device
    )

    _set_seed(seed)
    _reset_peak_gpu_memory()
    mem_before = _memory_snapshot()
    _sync_cuda()

    start = time.perf_counter()
    is_success, grasp_poses, open_lengths, total_cost = generator.get_valid_grasp_poses(
        object_pose=object_pose,
        approach_direction=approach_direction,
        visualize_collision=False,
    )
    _sync_cuda()
    elapsed = time.perf_counter() - start

    mem_after = _memory_snapshot()
    deltas = {
        "cpu_mb": mem_after["cpu_mb"] - mem_before["cpu_mb"],
        "gpu_mb": mem_after["gpu_mb"] - mem_before["gpu_mb"],
    }
    return (
        elapsed,
        deltas,
        _peak_gpu_memory_mb(),
        is_success,
        grasp_poses,
        open_lengths,
        total_cost,
    )


def _build_leaderboard_rows(
    pipeline_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Rank parameter settings by success rate and total runtime."""
    ranked = sorted(
        pipeline_rows,
        key=lambda row: (-float(row["success_rate"]), float(row["total_time_ms"])),
    )
    leaderboard_rows: list[dict[str, object]] = []
    for rank, row in enumerate(ranked, start=1):
        leaderboard_rows.append(
            {
                "rank": rank,
                "algorithm": (
                    f"n_sample={row['sample_size']}, "
                    f"n_top_grasps={row['n_top_grasps']}"
                ),
                "success_rate": f"{float(row['success_rate']):.2%}",
                "total_time_ms": f"{float(row['total_time_ms']):.6f}",
                "hit_pairs": row["hit_pairs"],
                "selected_grasps": row["selected_grasps"],
                "best_cost": row["best_cost"],
            }
        )
    return leaderboard_rows


def benchmark_grasp_pose_generator(
    n_samples: list[int],
    n_top_grasps_list: list[int],
    device: torch.device,
    seed: int,
    skip_pose_selection: bool,
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[str]
]:
    """Benchmark antipodal sampling and grasp-pose selection parameter sweeps."""
    vertices, triangles = _load_tutorial_mesh(device)
    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    pipeline_rows: list[dict[str, object]] = []
    notes: list[str] = [
        "Mesh source matches scripts/tutorials/grasp/grasp_generator.py: CoffeeCup/cup.ply scaled by 4.0.",
        "GraspGenerator.generate() cache is cleared before each timed generation so n_sample changes are measured.",
    ]

    if not torch.cuda.is_available() and not skip_pose_selection:
        skip_pose_selection = True
        notes.append(
            "Pose selection was skipped because collision checking requires CUDA in the current implementation."
        )

    print("\n=== Grasp Pose Generator Benchmark ===")
    print(f"Device: {device}")
    print(f"n_sample values: {', '.join(str(value) for value in n_samples)}")
    print(
        "n_top_grasps values: "
        f"{', '.join(str(value) for value in n_top_grasps_list)}"
    )

    for n_sample in n_samples:
        generator = _make_grasp_generator(
            vertices=vertices,
            triangles=triangles,
            n_sample=n_sample,
            n_top_grasps=max(n_top_grasps_list),
        )

        generate_elapsed, generate_mem, generate_peak_gpu, hit_point_pairs = (
            _timed_generate(generator, seed + n_sample)
        )
        hit_pairs = int(hit_point_pairs.shape[0])
        generate_ms = generate_elapsed * 1000.0

        print(f"**** n_sample={n_sample}: generated {hit_pairs} antipodal pairs")
        print(
            f"===Antipodal generate time: {generate_ms:.6f} ms  "
            f"CPU Δ={generate_mem['cpu_mb']:+.1f} MB  "
            f"GPU Δ={generate_mem['gpu_mb']:+.1f} MB  "
            f"peak GPU={generate_peak_gpu:.1f} MB"
        )

        perf_rows.append(
            {
                "sample_size": n_sample,
                "n_top_grasps": "-",
                "impl": "grasp_pose_generator",
                "component": "antipodal_generate",
                "cost_time_ms": f"{generate_ms:.6f}",
                "cpu_delta_mb": f"{generate_mem['cpu_mb']:.6f}",
                "gpu_delta_mb": f"{generate_mem['gpu_mb']:.6f}",
                "peak_gpu_mb": f"{generate_peak_gpu:.6f}",
            }
        )
        metric_rows.append(
            {
                "sample_size": n_sample,
                "n_top_grasps": "-",
                "impl": "grasp_pose_generator",
                "component": "antipodal_generate",
                "success_rate": f"{float(hit_pairs > 0):.6f}",
                "hit_pairs": hit_pairs,
                "selected_grasps": "-",
                "best_cost": "-",
            }
        )

        if skip_pose_selection:
            continue

        for n_top_grasps in n_top_grasps_list:
            (
                pose_elapsed,
                pose_mem,
                pose_peak_gpu,
                is_success,
                grasp_poses,
                open_lengths,
                total_cost,
            ) = _timed_pose_selection(
                generator=generator,
                n_top_grasps=n_top_grasps,
                seed=seed + n_sample + n_top_grasps,
            )
            pose_ms = pose_elapsed * 1000.0
            selected_grasps = int(grasp_poses.shape[0]) if is_success else 0
            best_cost = float(total_cost.min().item()) if is_success else float("inf")
            total_ms = generate_ms + pose_ms

            print(
                f"===Pose select n_top_grasps={n_top_grasps}: "
                f"{pose_ms:.6f} ms  selected={selected_grasps}  "
                f"success={float(is_success):.0f}"
            )
            print(
                "   "
                f"CPU Δ={pose_mem['cpu_mb']:+.1f} MB  "
                f"GPU Δ={pose_mem['gpu_mb']:+.1f} MB  "
                f"peak GPU={pose_peak_gpu:.1f} MB"
            )

            perf_rows.append(
                {
                    "sample_size": n_sample,
                    "n_top_grasps": n_top_grasps,
                    "impl": "grasp_pose_generator",
                    "component": "grasp_pose_select",
                    "cost_time_ms": f"{pose_ms:.6f}",
                    "cpu_delta_mb": f"{pose_mem['cpu_mb']:.6f}",
                    "gpu_delta_mb": f"{pose_mem['gpu_mb']:.6f}",
                    "peak_gpu_mb": f"{pose_peak_gpu:.6f}",
                }
            )
            metric_rows.append(
                {
                    "sample_size": n_sample,
                    "n_top_grasps": n_top_grasps,
                    "impl": "grasp_pose_generator",
                    "component": "grasp_pose_select",
                    "success_rate": f"{float(is_success):.6f}",
                    "hit_pairs": hit_pairs,
                    "selected_grasps": selected_grasps,
                    "best_cost": f"{best_cost:.6f}" if is_success else "inf",
                }
            )
            pipeline_rows.append(
                {
                    "sample_size": n_sample,
                    "n_top_grasps": n_top_grasps,
                    "success_rate": float(is_success),
                    "total_time_ms": total_ms,
                    "hit_pairs": hit_pairs,
                    "selected_grasps": selected_grasps,
                    "best_cost": f"{best_cost:.6f}" if is_success else "inf",
                }
            )

    if skip_pose_selection:
        for n_sample in n_samples:
            pipeline_rows.append(
                {
                    "sample_size": n_sample,
                    "n_top_grasps": "skipped",
                    "success_rate": 0.0,
                    "total_time_ms": float("inf"),
                    "hit_pairs": "N/A",
                    "selected_grasps": "N/A",
                    "best_cost": "N/A",
                }
            )

    return perf_rows, metric_rows, pipeline_rows, notes


def run_all_benchmarks(
    n_samples: list[int] | None = None,
    n_top_grasps_list: list[int] | None = None,
    device_name: str = "auto",
    seed: int = 0,
    skip_pose_selection: bool = False,
) -> None:
    """Run grasp pose generator benchmarks and write a markdown report."""
    device = _resolve_device(device_name)
    n_samples = n_samples or DEFAULT_N_SAMPLES
    n_top_grasps_list = n_top_grasps_list or DEFAULT_N_TOP_GRASPS

    print("=" * 60)
    print("Grasp Pose Generator Performance Benchmarks")
    print("=" * 60)

    perf_rows, metric_rows, pipeline_rows, notes = benchmark_grasp_pose_generator(
        n_samples=n_samples,
        n_top_grasps_list=n_top_grasps_list,
        device=device,
        seed=seed,
        skip_pose_selection=skip_pose_selection,
    )
    leaderboard_rows = _build_leaderboard_rows(pipeline_rows)

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    print("=" * 60)

    report_path = _write_markdown_report(
        benchmark_name="grasp_pose_generator",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            "CPU/GPU memory fields are deltas measured around timed calls.",
            "This report contains exactly three tables: Time & Memory, Success & Other Metrics, and Leaderboard.",
        ]
        + notes,
    )
    print(f"Markdown report saved: {report_path}")


if __name__ == "__main__":
    args = _parse_args()
    run_all_benchmarks(
        n_samples=args.n_samples,
        n_top_grasps_list=args.n_top_grasps,
        device_name=args.device,
        seed=args.seed,
        skip_pose_selection=args.skip_pose_selection,
    )
