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

"""Compare fluid render clipping and optional native surface publication.

Run: python -m scripts.benchmark.scene.fluid_render_sync --device cuda:0 --native
Inputs are device resident. Physics stepping is excluded. The optional native
cases measure bridge publication and a complete camera frame separately.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace

import numpy as np
import psutil
import torch
import warp as wp

from embodichain.lab.sim.physics.newton import NewtonPhysicsBackend


def _legacy_clip(
    positions: torch.Tensor, ranges: tuple[tuple[int, int, float], ...]
) -> torch.Tensor:
    """Reproduce the original clone and per-fluid boolean assignment."""
    output = positions.clone()
    for start, end, height in ranges:
        selected = output[start:end]
        selected[selected[:, 2] < height] = float("nan")
    return output


def _synchronize(device: torch.device) -> None:
    """Complete all participating CUDA streams before recording wall time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    operation: Callable[[], object], device: torch.device, args: argparse.Namespace
) -> dict[str, object]:
    """Measure steady-state completed calls and a separate retained output."""
    cuda = device.type == "cuda"
    _synchronize(device)
    initial = torch.cuda.memory_allocated(device) if cuda else 0
    start = time.perf_counter()
    result = operation()
    _synchronize(device)
    first_call_ms = (time.perf_counter() - start) * 1000
    del result
    for _ in range(args.warmup):
        result = operation()
        _synchronize(device)
        del result
    persistent = torch.cuda.memory_allocated(device) - initial if cuda else None
    samples = []
    for _ in range(args.repeats):
        _synchronize(device)
        start = time.perf_counter()
        result = operation()
        _synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
        del result

    process = psutil.Process()
    cpu_before = process.memory_info().rss
    gpu_before = torch.cuda.memory_allocated(device) if cuda else 0
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    result = operation()
    _synchronize(device)
    mib = 1024**2
    memory = {
        "cpu_delta_mb": (process.memory_info().rss - cpu_before) / mib,
        "gpu_delta_mb": (
            (torch.cuda.memory_allocated(device) - gpu_before) / mib if cuda else None
        ),
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / mib if cuda else None,
        "extra_peak_gpu_mb": (
            (torch.cuda.max_memory_allocated(device) - gpu_before) / mib
            if cuda
            else None
        ),
        "warmup_gpu_delta_mb": persistent / mib if persistent is not None else None,
    }
    del result
    return {
        "cost_time_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "first_call_ms": first_call_ms,
        **memory,
    }


def _paired_latencies(
    operations: dict[str, Callable[[], object]],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    """Alternate equivalent calls to reduce GPU clock and thermal-order bias."""
    samples: dict[str, list[float]] = {name: [] for name in operations}
    order = list(operations)
    for index in range(args.warmup + args.repeats):
        for name in order if index % 2 == 0 else reversed(order):
            _synchronize(device)
            start = time.perf_counter()
            result = operations[name]()
            _synchronize(device)
            if index >= args.warmup:
                samples[name].append((time.perf_counter() - start) * 1000)
            del result
    return {
        name: {
            "cost_time_ms": statistics.median(values),
            "p95_ms": float(np.percentile(values, 95)),
        }
        for name, values in samples.items()
    }


def _benchmark_clipping(
    count: int, fluids: int, device: torch.device, args: argparse.Namespace
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compare equivalent clipping masks, including partial particle ranges."""
    generator = torch.Generator(device=device).manual_seed(42)
    positions = torch.randn((count, 3), device=device, generator=generator)
    before = positions.clone()
    # Multiple ranges leave gaps to exercise particles outside the clipped fluids.
    ranges = tuple(
        (
            count * i // fluids,
            count * (i + 1) // fluids - (1 if fluids > 1 else 0),
            0.1 * i,
        )
        for i in range(fluids)
    )
    backend = NewtonPhysicsBackend(SimpleNamespace())
    operations = {
        "legacy": lambda: _legacy_clip(positions, ranges),
        "vectorized": lambda: backend._clip_fluid_render_positions(positions, ranges),
    }
    expected = positions.clone()
    heights = torch.full((count,), float("-inf"), device=device)
    for start, end, height in ranges:
        heights[start:end] = height
    expected[positions[:, 2] < heights] = float("nan")
    perf_rows, metric_rows = [], []
    for algorithm, operation in operations.items():
        # Warmup is part of _measure; correctness checks must not prefill its cache.
        perf = _measure(operation, device, args)
        cutoff = backend._fluid_render_clip_z if algorithm == "vectorized" else None
        perf["cutoff_cache_mb"] = (
            cutoff.numel() * cutoff.element_size() / 1024**2
            if cutoff is not None
            else 0.0
        )
        actual = operation()
        torch.testing.assert_close(actual, expected, equal_nan=True, rtol=0, atol=0)
        torch.testing.assert_close(positions, before, rtol=0, atol=0)
        identity = {
            "scope": "clipping",
            "particles": count,
            "fluids": fluids,
            "algorithm": algorithm,
        }
        perf_rows.append({**identity, **perf})
        metric_rows.append({**identity, "success_rate": 1.0, "source_unchanged": True})
        del actual
    paired = _paired_latencies(operations, device, args)
    for row in perf_rows:
        row.update(paired[str(row["algorithm"])])
    return perf_rows, metric_rows


def _benchmark_native(
    count: int, device: torch.device, args: argparse.Namespace
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Measure full native surface publication with identical quality settings."""
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        FluidObjectCfg,
        FluidRenderCfg,
        FluidSpawnCfg,
        NewtonPhysicsCfg,
        RenderCfg,
    )
    from embodichain.lab.sim.sensors import CameraCfg

    side = int(np.ceil(count ** (1 / 3)))
    points = np.column_stack(np.unravel_index(np.arange(count), (side,) * 3))
    points = np.asarray((points - side / 2) * 0.003, dtype=np.float32)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device=str(device),
            num_envs=1,
            render_cfg=RenderCfg(renderer="fast-rt"),
            physics_cfg=NewtonPhysicsCfg(
                num_substeps=1,
                collision_cfg=None,
                solver_cfg={"solver_type": "sph", "volume_map_boundary_enabled": False},
            ),
        )
    )
    fluid = camera = None
    try:
        fluid = sim.add_fluid_object(
            FluidObjectCfg(
                uid="water",
                particle_spacing=0.003,
                spawn=FluidSpawnCfg(positions=points),
                render=FluidRenderCfg(
                    mode="mesh_surface",
                    clip_below_z=0.0,
                    mesh_surface={"gpu_enabled": True, "max_res": 128},
                ),
            )
        )
        camera = sim.add_sensor(
            CameraCfg(
                uid="benchmark_camera",
                width=320,
                height=240,
                intrinsics=(350, 350, 160, 120),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=(0.18, -0.24, 0.28), target=(0, 0, 0), up=(0, 0, 1)
                ),
            )
        )
        sim.prepare()
        native = get_newton_backend(sim.spawn_result.world)
        positions = wp.to_torch(native.state_0.particle_q)
        before = positions.clone()

        def legacy() -> None:
            # Include the manager/backend lookup and range traversal performed
            # by the previous public sync_render_state implementation.
            result = sim.spawn_result
            current = get_newton_backend(result.world)
            clipped_fluids = [
                item
                for item in sim._fluid_objects.values()
                if item.cfg.render.clip_below_z is not None
            ]
            clipped = wp.to_torch(current.state_0.particle_q).clone()
            for item in clipped_fluids:
                for entity in item._entities:
                    start = entity._physics_binding.particle_start
                    selected = clipped[start : start + entity.particle_count]
                    selected[selected[:, 2] < item.cfg.render.clip_below_z] = float(
                        "nan"
                    )
            current.render_sync.sync(
                result.world,
                current.state_0.body_q,
                wp.from_torch(clipped, dtype=wp.vec3),
                simulation_time=current.sim_time,
            )
            if sim._render_state_published is not None:
                sim._render_state_published = True

        perf_rows, metric_rows = [], []
        for scope in ("native bridge publication", "native camera frame"):
            operations = {}
            for algorithm, publish in {
                "legacy": legacy,
                "vectorized": sim.sync_render_state,
            }.items():

                def operation(publish: Callable[[], None] = publish) -> object:
                    publish()
                    if scope == "native camera frame":
                        sim._world.render_camera_group([camera.group_id])
                        camera.update(fetch_only=True)
                        return camera.get_data()["color"]
                    return None

                operations[algorithm] = operation
                perf = _measure(operation, device, args)
                cutoff = (
                    sim.physics._fluid_render_clip_z
                    if algorithm == "vectorized"
                    else None
                )
                perf["cutoff_cache_mb"] = (
                    cutoff.numel() * cutoff.element_size() / 1024**2
                    if cutoff is not None
                    else 0.0
                )
                torch.testing.assert_close(positions, before, rtol=0, atol=0)
                identity = {
                    "scope": scope,
                    "particles": count,
                    "fluids": 1,
                    "algorithm": algorithm,
                }
                perf_rows.append({**identity, **perf})
                metric_rows.append(
                    {**identity, "success_rate": 1.0, "source_unchanged": True}
                )
            paired = _paired_latencies(operations, device, args)
            for row in perf_rows:
                if row["scope"] == scope:
                    row.update(paired[str(row["algorithm"])])
        return perf_rows, metric_rows
    finally:
        fluid = camera = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _write_report(
    path: Path,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    notes: list[str],
) -> None:
    """Write exactly three tables, including every evaluated algorithm."""
    leaderboard = []
    for algorithm in sorted({str(row["algorithm"]) for row in metric_rows}):
        results = [row for row in metric_rows if row["algorithm"] == algorithm]
        leaderboard.append(
            {
                "algorithm": algorithm,
                "overall_success_rate": statistics.mean(
                    float(row["success_rate"]) for row in results
                ),
                "cases": len(results),
            }
        )
    leaderboard.sort(key=lambda row: -float(row["overall_success_rate"]))
    leaderboard = [{"rank": index, **row} for index, row in enumerate(leaderboard, 1)]
    lines = ["# Fluid render synchronization benchmark", "", *notes, ""]
    for title, rows in (
        ("Time & Memory", perf_rows),
        ("Success & Other Metrics", metric_rows),
        ("Leaderboard", leaderboard),
    ):
        lines.extend([f"## {title}", ""])
        if rows:
            headers = list(rows[0])
            lines.extend(
                [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |",
                ]
            )
            for row in rows:
                values = [
                    f"{row[key]:.6f}" if isinstance(row[key], float) else str(row[key])
                    for key in headers
                ]
                lines.append("| " + " | ".join(values) + " |")
        else:
            lines.extend(["| status |", "| --- |", "| skipped |"])
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_all_benchmarks(args: argparse.Namespace) -> None:
    """Run clipping/native scopes and save one reproducible Markdown report."""
    device = torch.device(args.device)
    notes = [
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- CPU: {platform.processor()}; Torch {torch.__version__}; Warp {wp.__version__}",
        f"- Device: {device}; dtype=float32; seed=42; CPU threads={torch.get_num_threads()}",
        f"- Warmup={args.warmup}; repeats={args.repeats}; median/p95 completed-call wall time.",
        "- Latency uses paired alternating implementation order. Memory/cache statistics use a separate warmed pass.",
        "- First-call timing precedes warmup and includes cutoff-cache setup; process/World startup and input generation are excluded.",
        "- Device-resident inputs; device-wide CUDA synchronization; no physics steps. Native camera scope includes 320x240 FastRT rendering and image fetch.",
        "- CPU memory is retained RSS delta. GPU memory covers the Torch allocator only, excluding native/Warp allocations.",
        "- Incremental peak is measured separately after warmup. Cutoff cache reports logical buffer bytes; warmup GPU delta may include unrelated lazy allocation/release.",
        "- Clipping quality checks exact positions/NaN masks and source preservation. Native success checks completed publication and unchanged physics; camera pixels are covered by test_fluid_visibility.py.",
        "- Native surface settings: FastRT, spp=1, mesh_surface GPU enabled, max_res=128; other RenderCfg/FluidMeshSurfaceRenderDesc defaults.",
    ]
    try:
        notes.append(f"- DexSim distribution: {version('dexsim_engine')}")
    except PackageNotFoundError:
        notes.append("- DexSim distribution metadata unavailable.")
    notes.extend(f"- {note}" for note in args.note)
    perf_rows, metric_rows = [], []
    if device.type == "cuda" and not torch.cuda.is_available():
        notes.append("- Skipped: CUDA is unavailable.")
    else:
        if device.type == "cuda":
            notes.append(f"- GPU: {torch.cuda.get_device_name(device)}")
        with torch.no_grad():
            for count in args.sizes:
                for fluids in args.fluid_counts:
                    perf, metrics = _benchmark_clipping(count, fluids, device, args)
                    perf_rows.extend(perf)
                    metric_rows.extend(metrics)
            if args.native and device.type == "cuda":
                for count in args.native_sizes:
                    perf, metrics = _benchmark_native(count, device, args)
                    perf_rows.extend(perf)
                    metric_rows.extend(metrics)
            elif args.native:
                notes.append("- Native publication skipped: CUDA is required.")
    _write_report(args.output, perf_rows, metric_rows, notes)
    for row in perf_rows:
        print(
            f"{row['scope']}: n={row['particles']} fluids={row['fluids']} "
            f"{row['algorithm']} {row['cost_time_ms']:.6f} ms"
        )
    print(f"Markdown report saved: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[1728, 76607, 1_000_000]
    )
    parser.add_argument("--fluid-counts", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--native-sizes", nargs="+", type=int, default=[1728, 76607])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument(
        "--note", action="append", default=[], help="Additional measurement conditions"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmarks")
        / f"fluid_render_sync_{datetime.now():%Y%m%d_%H%M%S}.md",
    )
    args = parser.parse_args()
    if (
        args.warmup < 1
        or args.repeats < 1
        or min(args.sizes + args.native_sizes + args.fluid_counts) < 1
    ):
        parser.error("sizes, fluid counts, warmup and repeats must be positive")
    if min(args.sizes) < max(args.fluid_counts):
        parser.error("particle counts must be at least the number of fluid ranges")
    run_all_benchmarks(args)
