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

"""Measure batched geometry kernels on CPU and CUDA, including transfer cost.

Run: python -m scripts.benchmark.scene.geometry_kernels --device cuda:0
Exactly one Markdown report records time/memory, accuracy and all measured
implementations. This measures geometry, not the CPU scene layout solver.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
import gc
import os
from pathlib import Path
import platform
import statistics
import time
from zoneinfo import ZoneInfo

import numpy as np
import psutil
import torch

from embodichain.compute.geometry import (
    aabb_containment_violation,
    aabb_overlap_depth,
    group_height_offset,
    transform_aabbs,
    triangle_geometry,
)


@torch.no_grad()
def _workload(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    bounds, transforms, support, mask, pairs, triangles, region = inputs
    transformed = transform_aabbs(bounds, transforms)
    overlap = aabb_overlap_depth(transformed[:, pairs[0]], transformed[:, pairs[1]])
    containment = aabb_containment_violation(transformed, region)
    height, valid = group_height_offset(
        transformed, support, valid_mask=mask, clearance=0.02
    )
    normals, areas, triangle_valid = triangle_geometry(triangles)
    return (
        transformed,
        overlap,
        containment,
        height,
        valid,
        normals,
        areas,
        triangle_valid,
    )


def _reference(inputs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """Independent NumPy reference; transforms all eight corners explicitly."""
    bounds, transforms, support, mask, pairs, triangles, region = inputs
    selectors = np.asarray(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=bounds.dtype
    )
    low, high = bounds[..., 0, :], bounds[..., 1, :]
    corners = low[..., None, :] + selectors * (high - low)[..., None, :]
    corners = (
        np.einsum("bnij,bnkj->bnki", transforms[..., :3, :3], corners)
        + transforms[..., :3, 3][..., None, :]
    )
    transformed = np.stack([corners.min(axis=-2), corners.max(axis=-2)], axis=-2)
    a, b = transformed[:, pairs[0]], transformed[:, pairs[1]]
    overlap = np.maximum(
        0,
        np.minimum(a[..., 1, :], b[..., 1, :]) - np.maximum(a[..., 0, :], b[..., 0, :]),
    )
    containment = np.maximum(
        0,
        np.maximum(
            region[0] - transformed[..., 0, :], transformed[..., 1, :] - region[1]
        ),
    )
    valid = mask.any(axis=-1)
    minimum = np.min(np.where(mask, transformed[..., 0, 2], np.inf), axis=-1)
    height = np.where(valid, support + 0.02 - minimum, 0)
    cross = np.cross(
        triangles[..., 1, :] - triangles[..., 0, :],
        triangles[..., 2, :] - triangles[..., 0, :],
    )
    length = np.linalg.norm(cross, axis=-1)
    triangle_valid = length > 0
    normals = np.divide(
        cross,
        length[..., None],
        out=np.zeros_like(cross),
        where=triangle_valid[..., None],
    )
    areas = length * 0.5
    return (
        transformed,
        overlap,
        containment,
        height,
        valid,
        normals,
        areas,
        triangle_valid,
    )


def _inputs(batch: int, objects: int) -> tuple[torch.Tensor, ...]:
    """Prepare reproducible nonuniform boxes and non-symmetric rigid transforms."""
    rng = np.random.default_rng(42)
    center = rng.uniform(-0.8, 0.8, size=(batch, objects, 3)).astype(np.float32)
    half = rng.uniform(0.02, 0.15, size=(batch, objects, 3)).astype(np.float32)
    bounds = np.stack([center - half, center + half], axis=-2)
    # QR yields orthogonal frames; scale independently on each local axis.
    rotation, _ = np.linalg.qr(rng.normal(size=(batch, objects, 3, 3)))
    rotation *= rng.uniform(0.6, 1.5, size=(batch, objects, 1, 3))
    transforms = (
        np.broadcast_to(np.eye(4), (batch, objects, 4, 4)).copy().astype(np.float32)
    )
    transforms[..., :3, :3] = rotation
    transforms[..., :3, 3] = rng.uniform(-0.2, 0.2, size=(batch, objects, 3))
    support = np.full(batch, 0.75, dtype=np.float32)
    mask = rng.random((batch, objects)) > 0.15
    mask[::97] = False
    pairs = np.asarray(np.triu_indices(objects, k=1), dtype=np.int64)
    triangles = rng.normal(size=(batch, objects, 3, 3)).astype(np.float32)
    triangles[:, 0, 2] = triangles[:, 0, 1]
    region = np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    return tuple(
        torch.from_numpy(x)
        for x in (bounds, transforms, support, mask, pairs, triangles, region)
    )


def _measure(
    fn: Callable[[], tuple], *, iterations: int, device: torch.device | None
) -> tuple[tuple, dict[str, float]]:
    """Measure synchronized steady-state latency, RSS delta and allocated VRAM."""
    for _ in range(5):
        result = fn()
    del result
    if device is not None:
        torch.cuda.synchronize(device)
    gc.collect()
    process = psutil.Process(os.getpid())
    cpu_before = process.memory_info().rss
    gpu_before = torch.cuda.memory_allocated(device) if device is not None else 0
    if device is not None:
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        if device is not None:
            torch.cuda.synchronize(device)
        elapsed.append((time.perf_counter() - start) * 1000)
        del result
    # One representative output is retained for correctness and memory accounting.
    result = fn()
    if device is not None:
        torch.cuda.synchronize(device)
    mib = 1024**2
    return result, {
        "cost_time_ms": statistics.median(elapsed),
        "cpu_delta_mb": (process.memory_info().rss - cpu_before) / mib,
        "gpu_delta_mb": (
            (torch.cuda.memory_allocated(device) - gpu_before) / mib
            if device is not None
            else 0.0
        ),
        "peak_gpu_mb": (
            torch.cuda.max_memory_allocated(device) / mib if device is not None else 0.0
        ),
    }


def _accuracy(result: tuple, expected: tuple[np.ndarray, ...]) -> tuple[float, float]:
    passed = True
    error = 0.0
    for value, reference in zip(result, expected, strict=True):
        actual = (
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        )
        if actual.dtype == np.bool_:
            passed &= np.array_equal(actual, reference)
        else:
            error = max(error, float(np.max(np.abs(actual - reference), initial=0)))
            passed &= np.allclose(actual, reference, atol=5e-6, rtol=1e-5)
    return float(passed), error


def _table(rows: list[dict[str, object]]) -> list[str]:
    headers = list(rows[0])
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *(
            "| "
            + " | ".join(
                f"{row[key]:.6f}" if isinstance(row[key], float) else str(row[key])
                for key in headers
            )
            + " |"
            for row in rows
        ),
    ]


def run_all_benchmarks(
    *,
    device: str,
    iterations: int,
    output: Path,
    batches: tuple[int, ...] = (1, 64, 1024),
    objects: int = 32,
    threads: int = 4,
) -> None:
    """Compare NumPy, Torch CPU, CUDA resident and CUDA round-trip workloads.

    Args:
        device: Explicit CUDA device or ``cpu`` to skip CUDA.
        iterations: Timed iterations after warmup, reported as median latency.
        output: Destination of the single Markdown report.
        batches: Independent scene/candidate counts.
        objects: Boxes/triangles per independent scene.
        threads: CPU Torch thread count used consistently across all rows.
    """
    if (
        iterations < 1
        or objects < 2
        or threads < 1
        or not batches
        or any(batch < 1 for batch in batches)
    ):
        raise ValueError(
            "Positive iterations/batches/threads and objects >= 2 required"
        )
    requested = torch.device(device)
    cuda = requested if requested.type == "cuda" and torch.cuda.is_available() else None
    if requested.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if requested.type == "cuda" and cuda is None:
        print("CUDA unavailable; reporting CPU implementations only.")
    original_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    perf = []
    metrics = []
    try:
        for batch in batches:
            inputs = _inputs(batch, objects)
            numpy_inputs = tuple(x.numpy() for x in inputs)
            reference = _reference(
                tuple(
                    x.astype(np.float64) if np.issubdtype(x.dtype, np.floating) else x
                    for x in numpy_inputs
                )
            )
            implementations = [
                ("numpy_cpu", lambda: _reference(numpy_inputs), None),
                ("torch_cpu", lambda: _workload(inputs), None),
            ]
            if cuda is not None:
                resident = tuple(x.to(cuda) for x in inputs)

                def roundtrip() -> tuple[torch.Tensor, ...]:
                    transferred = tuple(x.to(cuda) for x in inputs)
                    return tuple(x.cpu() for x in _workload(transferred))

                implementations.extend(
                    [
                        ("torch_cuda_resident", lambda: _workload(resident), cuda),
                        ("torch_cuda_roundtrip", roundtrip, cuda),
                    ]
                )
            for name, fn, timed_device in implementations:
                result, timing = _measure(
                    fn, iterations=iterations, device=timed_device
                )
                success, error = _accuracy(result, reference)
                perf.append(
                    {"batch": batch, "objects": objects, "impl": name, **timing}
                )
                metrics.append(
                    {
                        "batch": batch,
                        "objects": objects,
                        "impl": name,
                        "success_rate": success,
                        "max_abs_error": error,
                    }
                )
                print(
                    f"B={batch:4d} N={objects:3d} {name:24s} {timing['cost_time_ms']:10.4f} ms | CPU delta {timing['cpu_delta_mb']:+.2f} MB GPU peak {timing['peak_gpu_mb']:.2f} MB | error {error:.3g}"
                )
                del result
    finally:
        torch.set_num_threads(original_threads)
    names = sorted({row["impl"] for row in metrics})
    ranked = sorted(
        [
            (
                name,
                statistics.mean(
                    row["success_rate"] for row in metrics if row["impl"] == name
                ),
                max(row["max_abs_error"] for row in metrics if row["impl"] == name),
            )
            for name in names
        ],
        key=lambda row: (-row[1], row[2], row[0]),
    )
    leaderboard = [
        {
            "rank": i + 1,
            "impl": name,
            "overall_success_rate": success,
            "max_abs_error": error,
        }
        for i, (name, success, error) in enumerate(ranked)
    ]
    lines = [
        "# Scene geometry kernel benchmark",
        "",
        f"Generated: {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}",
        "",
        f"Python {platform.python_version()}, Torch {torch.__version__}, NumPy {np.__version__}; CPU Torch threads={threads}; 5 warmups and {iterations} timed iterations per row.",
        f"CUDA device: {torch.cuda.get_device_name(cuda) if cuda is not None else 'not tested'}.",
        "",
        "## Time & Memory",
        "",
        *_table(perf),
        "",
        "## Success & Other Metrics",
        "",
        *_table(metrics),
        "",
        "## Leaderboard",
        "",
        *_table(leaderboard),
        "",
        "- B counts independent layouts/candidates; N counts objects in each layout.",
        "- Workload includes transformed bounds, all unique pair overlaps, containment, masked group-height reduction and triangle geometry.",
        "- CUDA resident excludes host/device transfers; CUDA roundtrip includes input H2D and all output D2H transfers. Kernel calls are warmed and synchronized.",
        "- NumPy independently transforms all eight corners; Torch uses the mathematically equivalent center/extent calculation. Accuracy uses a float64 reference with atol=5e-6 and rtol=1e-5, and exact Boolean masks.",
        "- Time is median wall latency; memory is process RSS delta and Torch allocated/peak VRAM. Allocator reuse and shared-machine load affect these measurements.",
        "- The scene CPU search/SLSQP solver is not timed or claimed to be GPU accelerated.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report saved: {output.resolve()}")
    if any(row["success_rate"] != 1.0 for row in metrics):
        raise RuntimeError("Numerical comparison failed; see benchmark report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--objects", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 1024])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmarks/scene_geometry_kernels.md"),
    )
    args = parser.parse_args()
    run_all_benchmarks(
        device=args.device,
        iterations=args.iterations,
        output=args.output,
        batches=tuple(args.batches),
        objects=args.objects,
        threads=args.threads,
    )
