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

"""Measure batched ray intersection latency on bounded scene workloads.

Run: python -m scripts.benchmark.sensors.raycast --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
import warp as wp

from embodichain.compute.geometry.raycast import TriangleRayMesh, cast_rays


def run_case(
    device: torch.device, envs: int, count: int, repeats: int
) -> tuple[float, float]:
    """Time a warmed callable and check its distances against a known plane."""
    vertices = torch.tensor(
        [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]],
        device=device,
    )
    mesh = TriangleRayMesh(
        vertices, torch.tensor([[0, 1, 2], [0, 2, 3]], device=device)
    )
    pose = torch.eye(4, device=device)[None]
    origins = torch.zeros((envs, count, 3), device=device)
    origins[..., 2] = 2
    directions = torch.zeros_like(origins)
    directions[..., 2] = -1
    args = (origins, directions, [(mesh, pose)], 10.0)
    expected = torch.full((envs, count), 2.0, device=device)
    for _ in range(5):
        cast_rays(*args)
    sync = (
        (lambda: torch.cuda.synchronize(device))
        if device.type == "cuda"
        else (lambda: None)
    )
    sync()
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        result, indices = cast_rays(*args)
        sync()
        timings.append((perf_counter() - start) * 1000.0)
    error = float((result - expected).abs().max())
    if error > 1e-6 or not bool((indices == 0).all()):
        raise RuntimeError("Ray benchmark disagrees with the analytic plane.")
    return median(timings), error


def main() -> None:
    """Benchmark single and batched rays on the requested device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; GPU results skipped.")
        return
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    rows = []
    for envs, rays in ((1, 64), (512, 256), (2048, 64)):
        ms, error = run_case(device, envs, rays, args.repeats)
        rows.append((envs, rays, ms, error))
    report = [
        "# Batched ray query benchmark",
        "",
        f"{hardware}, Torch {torch.__version__}, Warp {wp.__version__}, {args.repeats} repeats.",
        "Equivalent analytic plane per row. Timings include allocations, launch and device completion.",
        "BVH construction and simulation are outside timing. Allocator peak memory was not profiled.",
        "",
        "## Time & Memory",
        "",
        "| Environments | Rays per row | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---:|---:|---:|---:|---:|---:|",
        *[f"| {b} | {r} | {ms:.3f} | — | — | — |" for b, r, ms, _ in rows],
        "",
        "## Success & Other Metrics",
        "",
        "| Environments | Rays per row | success_rate | Max distance error (m) |",
        "|---:|---:|---:|---:|",
        *[f"| {b} | {r} | 100% | {error:.3g} |" for b, r, _, error in rows],
        "",
        "## Leaderboard",
        "",
        "| Implementation | Success rate |",
        "|---|---:|",
        "| Warp triangle BVH | 100% |",
    ]
    output = "\n".join(report)
    print(output)
    if args.output is not None:
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
