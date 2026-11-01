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

"""Measure joint-domain planar transforms, independently of simulation success.

Run: python -m scripts.benchmark.scene.feasibility --device cpu
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np
import torch

from embodichain.compute.geometry.planar import compose_planar_offsets


def benchmark(device: str, *, repeats: int = 20) -> dict:
    """Compare batch transforms with a NumPy float64 matrix oracle.

    Args:
        device: Torch CPU/CUDA device.
        repeats: Completed-call latency repetitions after five warmups.
    Returns:
        Software/hardware metadata, resident latency, accuracy and peak bytes.
    """
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        return {"device": device, "status": "unavailable"}
    rng = np.random.default_rng(7)
    results = []

    def synchronize():
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    for candidates, groups in ((1, 2), (256, 8), (4096, 32)):
        offsets = rng.uniform(-0.2, 0.2, (candidates, groups, 3))
        reference = np.tile(np.eye(4), (groups, 1, 1))
        reference[:, :3, 3] = rng.uniform(-1, 1, (groups, 3))
        expected = np.empty((candidates, groups, 4, 4))
        for i in range(candidates):
            for j in range(groups):
                x, y, yaw = offsets[i, j]
                rotation = np.eye(4)
                rotation[:2, :2] = [
                    [np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw)],
                ]
                expected[i, j] = rotation @ reference[j]
                expected[i, j, :3, 3] = reference[j, :3, 3] + [x, y, 0]
        ref = torch.tensor(reference, dtype=torch.float32, device=target)
        off = torch.tensor(offsets, dtype=torch.float32, device=target)
        for _ in range(5):
            compose_planar_offsets(ref, off)
        synchronize()
        latencies = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            out = compose_planar_offsets(ref, off)
            synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
        error = float(np.max(np.abs(out.cpu().numpy() - expected)))
        if error > 1e-6:
            raise AssertionError(f"Numerical mismatch: {error}")
        peak = None
        if target.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target)
            before = torch.cuda.memory_allocated(target)
            measured = compose_planar_offsets(ref, off)
            synchronize()
            peak = torch.cuda.max_memory_allocated(target) - before
            del measured
        results.append(
            {
                "candidates": candidates,
                "groups": groups,
                "median_ms": statistics.median(latencies),
                "max_abs_error": error,
                "peak_extra_cuda_bytes": peak,
            }
        )
    return {
        "scope": "resident planar transforms; excludes transfers, geometry extraction, planning and physical execution",
        "device": device,
        "hardware": (
            torch.cuda.get_device_name(target)
            if target.type == "cuda"
            else platform.processor()
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dtype": "float32",
        "threads": torch.get_num_threads(),
        "seed": 7,
        "warmups": 5,
        "repeats": repeats,
        "results": results,
    }


def main() -> None:
    """Print the explicitly scoped completed-call benchmark as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(4)
    print(json.dumps(benchmark(args.device, repeats=args.repeats), indent=2))


if __name__ == "__main__":
    main()
