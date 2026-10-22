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

"""Compare anchored NumPy, Torch and Warp batch resampling with accuracy checks.

Run: python -m scripts.benchmark.trajectory.anchored_batch --batches 1 13 64 128
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import platform
import time
from collections.abc import Callable

import numpy as np
import psutil
import torch
import warp as wp

from embodichain.compute.trajectory import (
    prepare_anchored_batch,
    resample_with_anchors,
    resample_with_anchors_batched,
)


def _fixture(batch: int, samples: int, dtype: torch.dtype) -> tuple:
    """Generate seeded ragged paths with independent event counts and padding."""
    rng = np.random.default_rng(192)
    lengths = rng.integers(max(8, samples // 2), samples + 1, size=batch)
    q = torch.full((batch, samples, 13), float("nan"), dtype=dtype)
    times = torch.full((batch, samples), float("nan"), dtype=torch.float64)
    anchors = torch.full((batch, 6), -1, dtype=torch.long)
    counts = torch.tensor(rng.integers(0, 7, size=batch))
    for b, n in enumerate(lengths):
        clock = np.r_[0.0, np.cumsum(rng.uniform(0.05, 0.09, n - 1))]
        times[b, :n] = torch.tensor(clock)
        q[b, :n] = torch.tensor(
            np.sin(clock[:, None] * np.linspace(0.1, 1.0, 13)[None]) + b * 0.01,
            dtype=dtype,
        )
        anchors[b, : counts[b]] = torch.tensor(
            rng.choice(n, int(counts[b]), replace=False)
        )
    return (
        q,
        times,
        anchors,
        dict(source_lengths=torch.tensor(lengths), anchor_counts=counts),
    )


def _numpy_batch(
    q: torch.Tensor, times: torch.Tensor, anchors: torch.Tensor, metadata: dict
) -> np.ndarray:
    """Evaluate individual references and apply identical terminal padding."""
    rows = []
    for b, n in enumerate(metadata["source_lengths"].tolist()):
        k = int(metadata["anchor_counts"][b])
        row, _ = resample_with_anchors(
            q[b, :n].numpy(), times[b, :n].numpy(), 0.01, anchors[b, :k].tolist()
        )
        rows.append(row)
    out = np.empty(
        (len(rows), max(len(r) for r in rows), q.shape[-1]), dtype=rows[0].dtype
    )
    for b, row in enumerate(rows):
        out[b] = row[-1]
        out[b, : len(row)] = row
    return out


def _measure(fn: Callable, device: str, repeats: int) -> tuple[object, dict]:
    """Measure synchronized warmed wall time, process RSS and Torch allocations."""
    fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    rss = process.memory_info().rss
    gpu = torch.cuda.memory_allocated() if device.startswith("cuda") else 0
    durations = []
    result = None
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    current_gpu = torch.cuda.memory_allocated() if device.startswith("cuda") else 0
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
    return result, dict(
        cost_time_ms=float(np.median(durations)),
        cpu_delta_mb=(process.memory_info().rss - rss) / 1024**2,
        gpu_delta_mb=(current_gpu - gpu) / 1024**2,
        peak_gpu_mb=peak / 1024**2,
    )


def _table(rows: list[dict]) -> list[str]:
    """Format one rectangular Markdown table."""
    keys = list(rows[0])
    return [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
        *(
            "| "
            + " | ".join(
                f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys
            )
            + " |"
            for r in rows
        ),
    ]


def run_all_benchmarks() -> None:
    """Run backend/device comparisons and write exactly one Markdown report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 13, 64, 128])
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.samples < 8
        or args.repeats < 1
        or args.threads < 1
        or any(b < 1 for b in args.batches)
    ):
        parser.error("Need samples >=8 and positive batches, repeats and threads")
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    perf = []
    metrics = []
    methods = [
        ("numpy_single", "cpu", None),
        ("torch_cpu", "cpu", "torch"),
        ("warp_cpu", "cpu", "warp"),
    ]
    if torch.cuda.is_available():
        methods.extend(
            [("torch_cuda", "cuda:0", "torch"), ("warp_cuda", "cuda:0", "warp")]
        )
    else:
        print("CUDA unavailable: skipping GPU comparisons")
    for batch in args.batches:
        q, times, anchors, metadata = _fixture(batch, args.samples, dtype)
        expected = _numpy_batch(q, times, anchors, metadata)
        for name, device, backend in methods:
            if backend is None:
                apply = lambda: _numpy_batch(q, times, anchors, metadata)
                prepare_ms = 0.0
                plan = None
                end_to_end = apply
            else:
                device_q = q.to(device)
                plan, preparation = _measure(
                    lambda: prepare_anchored_batch(
                        times, 0.01, anchors, device=device, **metadata
                    ),
                    device,
                    args.repeats,
                )
                prepare_ms = preparation["cost_time_ms"]
                apply = lambda: plan.apply(device_q, backend=backend)

                # Deliberately include input/output transfer and all mapping
                # preparation here to model the offline normalization caller.
                def end_to_end() -> torch.Tensor:
                    """Include CPU input/output transfer and clock preparation."""
                    out, _ = resample_with_anchors_batched(
                        q.to(device), times, 0.01, anchors, backend=backend, **metadata
                    )
                    return out.cpu()

            out, cost = _measure(apply, device, args.repeats)
            _, total = _measure(end_to_end, device, args.repeats)
            actual = (
                out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else out
            )
            error = float(np.max(np.abs(actual - expected)))
            tolerance = 2e-7 if dtype == torch.float32 else 2e-14
            passed = np.all(
                np.isclose(actual, expected, atol=tolerance, rtol=tolerance),
                axis=(1, 2),
            )
            exact = True
            if plan is not None:
                source = q.numpy()
                bit_type = np.int32 if dtype == torch.float32 else np.int64
                for b, row in enumerate(plan.rows):
                    exact &= np.array_equal(
                        actual[b, row.anchor_execution_indices].view(bit_type),
                        source[b, row.anchor_source_indices].view(bit_type),
                    )
            if not passed.all() or not exact:
                raise AssertionError(
                    f"{name}: batch {batch} failed accuracy/anchor check"
                )
            perf.append(
                dict(
                    batch=batch,
                    impl=name,
                    **cost,
                    prepare_ms=prepare_ms,
                    end_to_end_ms=total["cost_time_ms"],
                )
            )
            metrics.append(
                dict(
                    batch=batch,
                    impl=name,
                    success_rate=float(passed.mean()),
                    max_abs_error=error,
                    anchors_exact=exact,
                )
            )
            print(
                f"B={batch:4d} {name:13s} apply={cost['cost_time_ms']:9.3f} ms total={total['cost_time_ms']:9.3f} ms max_error={error:.3g}"
            )
    leaderboard = []
    for name, _, _ in methods:
        rows = [r for r in metrics if r["impl"] == name]
        costs = [r["cost_time_ms"] for r in perf if r["impl"] == name]
        leaderboard.append(
            dict(
                algorithm=name,
                overall_success_rate=float(np.mean([r["success_rate"] for r in rows])),
                mean_apply_ms=float(np.mean(costs)),
            )
        )
    leaderboard.sort(key=lambda r: (-r["overall_success_rate"], r["mean_apply_ms"]))
    leaderboard = [dict(rank=i + 1, **row) for i, row in enumerate(leaderboard)]
    report = (
        args.output
        or Path("outputs/benchmarks")
        / f"anchored_batch_{datetime.now():%Y%m%d_%H%M%S}.md"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    )
    lines = [
        "# Anchored batch resampling benchmark",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Platform: {platform.platform()}; CPU: {platform.processor()}; GPU: {gpu_name}",
        f"Torch {torch.__version__}; Warp {wp.__version__}; dtype {args.dtype}; CPU threads {args.threads}; repeats {args.repeats}",
        f"Batches {args.batches}; N_max {args.samples}; D=13; control_dt=0.01 s; seed=192.",
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
        "- Success denotes numerical agreement with individual NumPy references, not physical task success.",
        "- apply includes input validation; prepared batch plans and device inputs are reused. NumPy single includes preparation each call.",
        "- end_to_end includes mapping preparation, CPU input transfer and CPU output transfer; warm compilation is excluded everywhere.",
        "- prepare_ms is recorded separately; zero for NumPy means no separately prepared implementation.",
        "- CPU memory is process RSS delta; GPU fields are Torch allocator delta/absolute peak around apply and exclude driver/context/JIT memory.",
        "- Shorter paths use terminal padding; exact anchors, valid lengths and row independence are also covered by the production tests.",
        "",
    ]
    with report.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Markdown report saved: {report}")


if __name__ == "__main__":
    run_all_benchmarks()
