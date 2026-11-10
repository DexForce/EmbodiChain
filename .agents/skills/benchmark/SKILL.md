---
name: benchmark
description: Write benchmark scripts for EmbodiChain modules following project conventions
---

# EmbodiChain Benchmark Script Writer

This skill guides you through writing well-structured benchmark scripts for EmbodiChain modules, covering performance measurement of solvers, samplers, metrics, and other computationally intensive components.

## Usage

Invoke this skill when:

- A user asks to write or extend a benchmark script for any EmbodiChain module
- Comparing CPU vs GPU implementations (e.g., Warp CUDA vs pure-Python)
- Measuring throughput of samplers, metrics, FK/IK solvers, or data pipelines
- The file path contains `scripts/benchmark/` or the word "benchmark" appears in the request

## Key Conventions

### File Location

Place benchmark scripts under:

```
scripts/benchmark/<domain>/<benchmark_name>.py
```

Examples:
- `scripts/benchmark/robotics/kinematic_solver/ur_solver.py`
- `scripts/benchmark/scene/geometry_kernels.py`

### File Header

Every benchmark file **must** begin with the Apache 2.0 copyright header followed by a module-level docstring:

```python
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

"""One-line summary of what this benchmark measures.

Longer description of the optimizations or comparisons being evaluated.
Run: python -m scripts.benchmark.<domain>.<benchmark_name>
"""
```

---

## Steps

### 1. Identify What to Benchmark

Ask yourself:

- **What implementations are being compared?** (e.g., Warp CUDA vs. CPU, vectorized vs. loop-based)
- **What is the primary metric?** Latency, throughput or memory, with a stated correctness/quality threshold.
- **What sizes cover the caller's workload?** Include single-item/small batches when used, typical sizes and a larger supported size within the memory budget.
- **What is timed?** Distinguish device-resident computation, transfer-inclusive calls and end-to-end execution. Measure first-call/setup cost separately when relevant to the caller.

Record hardware, software versions, device, dtype, shapes, seeds, CPU thread
counts, warmup/repeats, synchronization method and measurement scope. For a
performance-sensitive implementation, use
[implement-performance](../implement-performance/SKILL.md) to establish the
workload and baseline before choosing an optimization.

### 2. Structure the Script

Use one helper function per concern, then a single orchestrator:

```
benchmark_<component_a>()   # e.g., benchmark_halton_sampler()
benchmark_<component_b>()   # e.g., benchmark_density_metric()
...
run_all_benchmarks()        # calls all of the above + prints header/footer
```

### 3. Write Individual Benchmark Functions

Use `time.perf_counter()` for wall latency on CPU and GPU. For asynchronous
device work, synchronize the participating backend/device before starting and
before stopping the clock. Torch synchronization is sufficient only when all
participating work is ordered on its streams; otherwise synchronize the owning
backend or establish explicit dependencies. Synchronization belongs in the
benchmark harness, not in the production implementation to make timing easier.

Warm each implementation and relevant shape before steady-state measurement;
exclude compilation, lazy initialization and input generation from that scope.
Repeat measurements and report median latency and repetition count. Add
dispersion or tail latency when relevant; use enough samples for a meaningful
tail estimate. Measure throughput with a separate repeated-work window when
per-call synchronization would disrupt the workload's intended overlap.

This helper measures completed-call wall latency. The caller supplies a no-op
for synchronous CPU work or a synchronization callback for the actual backend:

```python
from collections.abc import Callable
import statistics
import time


def measure_latency(
    operation: Callable[[], object],
    synchronize: Callable[[], None],
    *,
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, object]:
    """Measure repeated completed calls; release outputs outside timing."""
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    for _ in range(warmup):
        result = operation()
        synchronize()
        del result

    samples_ms: list[float] = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        result = operation()
        synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000)
        del result
    return {
        "cost_time_ms": statistics.median(samples_ms),
        "repeats": repeats,
        "samples_ms": samples_ms,
    }
```

For Torch CUDA, pass `lambda: torch.cuda.synchronize(device)` with the explicit
target device; for Warp-owned work use a matching Warp synchronization callback.
CUDA events are an alternative for device elapsed time when all measured work
is ordered between the events; label that scope separately from wall latency.

Keep logging, correctness checks and memory profiling outside latency samples.
Use the caller's intended autograd/inference mode. Reset mutable inputs/state
between samples outside timing, unless resetting is part of the measured API.
For extremely short calls, use repeated-work windows and report the window size
and synchronization policy rather than relying on timer-resolution noise.

### 4. Comparing Two Implementations

Use equivalent seeded inputs, precision, quality thresholds, convergence
budgets and success semantics for baseline and proposed implementations.
Disclose differing settings as a quality/performance tradeoff. Warm each backend
independently and apply the same timing scope; repeat or alternate comparison
order when shared-machine load or thermal drift affects the result.

For device-resident comparison, prepare inputs on each device before timing.
Measure host-to-device and device-to-host copies separately or include them in
an explicitly labeled end-to-end case that consumes the result. Kernel speedup
does not establish application speedup. Report latency/throughput and memory
changes for each relevant size, including regressions and CPU/GPU crossover.

Use the synchronized timing and separate memory passes in
`scripts/benchmark/robotics/kinematic_solver/ur_solver.py` as a Torch/Warp example.
`scripts/benchmark/scene/geometry_kernels.py` illustrates separate resident and
transfer-inclusive cases. Verify measurement scope rather than copying results.

### 5. Report Accuracy Alongside Speed

For FK/IK solvers, always verify correctness by running FK on the IK output and measuring pose error:

Report success rate over all attempted inputs. Evaluate pose errors on valid
outputs, explicitly handle no-success cases, and count nonfinite/invalid outputs
as failures. Use an independent reference where practical. Preserve candidate
ordering, joint limits and failure-mask semantics; include gradient checks when
the API promises differentiability. Run these checks outside the timed region.

```python
def get_pose_err(matrix_a: np.ndarray, matrix_b: np.ndarray) -> tuple[float, float]:
    """Return (translation_error_m, rotation_error_rad)."""
    t_err = np.linalg.norm(matrix_a[:3, 3] - matrix_b[:3, 3])
    relative_rot = matrix_a[:3, :3].T @ matrix_b[:3, :3]
    cos_angle = np.clip((np.trace(relative_rot) - 1) / 2.0, -1.0, 1.0)
    r_err = np.arccos(cos_angle)
    return t_err, r_err


def get_poses_err(
    matrix_a_list: list[np.ndarray], matrix_b_list: list[np.ndarray]
) -> tuple[float, float]:
    t_errs, r_errs = [], []
    for a, b in zip(matrix_a_list, matrix_b_list):
        t, r = get_pose_err(a, b)
        t_errs.append(t)
        r_errs.append(r)
    return float(np.mean(t_errs)), float(np.mean(r_errs))
```

### 6. Handle Benchmarks That Require External Resources

If a benchmark requires a live simulation, robot, or GPU device that may not be available, **skip gracefully** rather than raising an error:

```python
def benchmark_batch_fk():
    """Benchmark batch FK (requires GPU robot setup)."""
    print("\n=== Batch FK Benchmark (requires robot/simulation) ===")
    print("  Skipped -- requires live SimulationManager and Robot.")
    print("  To run manually, integrate with your robot setup:")
    print("    analyzer.compute_workspace_points(joint_configs, batch_size=512)")
```

### Task Program benchmark boundary

For Task Program work, keep these measurements separate:

- strict decode and provider-independent compile;
- configured component composition and catalog preflight;
- live grounding/planning;
- controller execution and physical acceptance.

Validate a configured deployment with `$add-task-program` before benchmarking
it. Do not report static compile/preflight success as an expert-demo success
rate. Physical tables must derive success from completed environment episodes,
effect assurance, and task validators, and must state the selected embodiment,
execution policy, seeds/randomization, and number of completed episodes.

### 7. Write the Orchestrator

```python
def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("=" * 60)
    print("<Module Name> Performance Benchmarks")
    print("=" * 60)

    benchmark_component_a()
    benchmark_component_b()
    # ...

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()
```

### 8. Save Results to One Markdown Report (Required)

Every benchmark script must write its final results to **one Markdown file** after execution.

- Output directory recommendation: `outputs/benchmarks/`
- File naming recommendation: `<benchmark_name>_<YYYYMMDD_HHMMSS>.md`
- Requirement: output **exactly three Markdown tables** in the report
    1. `Time & Memory` table (cost time + memory columns)
    2. `Success & Other Metrics` table (success rate + quality/accuracy/extra metrics)
    3. `Leaderboard` table (algorithm ranking by overall success rate, descending)
- `Leaderboard` coverage rule: include **all algorithms evaluated in the current benchmark scope**. If a provided leaderboard artifact is incomplete, backfill missing algorithms from aggregate summaries before rendering.

Use this pattern:

```python
from datetime import datetime
from pathlib import Path


def write_markdown_report(
    benchmark_name: str,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    leaderboard_rows: list[dict[str, object]],
    notes: list[str] | None = None,
) -> Path:
    """Write benchmark results into a single markdown report file."""
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{benchmark_name}_{ts}.md"

    lines: list[str] = [
        f"# {benchmark_name} Benchmark Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Time & Memory",
        "",
    ]

    if perf_rows:
        perf_headers = list(perf_rows[0].keys())
        lines.append("| " + " | ".join(perf_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(perf_headers)) + " |")
        for row in perf_rows:
            lines.append("| " + " | ".join(str(row[h]) for h in perf_headers) + " |")
    else:
        lines.append("No time/memory rows were produced.")

    lines.extend(["", "## Success & Other Metrics", ""])

    if metric_rows:
        metric_headers = list(metric_rows[0].keys())
        lines.append("| " + " | ".join(metric_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(metric_headers)) + " |")
        for row in metric_rows:
            lines.append(
                "| " + " | ".join(str(row[h]) for h in metric_headers) + " |"
            )
    else:
        lines.append("No success/metric rows were produced.")

    lines.extend(["", "## Leaderboard", ""])

    if leaderboard_rows:
        leaderboard_headers = list(leaderboard_rows[0].keys())
        lines.append("| " + " | ".join(leaderboard_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(leaderboard_headers)) + " |")
        for row in leaderboard_rows:
            lines.append(
                "| " + " | ".join(str(row[h]) for h in leaderboard_headers) + " |"
            )
    else:
        lines.append("No leaderboard rows were produced.")

    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in notes])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
```

And call it at the end of `run_all_benchmarks()`:

```python
def run_all_benchmarks() -> None:
    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    perf_part, metric_part = benchmark_halton_sampler()
    perf_rows.extend(perf_part)
    metric_rows.extend(metric_part)
    perf_part, metric_part = benchmark_density_metric()
    perf_rows.extend(perf_part)
    metric_rows.extend(metric_part)
    # ...

    leaderboard_rows = build_leaderboard_rows(metric_rows)
    # `build_leaderboard_rows` should aggregate per algorithm and sort by
    # overall success rate in descending order.

    report_path = write_markdown_report(
        benchmark_name="workspace_analyzer",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=["Memory uses a separate warmed call; latency is median completed-call wall time."],
    )
    print(f"Markdown report saved: {report_path}")
```

---

## Output Format Reference

| Scenario | Print format |
|----------|-------------|
| Single implementation, many sizes | `n={n:>7d}: {elapsed*1000:>10.2f} ms \| CPU Δ={...:+.1f} MB  GPU Δ={...:+.1f} MB  peak GPU={...:.1f} MB` |
| Two implementations compared | `===<Impl> time: {ms:.6f} ms` then error & memory lines indented 3 spaces |
| Markdown report path | `Markdown report saved: outputs/benchmarks/<name>_<timestamp>.md` |
| Markdown table 1 (Time & Memory) | `| sample_size | impl | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |` |
| Markdown table 2 (Success & Metrics) | `| sample_size | impl | success_rate | translation_err_mm | rotation_err_deg | ... |` |
| Markdown table 3 (Leaderboard) | `| rank | algorithm | overall_success_rate | ... |` (sorted by `overall_success_rate` descending) |
| Section header | `\n=== <Name> Benchmark ===` |
| Top-level separator | `"=" * 60` |

---

## Measuring Memory Usage

Measure memory in a separate representative call after warming the implementation
and releasing earlier outputs. Keep inputs and retained-output policy equivalent
across implementations; allocator reuse is part of the stated steady-state
scope. Report persistent buffer/cache storage separately when warmup allocated
it. A cold-start or capacity-growth case needs its own labeled measurement.

Use process RSS for CPU retained-memory delta. Before/after RSS does not measure
transient peak memory; use an appropriate peak profiler when CPU peak memory is
material. Torch allocated/peak statistics cover its CUDA allocator, not all
Warp/native allocations or total device usage. State allocator coverage and use
backend-specific accounting or explicit buffer sizes where needed. Mark
unavailable or inapplicable metrics as such rather than reporting zero usage.

This example retains one output and reports total Torch peak allocated memory
plus the additional peak above the warmed baseline. Synchronization must cover
all participating work, as in the latency helper:

```python
from collections.abc import Callable
import psutil
import torch


def measure_memory(
    operation: Callable[[], object],
    synchronize: Callable[[], None],
    *,
    device: torch.device | None = None,
) -> tuple[object, dict[str, float | None]]:
    """Measure a separate warmed call, retaining one output for validation."""
    synchronize()
    process = psutil.Process()
    mib = 1024**2
    cpu_before = process.memory_info().rss
    cuda = device is not None and device.type == "cuda"
    gpu_before = torch.cuda.memory_allocated(device) if cuda else 0
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    result = operation()
    synchronize()
    metrics: dict[str, float | None] = {
        "cpu_delta_mb": (process.memory_info().rss - cpu_before) / mib,
        "gpu_delta_mb": None,
        "peak_gpu_mb": None,
        "peak_gpu_delta_mb": None,
    }
    if cuda:
        peak = torch.cuda.max_memory_allocated(device)
        metrics.update(
            gpu_delta_mb=(torch.cuda.memory_allocated(device) - gpu_before) / mib,
            peak_gpu_mb=peak / mib,
            peak_gpu_delta_mb=(peak - gpu_before) / mib,
        )
    return result, metrics
```

Use existing project dependencies for optional instrumentation. Missing hardware
or instrumentation must produce an explicit skipped/unmeasured field and scope
limitation, not a fabricated result or a default dependency change.

---

## Common Imports

```python
import os
import time
import psutil
import numpy as np
import torch
import warp as wp                          # only when GPU kernels are benchmarked
from scipy.spatial.transform import Rotation  # only when needed
from collections.abc import Callable       # when building measurement helpers
```

---

## Quick Checklist

Before finishing a benchmark script:

- [ ] Apache 2.0 copyright header is present
- [ ] Module-level docstring with `Run:` line
- [ ] Each function has a one-line docstring
- [ ] Workload sizes, dtype/device, hardware/software and CPU thread counts recorded
- [ ] Equivalent seeded inputs, quality settings and success criteria for comparisons
- [ ] Warmup is excluded from repeated steady-state samples; cold-start cost labeled if measured
- [ ] Wall timing uses `time.perf_counter()` with explicit backend/device synchronization
- [ ] Median, repeats and relevant variability/tail metrics reported
- [ ] Device-resident, transfer-inclusive and end-to-end scopes distinguished
- [ ] Memory measured in a separate warmed call with equivalent output lifetime
- [ ] CPU RSS delta labeled as retained delta, not transient peak
- [ ] GPU allocator coverage and total versus incremental peak clearly labeled
- [ ] Persistent buffers/caches and unavailable metrics explicitly accounted for
- [ ] Accuracy metrics reported alongside timing (for solver benchmarks)
- [ ] Graceful skip for benchmarks that need unavailable hardware
- [ ] `run_all_benchmarks()` orchestrator with formatted separators
- [ ] Results are written to exactly one Markdown report file per run
- [ ] Report contains exactly three Markdown tables: `Time & Memory`, `Success & Other Metrics`, and `Leaderboard`
- [ ] `Time & Memory` table includes `cost_time_ms`, `cpu_delta_mb`, `gpu_delta_mb`, `peak_gpu_mb`
- [ ] `Success & Other Metrics` table includes `success_rate` and domain-specific quality metrics
- [ ] `Leaderboard` table ranks algorithms by overall success rate in descending order
- [ ] `Leaderboard` table includes all benchmarked algorithms (missing entries are backfilled from aggregate summaries if needed)
- [ ] Console log includes final report path
- [ ] `if __name__ == "__main__":` entry point
- [ ] `black .` formatting applied
