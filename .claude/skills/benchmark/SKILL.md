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
- `scripts/benchmark/robotics/kinematic_solver/opw_solver.py`
- `scripts/benchmark/workspace_analyzer/benchmark_workspace_analyzer.py`

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
- **What is the primary metric?** (wall-clock time, mean error, throughput)
- **What sample sizes cover realistic usage?** Typically: `[100, 1000, 10000, 100000]`

### 2. Structure the Script

Use one helper function per concern, then a single orchestrator:

```
benchmark_<component_a>()   # e.g., benchmark_halton_sampler()
benchmark_<component_b>()   # e.g., benchmark_density_metric()
...
run_all_benchmarks()        # calls all of the above + prints header/footer
```

### 3. Write Individual Benchmark Functions

Each benchmark function follows this pattern:

```python
def benchmark_<name>():
    """One-line description of what is being measured."""
    from embodichain.<module.path> import SomeClass, SomeCfg

    # --- Setup (not timed) ---
    cfg = SomeCfg(...)
    obj = cfg.init_solver(...)  # or SomeClass(cfg)

    print("\n=== <Name> Benchmark ===")
    for n in [100, 1000, 10000, 100000]:
        # Prepare inputs (not timed)
        inputs = ...

        # --- Timed block ---
        start = time.perf_counter()
        result = obj.compute(inputs)       # or obj.get_ik(...) etc.
        elapsed = time.perf_counter() - start

        print(f"  n={n:>7d}: {elapsed*1000:>10.2f} ms (...)")
```

Key rules:
- Use `time.perf_counter()` for high-resolution wall-clock timing, **not** `time.time()`.
- Only time the core computation — exclude setup, data preparation, and print statements.
- Print results in milliseconds (`elapsed * 1000`) with consistent column alignment using `>` format specs.

> **Exception**: When benchmarking GPU (Warp/CUDA) code alongside a CPU baseline, it is acceptable to use `time.time()` for coarser comparison timing, as seen in `opw_solver.py`. Prefer `time.perf_counter()` for CPU-only benchmarks.

### 4. Comparing Two Implementations

When the benchmark compares two backends (e.g., Warp CUDA vs. Python OPW):

```python
def check_<name>(solver_a, solver_b, n_samples=1000):
    """Run both solvers and return timing + accuracy metrics."""
    # shared input generation
    qpos = ...

    # --- Solver A (e.g., Warp CUDA) ---
    start = time.time()
    success_a, result_a = solver_a.get_ik(xpos, ...)
    time_a = time.time() - start
    t_err_a, r_err_a = get_poses_err(...)

    # --- Solver B (e.g., CPU) ---
    start = time.time()
    success_b, result_b = solver_b.get_ik(xpos, ...)
    time_b = time.time() - start
    t_err_b, r_err_b = get_poses_err(...)

    return time_a, t_err_a, r_err_a, time_b, t_err_b, r_err_b


def benchmark_<name>():
    cfg = ...
    solver_a = cfg.init_solver(device=torch.device("cuda"), ...)
    solver_b = cfg.init_solver(device=torch.device("cpu"),  ...)

    for n in [100, 1000, 10000, 100000]:
        time_a, t_err_a, r_err_a, time_b, t_err_b, r_err_b = check_<name>(
            solver_a, solver_b, n_samples=n
        )
        print(f"**** Test over {n} samples:")
        print(f"===Impl A time:  {time_a * 1000:.6f} ms")
        print(f"   Translation mean error: {t_err_a * 1000:.6f} mm")
        print(f"   Rotation mean error:    {r_err_a * 180 / np.pi:.6f} degrees")
        print(f"===Impl B time:  {time_b * 1000:.6f} ms")
        ...
```

### 5. Report Accuracy Alongside Speed

For FK/IK solvers, always verify correctness by running FK on the IK output and measuring pose error:

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
- Requirement: output **exactly two Markdown tables** in the report
    1. `Time & Memory` table (cost time + memory columns)
    2. `Success & Other Metrics` table (success rate + quality/accuracy/extra metrics)

Use this pattern:

```python
from datetime import datetime
from pathlib import Path


def write_markdown_report(
    benchmark_name: str,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
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

    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in notes])

    report_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
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

    report_path = write_markdown_report(
        benchmark_name="workspace_analyzer",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        notes=["CPU/GPU memory fields are deltas measured around timed calls."],
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
| Section header | `\n=== <Name> Benchmark ===` |
| Top-level separator | `"=" * 60` |

---

## Measuring Memory Usage

Always measure **both GPU VRAM and CPU RAM** alongside wall-clock time. Use the helpers below.

### GPU VRAM (via PyTorch CUDA)

```python
import torch

def get_gpu_memory_mb() -> float:
    """Return current GPU VRAM allocated by PyTorch in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0

# Usage pattern inside a benchmark loop:
torch.cuda.reset_peak_memory_stats()          # reset peak counter before timed block
mem_before = get_gpu_memory_mb()

start = time.perf_counter()
result = obj.compute(inputs)
elapsed = time.perf_counter() - start

mem_after = get_gpu_memory_mb()
peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 2  # peak during timed block

print(
    f"  n={n:>7d}: {elapsed*1000:>10.2f} ms | "
    f"VRAM delta={mem_after - mem_before:+.1f} MB  peak={peak_vram:.1f} MB"
)
```

### CPU RAM (via `psutil`)

```python
import psutil, os

def get_cpu_memory_mb() -> float:
    """Return current process RSS (resident set size) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2

# Usage pattern:
mem_before = get_cpu_memory_mb()

start = time.perf_counter()
result = obj.compute(inputs)
elapsed = time.perf_counter() - start

mem_after = get_cpu_memory_mb()

print(
    f"  n={n:>7d}: {elapsed*1000:>10.2f} ms | "
    f"RAM delta={mem_after - mem_before:+.1f} MB"
)
```

### Combined Helper (recommended)

For benchmarks that use both CPU and GPU, combine into a single snapshot:

```python
import os, psutil, torch

def memory_snapshot() -> dict:
    """Return a dict with current CPU RSS and GPU allocated memory in MB."""
    process = psutil.Process(os.getpid())
    cpu_mb = process.memory_info().rss / 1024 ** 2
    gpu_mb = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
    return {"cpu_mb": cpu_mb, "gpu_mb": gpu_mb}

# Usage:
torch.cuda.reset_peak_memory_stats()
before = memory_snapshot()

start = time.perf_counter()
result = obj.compute(inputs)
elapsed = time.perf_counter() - start

after = memory_snapshot()
peak_gpu = torch.cuda.max_memory_allocated() / 1024 ** 2

print(
    f"  n={n:>7d}: {elapsed*1000:>10.2f} ms | "
    f"CPU Δ={after['cpu_mb'] - before['cpu_mb']:+.1f} MB  "
    f"GPU Δ={after['gpu_mb'] - before['gpu_mb']:+.1f} MB  peak GPU={peak_gpu:.1f} MB"
)
```

> Add `psutil` to the project's dev-dependencies if not already present (`pip install psutil`).

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
from typing import Tuple, List             # or use built-in generics (Python ≥ 3.10)
```

---

## Quick Checklist

Before finishing a benchmark script:

- [ ] Apache 2.0 copyright header is present
- [ ] Module-level docstring with `Run:` line
- [ ] Each function has a one-line docstring
- [ ] Setup code is **outside** the timed block
- [ ] Timing uses `time.perf_counter()` (or `time.time()` when comparing GPU/CPU coarsely)
- [ ] CPU RAM measured with `psutil` (delta MB before/after timed block)
- [ ] GPU VRAM measured with `torch.cuda.memory_allocated()` + `torch.cuda.max_memory_allocated()` (delta + peak)
- [ ] `torch.cuda.reset_peak_memory_stats()` called before each timed block
- [ ] Accuracy metrics reported alongside timing (for solver benchmarks)
- [ ] Graceful skip for benchmarks that need unavailable hardware
- [ ] `run_all_benchmarks()` orchestrator with formatted separators
- [ ] Results are written to exactly one Markdown report file per run
- [ ] Report contains exactly two Markdown tables: `Time & Memory` and `Success & Other Metrics`
- [ ] `Time & Memory` table includes `cost_time_ms`, `cpu_delta_mb`, `gpu_delta_mb`, `peak_gpu_mb`
- [ ] `Success & Other Metrics` table includes `success_rate` and domain-specific quality metrics
- [ ] Console log includes final report path
- [ ] `if __name__ == "__main__":` entry point
- [ ] `black .` formatting applied
