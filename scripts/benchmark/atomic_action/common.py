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

"""Shared helpers for atomic-action benchmark scripts."""

from __future__ import annotations

import argparse
import math
import os
import resource
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

CPU_MEMORY_BACKEND = "psutil" if psutil is not None else "resource"


def ensure_repo_root() -> None:
    """Add the repository root to sys.path for module execution."""
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def ensure_torch():
    """Import torch or raise a clear benchmark runtime error."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Atomic action benchmark requires the EmbodiChain simulation runtime "
            f"and PyTorch. Missing module: {exc.name}."
        ) from exc
    return torch


def add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add common atomic-action benchmark CLI arguments."""
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repeats for every benchmark case.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one representative case only for quick validation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Simulation device, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        choices=("auto", "hybrid", "fast-rt", "rt"),
        default="auto",
        help="Renderer backend used by SimulationManager.",
    )


def add_grasp_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add common grasp-affordance setup arguments."""
    parser.add_argument(
        "--n_sample",
        type=int,
        default=10000,
        help="Number of samples for antipodal grasp generation.",
    )
    parser.add_argument(
        "--force_reannotate",
        action="store_true",
        help="Force grasp region re-annotation instead of using cached data.",
    )


def sync_cuda() -> None:
    """Synchronize CUDA stream when available."""
    torch = ensure_torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_gpu_memory() -> None:
    """Reset PyTorch peak GPU memory stats when CUDA is available."""
    torch = ensure_torch()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb() -> float:
    """Return peak GPU memory allocated by PyTorch in MB."""
    torch = ensure_torch()
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def memory_snapshot() -> dict[str, float]:
    """Return current process memory usage snapshot in MB."""
    torch = ensure_torch()
    if psutil is not None:
        process = psutil.Process(os.getpid())
        cpu_mb = process.memory_info().rss / 1024**2
    else:
        cpu_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    gpu_mb = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    return {"cpu_mb": cpu_mb, "gpu_mb": gpu_mb}


def timed_call(
    callable_fn: Callable[[], object],
) -> tuple[float, dict[str, float], float, object]:
    """Time a callable and return elapsed seconds, memory deltas, peak GPU, result."""
    reset_peak_gpu_memory()
    before = memory_snapshot()
    sync_cuda()

    start = time.perf_counter()
    result = callable_fn()
    sync_cuda()
    elapsed = time.perf_counter() - start

    after = memory_snapshot()
    deltas = {
        "cpu_mb": after["cpu_mb"] - before["cpu_mb"],
        "gpu_mb": after["gpu_mb"] - before["gpu_mb"],
    }
    return elapsed, deltas, peak_gpu_memory_mb(), result


def reset_robot(robot, initial_qpos) -> None:
    """Reset current and target robot qpos to the benchmark initial posture."""
    for target in (False, True):
        robot.set_qpos(initial_qpos, target=target)
    robot.clear_dynamics()


def format_float(value: float | None, precision: int = 6) -> str:
    """Format finite floats for tables and use N/A for missing values."""
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{precision}f}"


def format_markdown_table(rows: list[dict[str, object]]) -> list[str]:
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


def write_markdown_report(
    benchmark_name: str,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    leaderboard_rows: list[dict[str, object]],
    notes: list[str] | None = None,
) -> Path:
    """Write benchmark results into one markdown report file."""
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
    lines.extend(format_markdown_table(perf_rows))
    lines.extend(["", "## Success & Other Metrics", ""])
    lines.extend(format_markdown_table(metric_rows))
    lines.extend(["", "## Leaderboard", ""])
    lines.extend(format_markdown_table(leaderboard_rows))

    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in notes])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_single_action_leaderboard(
    action_name: str,
    metric_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate rows for one action by benchmark case."""
    if not metric_rows:
        return []

    success_sum = sum(float(row["success_rate"]) for row in metric_rows)
    count = len(metric_rows)
    return [
        {
            "rank": 1,
            "algorithm": action_name,
            "overall_success_rate": f"{success_sum / max(count, 1):.2%}",
            "evaluated_cases": count,
        }
    ]


__all__ = [
    "CPU_MEMORY_BACKEND",
    "add_common_benchmark_args",
    "add_grasp_benchmark_args",
    "build_single_action_leaderboard",
    "ensure_repo_root",
    "ensure_torch",
    "format_float",
    "reset_robot",
    "timed_call",
    "write_markdown_report",
]
