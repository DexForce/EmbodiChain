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

"""Private timing and report helpers for the standalone grasp-yaw example."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

import numpy as np
import psutil
import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    operation: Callable[[], object], device: torch.device, repeats: int
) -> dict[str, float | None]:
    """Completed-call timing; memory uses a separate warmed call."""
    for _ in range(2):
        value = operation()
        _sync(device)
        del value
    latencies = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        value = operation()
        _sync(device)
        latencies.append((time.perf_counter() - start) * 1000)
        del value
    process = psutil.Process()
    _sync(device)
    rss = process.memory_info().rss
    gpu = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    if gpu is not None:
        torch.cuda.reset_peak_memory_stats(device)
    value = operation()
    _sync(device)
    memory = {
        "cpu_delta_mb": (process.memory_info().rss - rss) / 1024**2,
        "gpu_delta_mb": None,
        "peak_gpu_mb": None,
        "peak_gpu_delta_mb": None,
    }
    if gpu is not None:
        peak = torch.cuda.max_memory_allocated(device)
        memory.update(
            gpu_delta_mb=(torch.cuda.memory_allocated(device) - gpu) / 1024**2,
            peak_gpu_mb=peak / 1024**2,
            peak_gpu_delta_mb=(peak - gpu) / 1024**2,
        )
    del value
    return {
        "cost_time_ms": float(np.median(latencies)),
        "p90_time_ms": float(np.percentile(latencies, 90)),
        **memory,
    }


def _table(rows: list[dict[str, Any]]) -> str:
    headers = list(rows[0])

    def cell(value: Any) -> str:
        return (
            f"{value:.5g}"
            if isinstance(value, float)
            else "n/a" if value is None else str(value)
        )

    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join("---" for _ in headers)
        + " |\n"
        + "\n".join(
            "| " + " | ".join(cell(row[key]) for key in headers) + " |" for row in rows
        )
    )
