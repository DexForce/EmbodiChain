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

"""Wall-clock and process-memory measurement helpers."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import psutil
import torch

__all__ = ["TimedCall", "timed_call"]

_T = TypeVar("_T")


@dataclass(frozen=True)
class TimedCall(Generic[_T]):
    """Result and resource deltas captured around one callable."""

    result: _T
    cost_time_ms: float
    cpu_delta_mb: float
    gpu_delta_mb: float
    peak_gpu_mb: float


def _sync_cuda() -> None:
    """Synchronize CUDA before and after timed operations when available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _memory_snapshot() -> tuple[float, float]:
    """Return current process RSS and PyTorch GPU allocation in MB."""
    cpu_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    gpu_mb = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    return cpu_mb, gpu_mb


def timed_call(callable_fn: Callable[[], _T]) -> TimedCall[_T]:
    """Time only ``callable_fn`` and capture CPU/GPU memory deltas."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cpu_before, gpu_before = _memory_snapshot()
    _sync_cuda()

    start = time.perf_counter()
    result = callable_fn()
    _sync_cuda()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    cpu_after, gpu_after = _memory_snapshot()
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else None
    )
    return TimedCall(
        result=result,
        cost_time_ms=elapsed_ms,
        cpu_delta_mb=cpu_after - cpu_before,
        gpu_delta_mb=gpu_after - gpu_before,
        peak_gpu_mb=peak_gpu_mb,
    )
