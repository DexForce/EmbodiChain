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

"""Synchronous timing without tensor, image or simulator dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import statistics
import time
from typing import TypeVar

__all__ = ["Measurement", "measure_loop"]
T = TypeVar("T")


@dataclass(frozen=True)
class Measurement:
    """Raw completed-call times [s] and the measured loop window [s]."""

    warmup: int
    latencies_s: tuple[float, ...]
    window_s: float

    @property
    def count(self) -> int:
        """Return the number of measured, validated operations."""
        return len(self.latencies_s)

    @property
    def operations_per_s(self) -> float:
        """Return throughput over the full loop window."""
        return self.count / self.window_s

    @property
    def p50_s(self) -> float:
        """Return median completed-call latency [s]."""
        return statistics.median(self.latencies_s)

    @property
    def p95_s(self) -> float:
        """Return nearest-rank P95 completed-call latency [s]."""
        return sorted(self.latencies_s)[math.ceil(0.95 * self.count) - 1]


def measure_loop(
    operation: Callable[[], T],
    *,
    warmup: int,
    iterations: int,
    validate: Callable[[T], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Measurement:
    """Measure completed operations; domain adapters own GPU synchronization.

    Validation is outside each operation's latency but inside the throughput
    window. The core never labels asynchronous host return as GPU completion.
    A rejected value raises instead of adding a successful sample.
    """
    for name, value, minimum in (("warmup", warmup, 0), ("iterations", iterations, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for _ in range(warmup):
        operation()
    latencies = []
    start_window = clock()
    for _ in range(iterations):
        start = clock()
        value = operation()
        elapsed = clock() - start
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Operation duration must be finite and nonnegative")
        if validate is not None:
            validate(value)
        latencies.append(elapsed)
    window = clock() - start_window
    if not math.isfinite(window) or window <= 0:
        raise ValueError("Measurement window must be positive and finite")
    return Measurement(warmup, tuple(latencies), window)
