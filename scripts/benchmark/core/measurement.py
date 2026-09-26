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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import statistics
import time
from typing import Any, TypeVar

__all__ = [
    "AttemptMeasurement",
    "AttemptOutcome",
    "Measurement",
    "ResourceSample",
    "StageEvent",
    "StageMeasurement",
    "measure_attempts",
    "measure_loop",
    "measure_stages",
    "sample_resource",
]
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


@dataclass(frozen=True)
class StageEvent:
    """One lifecycle stage with an explicit success or failure status."""

    name: str
    status: str
    elapsed_s: float
    failure_reason: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable stage event."""
        return {
            "name": self.name,
            "status": self.status,
            "elapsed_s": self.elapsed_s,
            "failure_reason": self.failure_reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StageMeasurement:
    """Ordered stage events from loading, reset or scene construction."""

    events: tuple[StageEvent, ...]

    @property
    def total_s(self) -> float:
        """Return the sum of finite stage durations."""
        return sum(event.elapsed_s for event in self.events)

    @property
    def failed(self) -> int:
        """Return the number of failed stages."""
        return sum(event.status != "completed" for event in self.events)

    def to_dict(self) -> dict[str, object]:
        """Return the ordered stage timeline."""
        return {
            "events": [event.to_dict() for event in self.events],
            "total_s": self.total_s,
            "failed": self.failed,
        }


@dataclass(frozen=True)
class AttemptOutcome:
    """One bounded attempt, including validation and failure evidence."""

    index: int
    status: str
    elapsed_s: float
    metrics: Mapping[str, object] = field(default_factory=dict)
    failure_reason: str | None = None
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a serializable attempt outcome."""
        return {
            "index": self.index,
            "status": self.status,
            "elapsed_s": self.elapsed_s,
            "metrics": dict(self.metrics),
            "failure_reason": self.failure_reason,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class AttemptMeasurement:
    """Fixed-budget attempt outcomes with explicit zero-denominator semantics."""

    outcomes: tuple[AttemptOutcome, ...]

    @property
    def attempted(self) -> int:
        """Return the number of attempts actually started."""
        return len(self.outcomes)

    @property
    def completed(self) -> int:
        """Return the number of validated completed attempts."""
        return sum(outcome.status == "completed" for outcome in self.outcomes)

    @property
    def failed(self) -> int:
        """Return the number of failed attempts."""
        return self.attempted - self.completed

    @property
    def success_rate(self) -> float | None:
        """Return success rate, or ``None`` when no attempt was started."""
        return self.completed / self.attempted if self.attempted else None

    def to_dict(self) -> dict[str, object]:
        """Return outcomes and explicit denominator counters."""
        return {
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "attempted": self.attempted,
            "completed": self.completed,
            "failed": self.failed,
            "success_rate": self.success_rate,
        }


@dataclass(frozen=True)
class ResourceSample:
    """One resource observation with its collection method and unit."""

    kind: str
    value: float
    unit: str
    method: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible resource sample."""
        return {
            "kind": self.kind,
            "value": self.value,
            "unit": self.unit,
            "method": self.method,
        }


def _elapsed(clock: Callable[[], float], start: float) -> float:
    """Return and validate one monotonic duration."""
    duration = clock() - start
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Operation duration must be finite and nonnegative")
    return duration


def measure_stages(
    stages: Sequence[tuple[str, Callable[[], Any]]],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> StageMeasurement:
    """Run named lifecycle stages while retaining failure points.

    A failed stage is recorded and later stages continue so callers can report
    unsupported capabilities and cleanup evidence in one result.
    """
    events: list[StageEvent] = []
    for name, operation in stages:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("stage names must be nonempty strings")
        start = clock()
        try:
            value = operation()
            metadata = value if isinstance(value, Mapping) else {}
            events.append(
                StageEvent(name, "completed", _elapsed(clock, start), metadata=metadata)
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - evidence must preserve domain failures.
            events.append(
                StageEvent(
                    name,
                    "failed",
                    _elapsed(clock, start),
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return StageMeasurement(tuple(events))


def measure_attempts(
    operation: Callable[[int], Any],
    *,
    attempts: int,
    validate: Callable[[Any], bool | None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> AttemptMeasurement:
    """Execute a fixed number of attempts and retain every failure.

    The callback receives a zero-based attempt index. A validator returning
    ``False`` or raising marks that attempt failed; the next bounded attempt
    still runs. This makes retries visible in the denominator.
    """
    if type(attempts) is not int or attempts < 0:
        raise ValueError("attempts must be a nonnegative integer")
    outcomes: list[AttemptOutcome] = []
    for index in range(attempts):
        start = clock()
        try:
            value = operation(index)
            if validate is not None:
                valid = validate(value)
                if valid is False:
                    raise ValueError("validation_failed")
            outcomes.append(AttemptOutcome(index, "completed", _elapsed(clock, start)))
        except (
            Exception
        ) as exc:  # noqa: BLE001 - convert all domain outcomes to records.
            outcomes.append(
                AttemptOutcome(
                    index,
                    "failed",
                    _elapsed(clock, start),
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return AttemptMeasurement(tuple(outcomes))


def sample_resource(
    kind: str,
    sampler: Callable[[], int | float],
    *,
    unit: str,
    method: str = "callback",
) -> ResourceSample:
    """Collect one finite nonnegative resource sample through a domain callback."""
    if not all(
        isinstance(value, str) and value.strip() for value in (kind, unit, method)
    ):
        raise ValueError("resource kind, unit and method must be nonempty strings")
    value = float(sampler())
    if not math.isfinite(value) or value < 0:
        raise ValueError("resource sample must be finite and nonnegative")
    return ResourceSample(kind, value, unit, method)


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
