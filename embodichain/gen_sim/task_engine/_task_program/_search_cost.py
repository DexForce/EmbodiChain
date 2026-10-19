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

"""Call-local search measurements and optional cooperative wall deadlines."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
from time import perf_counter

__all__: list[str] = []


class PlanningBudgetExceeded(Exception):
    """A planning-only stop; never evidence of physical impossibility."""

    def __init__(self, operation: str, wall_seconds: float, elapsed: float) -> None:
        self.operation = operation
        self.wall_seconds = wall_seconds
        self.elapsed_seconds = elapsed
        super().__init__(
            f"Planning wall budget exhausted at {operation}: "
            f"{elapsed:.3f}s / {wall_seconds:.3f}s"
        )


@dataclass
class SearchCost:
    counters: Counter[str] = field(default_factory=Counter)
    inclusive_wall_seconds: dict[str, float] = field(default_factory=dict)
    wall_seconds: float | None = None
    started: float = 0.0

    def check_budget(self, operation: str) -> None:
        """Check at instrumented boundaries without changing a native operation."""
        if self.wall_seconds is not None:
            elapsed = perf_counter() - self.started
            if elapsed >= self.wall_seconds:
                raise PlanningBudgetExceeded(operation, self.wall_seconds, elapsed)

    def snapshot(self) -> dict[str, object]:
        """Return independent measurements, never a total derived from overlaps."""
        result = {
            "scope": "call_local_search_cost_not_physical_evidence",
            "counters": dict(self.counters),
            "inclusive_wall_seconds": dict(self.inclusive_wall_seconds),
            "timings_overlap": True,
            "budget_enforced": self.wall_seconds is not None,
        }
        if self.wall_seconds is not None:
            result.update(
                wall_budget_seconds=self.wall_seconds,
                wall_elapsed_seconds=perf_counter() - self.started,
                budget_mode="cooperative_operation_boundaries_not_native_preemption",
            )
        return result


_COST: ContextVar[SearchCost | None] = ContextVar("gen_sim_search_cost", default=None)


@contextmanager
def search_cost_scope(*, wall_seconds: float | None = None) -> Iterator[SearchCost]:
    """Share one ledger across nested previews; isolate independent calls."""
    cost = _COST.get()
    if cost is not None:
        yield cost
        return
    if wall_seconds is not None and (
        not math.isfinite(wall_seconds) or wall_seconds <= 0
    ):
        raise ValueError("Planning wall budget must be finite and positive.")
    cost = SearchCost(
        wall_seconds=wall_seconds,
        started=perf_counter() if wall_seconds is not None else 0.0,
    )
    token = _COST.set(cost)
    try:
        yield cost
    finally:
        _COST.reset(token)


def record_search(operation: str, *, count: int = 1) -> None:
    """Record only while an explicit outer planning scope owns the ledger."""
    cost = _COST.get()
    if cost is not None:
        cost.check_budget(operation)
        cost.counters[operation] += count


@contextmanager
def measure_search(operation: str) -> Iterator[None]:
    """Measure inclusive CPU wall time, including unsuccessful attempts."""
    cost = _COST.get()
    if cost is None:
        yield
        return
    cost.check_budget(operation)
    cost.counters[operation] += 1
    start = perf_counter()
    try:
        yield
    finally:
        cost.inclusive_wall_seconds[operation] = (
            cost.inclusive_wall_seconds.get(operation, 0.0) + perf_counter() - start
        )
    cost.check_budget(operation)
