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

"""Call-local search measurements; no budgets or planner behavior changes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter

__all__: list[str] = []


@dataclass
class SearchCost:
    counters: Counter[str] = field(default_factory=Counter)
    inclusive_wall_seconds: dict[str, float] = field(default_factory=dict)

    def snapshot(self) -> dict[str, object]:
        """Return independent measurements, never a total derived from overlaps."""
        return {
            "scope": "call_local_search_cost_not_physical_evidence",
            "counters": dict(self.counters),
            "inclusive_wall_seconds": dict(self.inclusive_wall_seconds),
            "timings_overlap": True,
            "budget_enforced": False,
        }


_COST: ContextVar[SearchCost | None] = ContextVar("gen_sim_search_cost", default=None)


@contextmanager
def search_cost_scope() -> Iterator[SearchCost]:
    """Share one ledger across nested previews; isolate independent calls."""
    cost = _COST.get()
    if cost is not None:
        yield cost
        return
    cost = SearchCost()
    token = _COST.set(cost)
    try:
        yield cost
    finally:
        _COST.reset(token)


def record_search(operation: str, *, count: int = 1) -> None:
    """Record only while an explicit outer planning scope owns the ledger."""
    cost = _COST.get()
    if cost is not None:
        cost.counters[operation] += count


@contextmanager
def measure_search(operation: str) -> Iterator[None]:
    """Measure inclusive CPU wall time, including unsuccessful attempts."""
    cost = _COST.get()
    if cost is None:
        yield
        return
    cost.counters[operation] += 1
    start = perf_counter()
    try:
        yield
    finally:
        cost.inclusive_wall_seconds[operation] = (
            cost.inclusive_wall_seconds.get(operation, 0.0) + perf_counter() - start
        )
