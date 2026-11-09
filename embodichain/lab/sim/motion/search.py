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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
import random
from time import perf_counter
from typing import Any

import numpy as np
import torch

__all__ = [
    "PlanningBudgetExceeded",
    "SearchCost",
    "current_search_cost",
    "search_cost_scope",
    "record_search",
    "measure_search",
    "PlanningRandomState",
]


@dataclass(frozen=True, slots=True, eq=False)
class PlanningRandomState:
    """Owned Python/NumPy/Torch streams for replaying bounded candidate searches.

    Construct through :meth:`capture` and restore at the exact candidate and
    finally boundaries selected by the owning skill. Capturing does not draw a
    random value or seed a new stream. This is transient runtime state, not a
    serializable motion policy.
    """

    _python: Any
    _numpy: Any
    _cpu: torch.Tensor
    _cuda: dict[int, torch.Tensor]

    @classmethod
    def capture(cls, devices: Iterable[torch.device | str] = ()) -> PlanningRandomState:
        """Capture current streams for the explicitly involved CUDA devices.

        Args:
            devices: Runtime tensor/robot devices involved in the search.
                CPU devices add no CUDA state; indexless CUDA resolves current.

        Returns:
            Independently owned states with the source capture/restore order.
        """
        cuda_devices = set()
        for value in devices:
            device = torch.device(value)
            if device.type == "cuda":
                cuda_devices.add(
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
        return cls(
            random.getstate(),
            np.random.get_state(),
            torch.random.get_rng_state(),
            {index: torch.cuda.get_rng_state(index) for index in cuda_devices},
        )

    def restore(self) -> None:
        """Restore every captured stream without changing a search's policy."""
        random.setstate(self._python)
        np.random.set_state(self._numpy)
        torch.random.set_rng_state(self._cpu)
        for index, value in self._cuda.items():
            torch.cuda.set_rng_state(value, index)


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
    """Mutable measurements owned by one outer search scope and its previews."""

    counters: Counter[str] = field(default_factory=Counter)
    inclusive_wall_seconds: dict[str, float] = field(default_factory=dict)
    wall_seconds: float | None = None
    started: float = 0.0

    def check_budget(self, operation: str) -> None:
        """Check at instrumented boundaries without changing a native operation.

        Args:
            operation: Operation reported if the caller-owned deadline expires.
        """
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


_COST: ContextVar[SearchCost | None] = ContextVar(
    "embodichain_motion_search_cost", default=None
)


def current_search_cost() -> SearchCost | None:
    """Return the active caller-owned ledger, or ``None`` outside a scope."""
    return _COST.get()


@contextmanager
def search_cost_scope(*, wall_seconds: float | None = None) -> Iterator[SearchCost]:
    """Share one ledger across nested previews; isolate independent calls.

    Args:
        wall_seconds: Optional positive cooperative wall deadline. The outer
            scope owns the value; nested scopes reuse its existing ledger.

    Yields:
        The mutable measurements for this search, restored on scope exit.

    Raises:
        ValueError: If an outer deadline is nonpositive or nonfinite.
    """
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


def record_search(
    operation: str, *, count: int = 1, budget_checkpoint: bool = True
) -> None:
    """Record only while an explicit outer planning scope owns the ledger.

    Args:
        operation: Caller-defined measurement name.
        count: Number of operations to add.
        budget_checkpoint: Check the deadline here. False records additional
            migration measurements without introducing a new stop boundary.
    """
    cost = _COST.get()
    if cost is not None:
        if budget_checkpoint:
            cost.check_budget(operation)
        cost.counters[operation] += count


@contextmanager
def measure_search(operation: str) -> Iterator[None]:
    """Measure inclusive CPU wall time, including unsuccessful attempts.

    Args:
        operation: Caller-defined measurement name.

    Yields:
        Control to the measured operation. Nested timings overlap and must not
        be summed as a total. Native work is never forcibly interrupted.
    """
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
