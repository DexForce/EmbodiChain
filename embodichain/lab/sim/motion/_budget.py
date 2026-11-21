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

"""Context-local cooperative budgets for planning and nested recovery calls.

No active scope means no policy change. Hosts own deadlines/cancellation; this
module owns no simulator, workflow, counter state or dependency on integrations.
Native calls are checked before and after return, not forcibly interrupted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import math
from typing import Any, Protocol

__all__ = []


class _PlanningBudget(Protocol):
    def check(self) -> None: ...
    def charge(self, kind: str, amount: int = 1) -> None: ...


_ACTIVE_BUDGET: ContextVar[_PlanningBudget | None] = ContextVar(
    "embodichain_planning_budget", default=None
)


@contextmanager
def _planning_budget_scope(budget: _PlanningBudget) -> Iterator[None]:
    token = _ACTIVE_BUDGET.set(budget)
    try:
        budget.check()
        yield
        budget.check()
    finally:
        _ACTIVE_BUDGET.reset(token)


def _planning_call(kind: str, *, batched: bool = False) -> Callable:
    def decorate(method: Callable) -> Callable:
        @wraps(method)
        def bounded(*args: Any, **kwargs: Any) -> Any:
            budget = _ACTIVE_BUDGET.get()
            if budget is not None:
                if kind == "check":
                    budget.check()
                else:
                    amount = 1
                    if batched:
                        pose = kwargs.get("pose", args[1] if len(args) > 1 else None)
                        shape = getattr(pose, "shape", ())
                        if len(shape) >= 3:
                            amount = max(
                                1,
                                math.prod(
                                    shape[:-2] if shape[-2:] == (4, 4) else shape[:-1]
                                ),
                            )
                    budget.charge(kind, amount)
            result = method(*args, **kwargs)
            if budget is not None:
                budget.check()
            return result

        return bounded

    return decorate
