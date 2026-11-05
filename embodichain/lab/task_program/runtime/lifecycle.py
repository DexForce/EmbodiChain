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

"""Per-call assembly of explicit Atomic execution-plan lifecycle ports."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from dataclasses import dataclass

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ExecutionPlanLifecycle,
)
from .expansion import TaskProgramPlanRequest

__all__ = [
    "TaskProgramPlanLifecycleFactory",
    "CompositeTaskProgramPlanLifecycleFactory",
]


@runtime_checkable
class TaskProgramPlanLifecycleFactory(Protocol):
    """Select a declared execution owner for one exact grounded occurrence.

    This factory does not transform plans, modify policies or sample simulation.
    The returned lifecycle sees validated installed plans before dispatch and
    receives abort notifications through the ordinary execution runner.
    """

    def create_plan_lifecycle(
        self,
        request: TaskProgramPlanRequest,
        *,
        engine: AtomicActionEngine,
    ) -> ExecutionPlanLifecycle | None:
        """Select one execution owner for the supplied grounded occurrence.

        Args:
            request: Exact grounded occurrence whose execution is being assembled.
            engine: Atomic engine owning its resolved bindings and plans.

        Returns:
            Explicit lifecycle port, or None for ordinary execution.
        """
        ...


@dataclass(frozen=True, slots=True)
class CompositeTaskProgramPlanLifecycleFactory:
    """Borrow explicit runtime owners and require one lifecycle per occurrence.

    Args:
        factories: Nonempty tuple of distinct explicitly selected runtime owners.
    """

    factories: tuple[TaskProgramPlanLifecycleFactory, ...]

    def __post_init__(self) -> None:
        if (
            type(self.factories) is not tuple
            or not self.factories
            or any(
                not isinstance(value, TaskProgramPlanLifecycleFactory)
                for value in self.factories
            )
        ):
            raise TypeError(
                "Lifecycle composition requires an explicit tuple of factories."
            )
        if len({id(value) for value in self.factories}) != len(self.factories):
            raise ValueError("Lifecycle factories cannot be registered twice.")

    def create_plan_lifecycle(
        self, request: TaskProgramPlanRequest, *, engine: AtomicActionEngine
    ) -> ExecutionPlanLifecycle | None:
        """Select one owner without copying its state or plan.

        Args:
            request: Exact grounded occurrence forwarded to every factory.
            engine: Atomic engine forwarded to the selected runtime owners.

        Returns:
            Selected lifecycle, or None when no factory owns this occurrence.

        Raises:
            TypeError: A factory returns an unsupported execution owner.
            ValueError: More than one factory owns this occurrence.
        """
        selected = [
            value
            for factory in self.factories
            if (value := factory.create_plan_lifecycle(request, engine=engine))
            is not None
        ]
        if any(not isinstance(value, ExecutionPlanLifecycle) for value in selected):
            raise TypeError(
                "Lifecycle factory returned an unsupported execution owner."
            )
        if len(selected) > 1:
            raise ValueError("More than one lifecycle owns this exact invocation.")
        return selected[0] if selected else None
