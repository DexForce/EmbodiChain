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

"""Provider-independent plan-transform requests for Task Program calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    AtomicActionEngine,
    PlanTransform,
)
from embodichain.lab.task_program.semantics.calls import SemanticCallSpec

__all__ = ["TaskProgramPlanRequest", "TaskProgramPlanTransformFactory"]


@dataclass(frozen=True, slots=True)
class TaskProgramPlanRequest:
    """Stable Task Program identity for one grounded Atomic planning call."""

    workflow_id: str
    workflow_call_index: int
    analysis_call_index: int
    call: SemanticCallSpec
    invocation: ActionInvocation

    def __post_init__(self) -> None:
        if (
            type(self.workflow_id) is not str
            or not self.workflow_id
            or self.workflow_id != self.workflow_id.strip()
        ):
            raise ValueError("workflow_id must be a non-empty stripped string")
        for name in ("workflow_call_index", "analysis_call_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.call, SemanticCallSpec):
            raise TypeError("call must be a SemanticCallSpec")
        if not isinstance(self.invocation, ActionInvocation):
            raise TypeError("invocation must be an ActionInvocation")


@runtime_checkable
class TaskProgramPlanTransformFactory(Protocol):
    """Create an optional Atomic plan transform for one grounded call."""

    def create_plan_transform(
        self,
        request: TaskProgramPlanRequest,
        *,
        engine: AtomicActionEngine,
    ) -> PlanTransform | None:
        """Return one session-scoped transform or ``None`` for no change."""
