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

"""Typed, replayable state transitions for cross-engine orchestration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .workflow_contracts import TaskRunRequest, validate_task_run_request

__all__ = [
    "StageStatus",
    "TaskEngineState",
    "WorkflowStage",
    "complete_stage",
    "fail_stage",
    "initial_state",
    "skip_stage",
    "start_stage",
]


class WorkflowStage(str, Enum):
    """Stable stages shared by all four supported input combinations."""

    INPUT = "input"
    TASK_CANDIDATES = "task_candidates"
    SCENE_PREPARATION = "scene_preparation"
    SCENE_EDIT = "scene_edit"
    CANDIDATE_SELECTION = "candidate_selection"
    SCENE_FINALIZATION = "scene_finalization"
    UNBOUND_ACTION = "unbound_action"
    FINAL_INSPECTION = "final_inspection"
    FINAL_BINDING = "final_binding"
    STATIC_FEASIBILITY = "static_feasibility"
    GROUNDED_ACTION = "grounded_action"
    EXECUTION = "execution"


class StageStatus(str, Enum):
    """Lifecycle of one independently schedulable workflow stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


_DEPENDENCIES: dict[WorkflowStage, frozenset[WorkflowStage]] = {
    WorkflowStage.INPUT: frozenset(),
    WorkflowStage.TASK_CANDIDATES: frozenset({WorkflowStage.INPUT}),
    WorkflowStage.SCENE_PREPARATION: frozenset({WorkflowStage.INPUT}),
    WorkflowStage.SCENE_EDIT: frozenset({WorkflowStage.SCENE_PREPARATION}),
    WorkflowStage.CANDIDATE_SELECTION: frozenset(
        {
            WorkflowStage.TASK_CANDIDATES,
            WorkflowStage.SCENE_PREPARATION,
            WorkflowStage.SCENE_EDIT,
        }
    ),
    WorkflowStage.SCENE_FINALIZATION: frozenset({WorkflowStage.CANDIDATE_SELECTION}),
    WorkflowStage.UNBOUND_ACTION: frozenset({WorkflowStage.CANDIDATE_SELECTION}),
    WorkflowStage.FINAL_INSPECTION: frozenset({WorkflowStage.SCENE_FINALIZATION}),
    WorkflowStage.FINAL_BINDING: frozenset(
        {WorkflowStage.FINAL_INSPECTION, WorkflowStage.UNBOUND_ACTION}
    ),
    WorkflowStage.STATIC_FEASIBILITY: frozenset({WorkflowStage.FINAL_BINDING}),
    WorkflowStage.GROUNDED_ACTION: frozenset({WorkflowStage.STATIC_FEASIBILITY}),
    WorkflowStage.EXECUTION: frozenset({WorkflowStage.GROUNDED_ACTION}),
}


@dataclass(frozen=True)
class TaskEngineState:
    """Immutable state snapshot plus an append-only transition audit."""

    request: TaskRunRequest
    stages: dict[WorkflowStage, StageStatus]
    events: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def terminal(self) -> bool:
        """Return whether execution succeeded or any stage failed."""
        return (
            self.stages[WorkflowStage.EXECUTION] == StageStatus.SUCCEEDED
            or StageStatus.FAILED in self.stages.values()
        )

    def to_dict(self) -> dict[str, Any]:
        """Return one JSON-safe audit snapshot."""
        return {
            "request": deepcopy(self.request),
            "stages": {
                stage.value: self.stages[stage].value for stage in WorkflowStage
            },
            "events": deepcopy(list(self.events)),
        }


def initial_state(request: TaskRunRequest) -> TaskEngineState:
    """Create a validated state with the optional edit stage resolved."""
    normalized = validate_task_run_request(request)
    stages = {stage: StageStatus.PENDING for stage in WorkflowStage}
    stages[WorkflowStage.INPUT] = StageStatus.SUCCEEDED
    events = (
        {
            "sequence": 1,
            "stage": WorkflowStage.INPUT.value,
            "from": StageStatus.PENDING.value,
            "to": StageStatus.SUCCEEDED.value,
        },
    )
    state = TaskEngineState(request=normalized, stages=stages, events=events)
    if normalized["scene_edit_prompt"] is None:
        state = skip_stage(state, WorkflowStage.SCENE_EDIT)
    return state


def start_stage(state: TaskEngineState, stage: WorkflowStage) -> TaskEngineState:
    """Start a pending stage only after every dependency has completed."""
    if state.terminal:
        raise ValueError("A terminal TaskEngineState cannot start another stage.")
    if state.stages[stage] != StageStatus.PENDING:
        raise ValueError(f"Stage {stage.value!r} is not pending.")
    incomplete = [
        dependency.value
        for dependency in _DEPENDENCIES[stage]
        if state.stages[dependency] not in {StageStatus.SUCCEEDED, StageStatus.SKIPPED}
    ]
    if incomplete:
        raise ValueError(
            f"Stage {stage.value!r} has incomplete dependencies: {incomplete}."
        )
    return _transition(state, stage, StageStatus.RUNNING)


def complete_stage(state: TaskEngineState, stage: WorkflowStage) -> TaskEngineState:
    """Complete one running stage."""
    if state.stages[stage] != StageStatus.RUNNING:
        raise ValueError(f"Stage {stage.value!r} is not running.")
    return _transition(state, stage, StageStatus.SUCCEEDED)


def fail_stage(
    state: TaskEngineState,
    stage: WorkflowStage,
    *,
    reason: str,
) -> TaskEngineState:
    """Fail a pending or running stage with one auditable reason."""
    if state.terminal:
        raise ValueError("A terminal TaskEngineState cannot fail another stage.")
    if state.stages[stage] not in {StageStatus.PENDING, StageStatus.RUNNING}:
        raise ValueError(f"Stage {stage.value!r} cannot be failed now.")
    normalized_reason = str(reason).strip()
    if not normalized_reason:
        raise ValueError("A failed stage requires a non-empty reason.")
    return _transition(
        state,
        stage,
        StageStatus.FAILED,
        details={"reason": normalized_reason},
    )


def skip_stage(state: TaskEngineState, stage: WorkflowStage) -> TaskEngineState:
    """Skip one optional pending stage."""
    if state.stages[stage] != StageStatus.PENDING:
        raise ValueError(f"Stage {stage.value!r} is not pending.")
    return _transition(state, stage, StageStatus.SKIPPED)


def _transition(
    state: TaskEngineState,
    stage: WorkflowStage,
    status: StageStatus,
    *,
    details: dict[str, Any] | None = None,
) -> TaskEngineState:
    previous = state.stages[stage]
    stages = dict(state.stages)
    stages[stage] = status
    event = {
        "sequence": len(state.events) + 1,
        "stage": stage.value,
        "from": previous.value,
        "to": status.value,
    }
    if details:
        event.update(deepcopy(details))
    return TaskEngineState(
        request=deepcopy(state.request),
        stages=stages,
        events=(*state.events, event),
    )
