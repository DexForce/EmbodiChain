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

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from .workflow_contracts import TaskRunRequest, validate_task_run_request

__all__ = [
    "StageStatus",
    "TaskEngineState",
    "WorkflowStage",
    "complete_stage",
    "fail_stage",
    "initial_state",
    "replay_events",
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
        }
    ),
    WorkflowStage.SCENE_FINALIZATION: frozenset(
        {WorkflowStage.CANDIDATE_SELECTION, WorkflowStage.SCENE_EDIT}
    ),
    WorkflowStage.UNBOUND_ACTION: frozenset({WorkflowStage.CANDIDATE_SELECTION}),
    WorkflowStage.FINAL_INSPECTION: frozenset({WorkflowStage.SCENE_FINALIZATION}),
    WorkflowStage.FINAL_BINDING: frozenset(
        {WorkflowStage.FINAL_INSPECTION, WorkflowStage.UNBOUND_ACTION}
    ),
    WorkflowStage.STATIC_FEASIBILITY: frozenset({WorkflowStage.FINAL_BINDING}),
    WorkflowStage.GROUNDED_ACTION: frozenset({WorkflowStage.STATIC_FEASIBILITY}),
    WorkflowStage.EXECUTION: frozenset({WorkflowStage.GROUNDED_ACTION}),
}

_SKIPPABLE_STAGES = frozenset({WorkflowStage.SCENE_EDIT})


@dataclass(frozen=True)
class TaskEngineState:
    """Immutable state snapshot plus an append-only transition audit."""

    request: Mapping[str, Any]
    stages: Mapping[WorkflowStage, StageStatus]
    events: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request",
            MappingProxyType(deepcopy(dict(self.request))),
        )
        object.__setattr__(
            self,
            "stages",
            MappingProxyType(dict(self.stages)),
        )
        object.__setattr__(
            self,
            "events",
            tuple(MappingProxyType(deepcopy(dict(event))) for event in self.events),
        )

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
            "request": deepcopy(dict(self.request)),
            "stages": {
                stage.value: self.stages[stage].value for stage in WorkflowStage
            },
            "events": deepcopy([dict(event) for event in self.events]),
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
    if stage not in _SKIPPABLE_STAGES:
        raise ValueError("Only the optional scene_edit stage can be skipped.")
    if state.stages[stage] != StageStatus.PENDING:
        raise ValueError(f"Stage {stage.value!r} is not pending.")
    return _transition(state, stage, StageStatus.SKIPPED)


def replay_events(
    request: TaskRunRequest,
    events: Sequence[Mapping[str, Any]],
) -> TaskEngineState:
    """Rebuild a state by validating and applying its transition audit.

    Args:
        request: Original workflow request used to create the state.
        events: Complete ordered event audit to validate and replay.

    Returns:
        The immutable state reconstructed from the supplied audit.

    Raises:
        TypeError: If the audit is not a sequence of event mappings.
        ValueError: If any event is missing, altered, or not a valid transition.
    """
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise TypeError("Task Engine events must be a sequence of mappings.")
    recorded = []
    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError("Each Task Engine event must be a mapping.")
        recorded.append(deepcopy(dict(event)))

    state = initial_state(request)
    initial_events = [dict(event) for event in state.events]
    if recorded[: len(initial_events)] != initial_events:
        raise ValueError("Replay event does not match the canonical initial state.")

    for expected in recorded[len(initial_events) :]:
        try:
            stage = WorkflowStage(expected["stage"])
            target = StageStatus(expected["to"])
            if target == StageStatus.RUNNING:
                replayed = start_stage(state, stage)
            elif target == StageStatus.SUCCEEDED:
                replayed = complete_stage(state, stage)
            elif target == StageStatus.FAILED:
                replayed = fail_stage(state, stage, reason=expected["reason"])
            elif target == StageStatus.SKIPPED:
                replayed = skip_stage(state, stage)
            else:
                raise ValueError(f"Unsupported replay target: {target.value!r}.")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Replay event does not match a valid transition.") from exc
        if dict(replayed.events[-1]) != expected:
            raise ValueError("Replay event does not match the generated transition.")
        state = replayed
    return state


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
        request=dict(state.request),
        stages=stages,
        events=(*state.events, event),
    )
