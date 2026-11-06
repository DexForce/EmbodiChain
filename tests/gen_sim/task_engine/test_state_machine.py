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

"""Request and state contracts are independently replayable without runtime owners."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from embodichain.gen_sim.task_engine.workflow_contracts import (
    TASK_RUN_REQUEST_SCHEMA,
    validate_task_run_request,
)
from embodichain.gen_sim.task_engine.state_machine import (
    StageStatus,
    WorkflowStage,
    complete_stage,
    fail_stage,
    initial_state,
    replay_events,
    start_stage,
)


def _request(tmp_path: Path) -> dict:
    return dict(
        schema_version=TASK_RUN_REQUEST_SCHEMA,
        task_id="fixed_task",
        task_instruction="Move the declared payload.",
        image_path=None,
        gym_project=str(tmp_path / "scene"),
        scene_edit_prompt=None,
        output_dir=str(tmp_path / "run"),
    )


def test_state_replay_preserves_ordered_outcomes_and_detached_request(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    original = deepcopy(request)
    state = initial_state(request)
    assert state.stages[WorkflowStage.INPUT] is StageStatus.SUCCEEDED
    assert state.stages[WorkflowStage.SCENE_EDIT] is StageStatus.SKIPPED
    running = start_stage(state, WorkflowStage.TASK_CANDIDATES)
    finished = complete_stage(running, WorkflowStage.TASK_CANDIDATES)
    failed = fail_stage(
        finished, WorkflowStage.SCENE_PREPARATION, reason="source missing"
    )
    assert failed.terminal
    assert state.stages[WorkflowStage.TASK_CANDIDATES] is StageStatus.PENDING
    replayed = replay_events(request, failed.events)
    assert replayed.to_dict() == failed.to_dict()
    assert replayed.events[-1]["reason"] == "source missing"
    request["task_instruction"] = "caller mutation"
    assert failed.request["task_instruction"] == original["task_instruction"]


def test_invalid_dependencies_audit_drift_and_overlapping_outputs_reject(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    state = initial_state(request)
    with pytest.raises(ValueError, match="incomplete dependencies"):
        start_stage(state, WorkflowStage.EXECUTION)
    with pytest.raises(ValueError, match="not running"):
        complete_stage(state, WorkflowStage.TASK_CANDIDATES)
    changed = deepcopy([dict(row) for row in state.events])
    changed[0]["sequence"] = 99
    with pytest.raises(ValueError, match="canonical initial state"):
        replay_events(request, changed)
    overlap = dict(request, output_dir=str(tmp_path / "scene" / "new-run"))
    with pytest.raises(ValueError, match="must not overlap"):
        validate_task_run_request(overlap)
    unsupported = dict(request, image_path="image.png")
    with pytest.raises(ValueError, match="existing scenes only"):
        validate_task_run_request(unsupported)
