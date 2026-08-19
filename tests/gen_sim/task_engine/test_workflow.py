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

from __future__ import annotations

from pathlib import Path

import pytest

from embodichain.gen_sim.task_engine.config import TaskEngineWorkflowCfg
from embodichain.gen_sim.task_engine.state_machine import (
    StageStatus,
    WorkflowStage,
    complete_stage,
    fail_stage,
    initial_state,
    replay_events,
    start_stage,
    skip_stage,
)
from embodichain.gen_sim.task_engine.workflow_contracts import (
    TASK_RUN_REQUEST_SCHEMA,
    scene_input_kind,
    validate_task_run_request,
)


def _request(tmp_path: Path, *, image: bool, edit: bool) -> dict[str, object]:
    return {
        "schema_version": TASK_RUN_REQUEST_SCHEMA,
        "task_id": "pick-cup",
        "task_instruction": "Pick up the red cup.",
        "image_path": str(tmp_path / "input.png") if image else None,
        "gym_project": None if image else str(tmp_path / "gym_project"),
        "scene_edit_prompt": "Add a tray." if edit else None,
        "output_dir": str(tmp_path / "output"),
    }


@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("edit", [False, True])
def test_run_request_accepts_all_four_input_combinations(
    tmp_path: Path,
    image: bool,
    edit: bool,
) -> None:
    request = validate_task_run_request(_request(tmp_path, image=image, edit=edit))
    assert scene_input_kind(request) == ("image" if image else "gym_project")
    assert request["scene_edit_prompt"] == ("Add a tray." if edit else None)


def test_run_request_rejects_two_scene_inputs(tmp_path: Path) -> None:
    request = _request(tmp_path, image=True, edit=False)
    request["gym_project"] = str(tmp_path / "gym_project")
    with pytest.raises(ValueError, match="exactly one"):
        validate_task_run_request(request)


def test_run_request_rejects_scene_generation_prompt(tmp_path: Path) -> None:
    request = _request(tmp_path, image=True, edit=False)
    request["scene_generation_prompt"] = "Make a kitchen."
    with pytest.raises(ValueError, match="fields differ"):
        validate_task_run_request(request)


def test_run_request_rejects_output_inside_gym_project(tmp_path: Path) -> None:
    request = _request(tmp_path, image=False, edit=False)
    request["output_dir"] = str(tmp_path / "gym_project" / "task_run")

    with pytest.raises(ValueError, match="must not overlap"):
        validate_task_run_request(request)


def test_run_request_rejects_output_containing_explicit_gym_config(
    tmp_path: Path,
) -> None:
    project = tmp_path / "gym_project"
    project.mkdir()
    config_path = project / "gym_config.json"
    config_path.write_text("{}", encoding="utf-8")
    request = _request(tmp_path, image=False, edit=False)
    request["gym_project"] = str(config_path)
    request["output_dir"] = str(project)

    with pytest.raises(ValueError, match="must not overlap"):
        validate_task_run_request(request)


def test_task_and_scene_stages_can_run_concurrently(tmp_path: Path) -> None:
    state = initial_state(_request(tmp_path, image=True, edit=False))
    state = start_stage(state, WorkflowStage.TASK_CANDIDATES)
    state = start_stage(state, WorkflowStage.SCENE_PREPARATION)
    assert state.stages[WorkflowStage.TASK_CANDIDATES] == StageStatus.RUNNING
    assert state.stages[WorkflowStage.SCENE_PREPARATION] == StageStatus.RUNNING
    assert state.stages[WorkflowStage.SCENE_EDIT] == StageStatus.SKIPPED


def test_candidate_selection_waits_for_both_branches(tmp_path: Path) -> None:
    state = initial_state(_request(tmp_path, image=False, edit=False))
    state = start_stage(state, WorkflowStage.TASK_CANDIDATES)
    state = complete_stage(state, WorkflowStage.TASK_CANDIDATES)
    with pytest.raises(ValueError, match="incomplete dependencies"):
        start_stage(state, WorkflowStage.CANDIDATE_SELECTION)


def test_only_scene_edit_can_be_skipped(tmp_path: Path) -> None:
    state = initial_state(_request(tmp_path, image=True, edit=True))

    with pytest.raises(ValueError, match="Only the optional scene_edit stage"):
        skip_stage(state, WorkflowStage.FINAL_BINDING)


def test_state_events_replay_to_the_same_snapshot(tmp_path: Path) -> None:
    request = _request(tmp_path, image=True, edit=False)
    state = initial_state(request)
    state = start_stage(state, WorkflowStage.TASK_CANDIDATES)
    state = start_stage(state, WorkflowStage.SCENE_PREPARATION)
    state = complete_stage(state, WorkflowStage.TASK_CANDIDATES)
    state = complete_stage(state, WorkflowStage.SCENE_PREPARATION)
    state = start_stage(state, WorkflowStage.CANDIDATE_SELECTION)
    state = complete_stage(state, WorkflowStage.CANDIDATE_SELECTION)

    replayed = replay_events(request, state.events)

    assert replayed.to_dict() == state.to_dict()


def test_state_replay_rejects_tampered_transition(tmp_path: Path) -> None:
    request = _request(tmp_path, image=True, edit=False)
    state = initial_state(request)
    events = [dict(event) for event in state.events]
    events[-1]["stage"] = WorkflowStage.FINAL_BINDING.value

    with pytest.raises(ValueError, match="event does not match"):
        replay_events(request, events)


def test_state_replay_preserves_failure_reason(tmp_path: Path) -> None:
    request = _request(tmp_path, image=True, edit=False)
    state = initial_state(request)
    state = start_stage(state, WorkflowStage.TASK_CANDIDATES)
    state = fail_stage(state, WorkflowStage.TASK_CANDIDATES, reason="model timeout")

    replayed = replay_events(request, state.events)

    assert replayed.terminal
    assert replayed.to_dict() == state.to_dict()


def test_state_snapshot_mappings_are_immutable(tmp_path: Path) -> None:
    state = initial_state(_request(tmp_path, image=True, edit=False))

    with pytest.raises(TypeError):
        state.stages[WorkflowStage.TASK_CANDIDATES] = StageStatus.SUCCEEDED
    with pytest.raises(TypeError):
        state.request["task_id"] = "changed"
    with pytest.raises(TypeError):
        state.events[0]["to"] = StageStatus.FAILED.value


def test_workflow_configuration_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="max_scene_attempts"):
        TaskEngineWorkflowCfg(max_scene_attempts=0)
