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

from embodichain.gen_sim.task_engine.config import (
    TaskEngineExecutionCfg,
    TaskEnginePlanningCfg,
    TaskEngineWorkflowCfg,
    load_task_engine_config,
)
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
    validate_scene_history_root,
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


def test_scene_history_root_allows_a_source_from_a_prior_run(
    tmp_path: Path,
) -> None:
    history = tmp_path / "task_history"
    source = history / "20260820_105939" / "attempts" / "scene_export"
    source.mkdir(parents=True)

    validate_scene_history_root(source, history)

    request = _request(tmp_path, image=False, edit=False)
    request["gym_project"] = str(source)
    request["output_dir"] = str(history / "20260820_130000")
    assert validate_task_run_request(request)["gym_project"] == source.as_posix()


@pytest.mark.parametrize("relative_output", [".", "new_runs", "new_runs/task"])
def test_scene_history_root_rejects_writes_into_source_project(
    tmp_path: Path,
    relative_output: str,
) -> None:
    source = tmp_path / "scene_export"
    source.mkdir()
    output_root = source / relative_output

    with pytest.raises(ValueError, match="read-only source"):
        validate_scene_history_root(source, output_root)


def test_scene_history_root_resolves_symlinks_before_comparison(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scene_export"
    source.mkdir()
    source_link = tmp_path / "scene_link"
    source_link.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="read-only source"):
        validate_scene_history_root(source_link, source / "new_runs")


def test_scene_history_root_protects_explicit_config_parent(tmp_path: Path) -> None:
    source = tmp_path / "scene_export"
    source.mkdir()
    config = source / "scene_config.json"
    config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="read-only source"):
        validate_scene_history_root(config, source)


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


def test_unbound_action_can_run_while_user_scene_edit_is_running(
    tmp_path: Path,
) -> None:
    state = initial_state(_request(tmp_path, image=True, edit=True))
    for stage in (WorkflowStage.TASK_CANDIDATES, WorkflowStage.SCENE_PREPARATION):
        state = start_stage(state, stage)
        state = complete_stage(state, stage)
    state = start_stage(state, WorkflowStage.CANDIDATE_SELECTION)
    state = complete_stage(state, WorkflowStage.CANDIDATE_SELECTION)
    state = start_stage(state, WorkflowStage.SCENE_EDIT)
    state = start_stage(state, WorkflowStage.UNBOUND_ACTION)

    assert state.stages[WorkflowStage.SCENE_EDIT] == StageStatus.RUNNING
    assert state.stages[WorkflowStage.UNBOUND_ACTION] == StageStatus.RUNNING
    with pytest.raises(ValueError, match="incomplete dependencies"):
        start_stage(state, WorkflowStage.SCENE_FINALIZATION)


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


def test_later_retry_can_fail_a_previously_successful_stage(tmp_path: Path) -> None:
    request = _request(tmp_path, image=True, edit=False)
    state = initial_state(request)
    state = start_stage(state, WorkflowStage.SCENE_PREPARATION)
    state = complete_stage(state, WorkflowStage.SCENE_PREPARATION)
    state = fail_stage(
        state,
        WorkflowStage.SCENE_PREPARATION,
        reason="later scene attempt failed",
    )

    assert state.terminal
    assert replay_events(request, state.events).to_dict() == state.to_dict()


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


def test_packaged_workflow_configuration_uses_recovery_defaults() -> None:
    workflow, planning, execution = load_task_engine_config()

    assert workflow.max_scene_attempts == 2
    assert workflow.max_action_attempts == 1
    assert planning.candidate_count == 3
    assert planning.planning_mode == "offline"
    assert planning.max_episodes == 1
    assert planning.max_episode_steps == 6000
    assert execution.num_envs == 1
    assert execution.required_successes == 1


def test_workflow_configuration_can_be_tuned_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "task_engine.yaml"
    config.write_text(
        """\
schema_version: embodichain.task-engine-defaults/v1
workflow:
  max_parallel_workers: 3
  max_scene_attempts: 4
  max_action_attempts: 5
planning:
  candidate_count: 7
  planning_mode: offline
  max_episodes: 2
  max_episode_steps: 5000
execution:
  num_envs: 6
  success_policy: at_least
  min_successful_envs: 2
""",
        encoding="utf-8",
    )

    workflow, planning, execution = load_task_engine_config(config)

    assert workflow.max_parallel_workers == 3
    assert workflow.max_scene_attempts == 4
    assert workflow.max_action_attempts == 5
    assert planning.candidate_count == 7
    assert planning.max_episodes == 2
    assert planning.max_episode_steps == 5000
    assert execution.num_envs == 6
    assert execution.required_successes == 2


def test_execution_configuration_validates_success_policy() -> None:
    assert TaskEngineExecutionCfg().num_envs == 1
    assert (
        TaskEngineExecutionCfg(
            num_envs=4,
            success_policy="at_least",
            min_successful_envs=2,
        ).required_successes
        == 2
    )
    with pytest.raises(ValueError, match="success_policy=all"):
        TaskEngineExecutionCfg(
            num_envs=4,
            success_policy="all",
            min_successful_envs=1,
        )


def test_planning_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="candidate_count"):
        TaskEnginePlanningCfg(candidate_count=0)
    with pytest.raises(ValueError, match="planning_mode"):
        TaskEnginePlanningCfg(planning_mode="unsupported")
    with pytest.raises(TypeError):
        TaskEnginePlanningCfg(gripper_model="unsupported")
    with pytest.raises(TypeError):
        TaskEnginePlanningCfg(ik_solver="unsupported")
    with pytest.raises(TypeError):
        TaskEnginePlanningCfg(planner={"mode": "toppra", "dynamic_collision": True})
