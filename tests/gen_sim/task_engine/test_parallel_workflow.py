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

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from threading import Barrier

import pytest

from embodichain.gen_sim.scene_engine.errors import SceneServiceError
from embodichain.gen_sim.task_engine.config import (
    TaskEngineExecutionCfg,
    TaskEnginePlanningCfg,
    TaskEngineWorkflowCfg,
)
from embodichain.gen_sim.task_engine.orchestration.scene_adapter import (
    CandidateSelection,
    SceneAdapterProtocolError,
)
from embodichain.gen_sim.task_engine.orchestration.scene_source import SceneSourceRef
from embodichain.gen_sim.task_engine.scene_backend import SceneAnalysis, SceneRevision
from embodichain.gen_sim.task_engine.workflow import (
    SubprocessActionExecutor,
    TaskEngineWorkflow,
    _environment_successes,
    _run_streaming_process,
)
from embodichain.gen_sim.task_engine.workflow_contracts import TASK_RUN_REQUEST_SCHEMA


def _candidate_set() -> dict:
    candidate = {
        "candidate_id": "candidate_01",
        "draft": {
            "task_id": "place_can",
            "instruction": "Place the can on the table.",
            "steps": [
                {
                    "id": "place",
                    "task_type": "E1",
                    "object": {
                        "kind": "scene_ref",
                        "step_id": "",
                        "reference": "the can",
                        "quantifier": "one",
                        "count": 0,
                    },
                    "target": {
                        "kind": "scene_ref",
                        "step_id": "",
                        "reference": "the table",
                        "quantifier": "one",
                        "count": 0,
                    },
                    "depends_on": [],
                }
            ],
        },
    }
    return {
        "task_id": "place_can",
        "instruction": "Place the can on the table.",
        "candidates": [candidate],
    }


def _selection(candidate_set: Mapping[str, object]) -> CandidateSelection:
    candidate = candidate_set["candidates"][0]
    return CandidateSelection(
        scene_manifest={},
        role_bindings={},
        binding_report={
            "status": "bound",
            "selection_reason": "test",
            "candidates": [{"candidate_id": "candidate_01", "status": "resolved"}],
        },
        selected_candidate=candidate,
        candidate_bindings={"candidate_01": {}},
    )


def _request(tmp_path: Path, *, existing: bool = False, edit: bool = False) -> dict:
    return {
        "schema_version": TASK_RUN_REQUEST_SCHEMA,
        "task_id": "place_can",
        "task_instruction": "Place the can on the table.",
        "image_path": None if existing else str(tmp_path / "input.png"),
        "gym_project": str(tmp_path / "project") if existing else None,
        "scene_edit_prompt": "Move the can left." if edit else None,
        "output_dir": str(tmp_path / "run"),
    }


class _TaskAgent:
    def __init__(self, candidates: dict, barrier: Barrier | None = None) -> None:
        self.candidates = candidates
        self.barrier = barrier

    def generate(self, *_args, **_kwargs) -> dict:
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        return self.candidates


class _SceneBackend:
    def __init__(
        self,
        selection: CandidateSelection,
        *,
        input_kind: str = "image",
        input_barrier: Barrier | None = None,
        materialize_barrier: Barrier | None = None,
        materialize_failures: int = 0,
        selection_error: Exception | None = None,
    ) -> None:
        self.selection = selection
        self.input_kind = input_kind
        self.input_barrier = input_barrier
        self.materialize_barrier = materialize_barrier
        self.materialize_failures = materialize_failures
        self.selection_error = selection_error
        self.seeds: list[int] = []

    def analyze(self, request, output_root) -> SceneAnalysis:
        if self.input_barrier is not None:
            self.input_barrier.wait(timeout=2)
        return SceneAnalysis(
            input_kind=self.input_kind,
            source=Path(request["image_path"] or request["gym_project"]),
            blueprint=None,
            source_fingerprint=None,
        )

    def select(self, *_args, **_kwargs) -> CandidateSelection:
        if self.selection_error is not None:
            raise self.selection_error
        return self.selection

    def materialize(
        self, _analysis, _request, output_root, *, seed: int
    ) -> SceneRevision:
        if self.materialize_barrier is not None:
            self.materialize_barrier.wait(timeout=2)
        root = Path(output_root)
        root.mkdir(parents=True)
        self.seeds.append(seed)
        if len(self.seeds) <= self.materialize_failures:
            raise SceneServiceError("scene service failed")
        source = root / "scene_config.json"
        source.write_text("{}\n", encoding="utf-8")
        return SceneRevision(
            source=source,
            output_root=root,
            revision_id="0" * 64,
            seed=seed,
            edit_plan=None,
            source_fingerprint=None,
        )

    def inspect(self, revision, output_path):
        value = {
            "schema_version": "embodichain.final-scene-inspection/v1",
            "scene_revision_id": revision.revision_id,
            "source_config_path": revision.source.as_posix(),
            "contact_tolerance_m": 0.03,
            "objects": [],
        }
        path = Path(output_path)
        path.write_text(json.dumps(value), encoding="utf-8")
        return value


class _Coordinator:
    def __init__(
        self,
        statuses: list[str],
        *,
        infeasible_remediation: str = "scene_remediable",
    ) -> None:
        self.statuses = list(statuses)
        self.infeasible_remediation = infeasible_remediation
        self.calls = 0
        self.kwargs: list[dict] = []
        self.sources: list[object] = []

    def prepare(self, _task_id, _instruction, _source, output_dir, **_kwargs):
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        self.kwargs.append(dict(_kwargs))
        self.sources.append(_source)
        root = Path(output_dir)
        root.mkdir(parents=True)
        for name in (
            "conservative_scene_graph.json",
            "semantic_task_graph.json",
            "task_program_deployment.yaml",
        ):
            (root / name).write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            status=status,
            output_dir=root,
            planning_attempts=(),
            feasibility_report=(
                {"remediation_class": self.infeasible_remediation}
                if status == "infeasible"
                else None
            ),
            selected_candidate_id="candidate_01" if status == "bound" else None,
        )


class _FailingCoordinator:
    def prepare(self, *_args, **_kwargs):
        raise RuntimeError("grounding service unavailable")


class _RebindingCoordinator(_Coordinator):
    def __init__(self, final_candidate: Mapping[str, object]) -> None:
        super().__init__(["bound"])
        self.final_candidate = final_candidate

    def prepare(self, *args, **kwargs):
        result = super().prepare(*args, **kwargs)
        result.selected_candidate_id = str(self.final_candidate["candidate_id"])
        result.unbound_action_plan = {
            "schema_version": "semantic_task_plan_draft/v1",
            "task_id": str(self.final_candidate["draft"]["task_id"]),
            "candidate_id": str(self.final_candidate["candidate_id"]),
            "steps": [
                {
                    "id": str(step["id"]),
                    "task_type": str(step["task_type"]),
                    "depends_on": list(step["depends_on"]),
                }
                for step in self.final_candidate["draft"]["steps"]
            ],
        }
        return result


class _InvalidSceneBackend(_SceneBackend):
    def materialize(self, *_args, **kwargs):
        self.seeds.append(int(kwargs["seed"]))
        raise ValueError("invalid deterministic scene input")


class _Executor:
    def __init__(
        self,
        successes: list[list[bool]],
        *,
        expected_dataset_saving: bool = False,
        expected_open_window: bool = False,
    ) -> None:
        self.successes = successes
        self.expected_dataset_saving = expected_dataset_saving
        self.expected_open_window = expected_open_window
        self.calls = 0

    def __call__(
        self,
        _bundle,
        _output_root,
        *,
        seed: int,
        num_envs: int,
        dataset_saving: bool = False,
        failure_policy: str = "stop",
        open_window: bool = False,
    ):
        values = self.successes[min(self.calls, len(self.successes) - 1)]
        self.calls += 1
        assert len(values) == num_envs
        assert dataset_saving is self.expected_dataset_saving
        assert open_window is self.expected_open_window
        assert failure_policy == "stop"
        return {
            "status": "succeeded" if all(values) else "failed",
            "seed": seed,
            "environments": [
                {"env_id": str(index), "success": success}
                for index, success in enumerate(values)
            ],
        }


def test_parallel_workflow_propagates_open_window(tmp_path: Path) -> None:
    candidates = _candidate_set()
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=_SceneBackend(_selection(candidates)),
        coordinator=_Coordinator(["bound"]),
        action_executor=_Executor(
            [[True]],
            expected_open_window=True,
        ),
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(),
        planning_cfg=TaskEnginePlanningCfg(),
        execution_cfg=TaskEngineExecutionCfg(num_envs=1),
        open_window=True,
    )

    assert result.succeeded


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("edit", [False, True])
def test_parallel_workflow_supports_all_four_scene_inputs(
    tmp_path: Path,
    *,
    existing: bool,
    edit: bool,
) -> None:
    candidates = _candidate_set()
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=_SceneBackend(
            _selection(candidates),
            input_kind="gym_project" if existing else "image",
        ),
        coordinator=_Coordinator(["bound"]),
        action_executor=_Executor(
            [[True, False, False, False]],
            expected_dataset_saving=True,
        ),
    )

    result = workflow.run(
        _request(tmp_path, existing=existing, edit=edit),
        workflow_cfg=TaskEngineWorkflowCfg(),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
        dataset_saving=True,
    )

    assert result.succeeded


def test_candidate_selection_programming_error_is_internal_failure(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    scene = _SceneBackend(
        _selection(candidates),
        selection_error=AttributeError("stale cross-engine field"),
    )
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=scene,
        coordinator=_Coordinator(["bound"]),
        action_executor=_Executor([[True]]),
    )

    result = workflow.run(_request(tmp_path))

    assert result.status == "failed"
    assert result.failure_class == "internal_error"
    state = json.loads(
        (result.output_dir / "workflow_state.json").read_text(encoding="utf-8")
    )
    assert state["events"][-1]["reason"] == "stale cross-engine field"


def test_candidate_selection_protocol_error_is_not_input_conflict(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    scene = _SceneBackend(
        _selection(candidates),
        selection_error=SceneAdapterProtocolError("invalid grounding response"),
    )
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=scene,
        coordinator=_Coordinator(["bound"]),
        action_executor=_Executor([[True]]),
    )

    result = workflow.run(_request(tmp_path))

    assert result.status == "failed"
    assert result.failure_class == "candidate_selection"


def test_parallel_workflow_preserves_requested_robot_profile(tmp_path: Path) -> None:
    candidates = _candidate_set()
    coordinator = _Coordinator(["bound"])
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_adapter=SimpleNamespace(robot_profile="ur10"),
        scene_backend=_SceneBackend(_selection(candidates)),
        coordinator=coordinator,
        action_executor=_Executor([[True]]),
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(),
        execution_cfg=TaskEngineExecutionCfg(num_envs=1),
    )

    assert result.succeeded
    assert isinstance(coordinator.sources[0], SceneSourceRef)
    assert coordinator.sources[0].robot_profile == "ur10"


def test_parallel_workflow_accepts_one_success_and_publishes_all_graphs(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    input_barrier = Barrier(2)
    scene = _SceneBackend(
        _selection(candidates),
        input_barrier=input_barrier,
    )
    coordinator = _Coordinator(["bound"])
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates, input_barrier),
        scene_backend=scene,
        coordinator=coordinator,
        action_executor=_Executor([[False, True, False, False]]),
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(),
        planning_cfg=TaskEnginePlanningCfg(
            candidate_count=3,
            planning_mode="offline",
            max_episodes=1,
            max_episode_steps=6000,
        ),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
        base_seed=11,
        run_id="20260820_072436",
    )

    assert result.succeeded
    assert scene.seeds == [11]
    assert result.final_bundle is not None
    assert (result.final_bundle / "conservative_scene_graph.json").is_file()
    assert (result.final_bundle / "semantic_task_graph.json").is_file()
    assert (result.final_bundle / "task_program_deployment.yaml").is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "20260820_072436"
    assert manifest["configuration"]["planning"] == {
        "candidate_count": 3,
        "planning_mode": "offline",
        "max_episodes": 1,
        "max_episode_steps": 6000,
    }
    assert manifest["configuration"]["execution"]["dataset_saving"] is False
    assert coordinator.kwargs[0]["max_episode_steps"] == 6000
    assert coordinator.kwargs[0]["final_inspection"]["scene_revision_id"] == "0" * 64
    assert (
        coordinator.kwargs[0]["unbound_action_plan"]["candidate_id"] == "candidate_01"
    )
    assert manifest["attempts"][0]["action_attempts"][0]["status"] == "succeeded"
    assert manifest["attempts"][0]["final_unbound_action_plan"]["candidate_id"] == (
        "candidate_01"
    )
    assert manifest["attempts"][0]["unbound_transition"]["changed"] is False
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    succeeded = [
        event["stage"] for event in state["events"] if event["to"] == "succeeded"
    ]
    assert (
        succeeded.index("scene_finalization")
        < succeeded.index("final_inspection")
        < succeeded.index("final_binding")
    )


def test_prepare_only_publishes_bundle_without_action_execution(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()

    def fail_execution(*_args, **_kwargs):
        pytest.fail("prepare-only workflow must not execute Action Engine")

    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=_SceneBackend(_selection(candidates)),
        coordinator=_Coordinator(["bound"]),
        action_executor=fail_execution,
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(),
        execution_cfg=TaskEngineExecutionCfg(),
        execute=False,
    )

    assert result.status == "prepared"
    assert result.final_bundle is not None
    assert result.final_bundle.is_dir()


def test_final_candidate_rebinding_updates_attempt_unbound_audit(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    final_candidate = deepcopy(candidates["candidates"][0])
    final_candidate["candidate_id"] = "candidate_02"
    candidates["candidates"].append(final_candidate)
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=_SceneBackend(_selection(candidates)),
        coordinator=_RebindingCoordinator(final_candidate),
        action_executor=_Executor([[True, False, False, False]]),
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    attempt = manifest["attempts"][0]
    assert attempt["unbound_action_plan"]["candidate_id"] == "candidate_01"
    assert attempt["final_unbound_action_plan"]["candidate_id"] == "candidate_02"
    assert attempt["unbound_transition"]["changed"] is True


@pytest.mark.parametrize(
    ("dataset_saving", "expects_filter"),
    [(False, True), (True, False)],
)
@pytest.mark.parametrize(
    ("open_window", "expects_headless"),
    [(False, True), (True, False)],
)
def test_subprocess_executor_controls_launch_options_and_copies_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_saving: bool,
    expects_filter: bool,
    open_window: bool,
    expects_headless: bool,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    trajectory = tmp_path / "trajectory-source"
    trajectory.mkdir()
    (trajectory / "episode.json").write_text("{}\n", encoding="utf-8")
    grasp_image = trajectory / "env_0000" / "grasp_poses" / "task_01.png"
    grasp_image.parent.mkdir(parents=True)
    grasp_image.write_bytes(b"grasp-pose-png")
    captured = {}

    def fake_run(command, log_path):
        captured["command"] = command
        captured["log_path"] = Path(log_path)
        Path(log_path).write_text("child output\n", encoding="utf-8")
        report = {
            "schema_version": "task_program_execution_report/v1",
            "status": "succeeded",
            "task_id": "place_can",
            "semantic_call_count": 1,
            "integration_fingerprint": "0" * 64,
            "record_dir": trajectory.as_posix(),
            "environments": [
                {
                    "env_id": index,
                    "success": True,
                    "terminal_reason": "success",
                    "semantic_success": {},
                }
                for index in range(4)
            ],
            "runtime_result": {},
            "failure": None,
        }
        (Path(log_path).parent / "execution_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "embodichain.gen_sim.task_engine.workflow._run_streaming_process",
        fake_run,
    )
    attempt = tmp_path / "attempt"

    report = SubprocessActionExecutor()(
        bundle,
        attempt,
        seed=7,
        num_envs=4,
        dataset_saving=dataset_saving,
        failure_policy="continue",
        open_window=open_window,
    )

    assert report["status"] == "succeeded"
    assert captured["command"][1:5] == [
        "-u",
        "-m",
        "embodichain.gen_sim.task_engine._bundle_runner",
        "--bundle",
    ]
    assert " prepare" not in " ".join(captured["command"])
    assert " workflow" not in " ".join(captured["command"])
    assert ("--filter_dataset_saving" in captured["command"]) is expects_filter
    assert ("--headless" in captured["command"]) is expects_headless
    if expects_headless:
        renderer_index = captured["command"].index("--renderer")
        assert captured["command"][renderer_index : renderer_index + 2] == [
            "--renderer",
            "fast-rt",
        ]
    else:
        assert "--renderer" not in captured["command"]
    assert "--show-grasp-poses" not in captured["command"]
    assert captured["command"][-2:] == ["--failure-policy", "continue"]
    assert captured["log_path"] == attempt / "action.log"
    assert (attempt / "action.log").read_text(encoding="utf-8") == "child output\n"
    assert (attempt / "trajectory" / "episode.json").is_file()
    assert (
        attempt / "trajectory" / "env_0000" / "grasp_poses" / "task_01.png"
    ).read_bytes() == b"grasp-pose-png"
    process = json.loads((attempt / "process.json").read_text(encoding="utf-8"))
    assert process["combined_log"] == "action.log"
    assert process["stdout"] == "ok"
    assert process["stderr"] == ""


def test_streaming_process_tees_combined_binary_output(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "action.log"
    script = (
        "import os; "
        "os.write(1, b'stdout\\x00'); "
        "os.write(2, b'stderr\\rprogress\\n'); "
        "raise SystemExit(7)"
    )

    completed = _run_streaming_process(
        [sys.executable, "-c", script],
        log_path,
    )

    expected = b"stdout\x00stderr\rprogress\n"
    assert completed.returncode == 7
    assert completed.stdout.encode("utf-8") == expected
    assert completed.stderr == ""
    assert log_path.read_bytes() == expected
    terminal = capfd.readouterr().out
    assert "stdout\x00" in terminal
    assert "stderr\rprogress" in terminal


def test_scene_remediation_changes_seed_before_action_execution(tmp_path: Path) -> None:
    candidates = _candidate_set()
    scene = _SceneBackend(_selection(candidates))
    coordinator = _Coordinator(["infeasible", "bound"])
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=scene,
        coordinator=coordinator,
        action_executor=_Executor([[True, False, False, False]]),
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(max_scene_attempts=2),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
        base_seed=20,
    )

    assert result.succeeded
    assert scene.seeds == [20, 21]
    assert coordinator.calls == 2
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert [item["status"] for item in manifest["attempts"]] == [
        "preparation_failed",
        "succeeded",
    ]


def test_input_conflict_feasibility_does_not_regenerate_scene(tmp_path: Path) -> None:
    candidates = _candidate_set()
    scene = _SceneBackend(_selection(candidates))
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=scene,
        coordinator=_Coordinator(
            ["infeasible", "bound"],
            infeasible_remediation="input_conflict",
        ),
        action_executor=_Executor([[True, False, False, False]]),
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(max_scene_attempts=2),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
    )

    assert result.failure_class == "input_conflict"
    assert scene.seeds == [0]


def test_scene_service_retry_keeps_completed_unbound_plan(tmp_path: Path) -> None:
    candidates = _candidate_set()
    scene = _SceneBackend(_selection(candidates), materialize_failures=1)
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=scene,
        coordinator=_Coordinator(["bound"]),
        action_executor=_Executor([[True, False, False, False]]),
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(max_scene_attempts=2),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
        base_seed=30,
    )

    assert result.succeeded
    assert scene.seeds == [30, 31]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["attempts"][0]["unbound_action_plan"] is not None


def test_nonremediable_scene_error_does_not_change_scene_attempt_seed(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    scene = _InvalidSceneBackend(_selection(candidates))
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=scene,
        coordinator=_Coordinator(["bound"]),
        action_executor=_Executor([[True, False, False, False]]),
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(max_scene_attempts=3),
        execution_cfg=TaskEngineExecutionCfg(),
        base_seed=9,
    )

    assert not result.succeeded
    assert scene.seeds == [9]


def test_execution_acceptance_requires_every_success_spec_term() -> None:
    report = {
        "environments": [
            {
                "success": True,
                "semantic_success": {"step_01": True, "step_02": False},
            },
            {
                "success": True,
                "semantic_success": {"step_01": True, "step_02": True},
            },
        ]
    }

    assert _environment_successes(
        report,
        required_semantic_steps=("step_01", "step_02"),
    ) == [False, True]


def test_unsupported_semantic_draft_retains_completed_parallel_scene(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    candidates["candidates"][0]["draft"]["steps"][0]["task_type"] = "E2"
    scene = _SceneBackend(_selection(candidates))
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=scene,
        coordinator=_Coordinator(["bound"]),
        action_executor=_Executor([[True, False, False, False]]),
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
    )

    assert not result.succeeded
    assert result.failure_class == "action_capability"
    assert scene.seeds == [0]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["attempts"][0]["scene_revision"] is not None
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    assert state["stages"]["scene_finalization"] == "succeeded"
    assert state["stages"]["unbound_action"] == "failed"


def test_preparation_exception_is_published_as_audited_failure(tmp_path: Path) -> None:
    candidates = _candidate_set()
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=_SceneBackend(_selection(candidates)),
        coordinator=_FailingCoordinator(),
        action_executor=_Executor([[True, False, False, False]]),
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
    )

    assert not result.succeeded
    assert result.failure_class == "preparation_error"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["attempts"][0]["status"] == "preparation_error"
    assert manifest["attempts"][0]["error"]["type"] == "RuntimeError"


def test_explicit_edit_may_materialize_initially_missing_reference(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    unresolved = CandidateSelection(
        scene_manifest={},
        role_bindings={},
        binding_report={
            "status": "unsatisfied",
            "selection_reason": "the can is not visible before the explicit edit",
            "candidates": [{"candidate_id": "candidate_01", "status": "unsatisfied"}],
        },
        selected_candidate=None,
        candidate_bindings={"candidate_01": {}},
    )
    scene = _SceneBackend(unresolved, input_kind="gym_project")
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=scene,
        coordinator=_Coordinator(["bound"]),
        action_executor=_Executor([[True, False, False, False]]),
    )

    result = workflow.run(
        _request(tmp_path, existing=True, edit=True),
        workflow_cfg=TaskEngineWorkflowCfg(),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
    )

    assert result.succeeded
    provisional = json.loads(
        (result.output_dir / "provisional_candidate.json").read_text(encoding="utf-8")
    )
    assert provisional == {
        "binding_status": "unsatisfied",
        "candidate_id": "candidate_01",
        "reason": "explicit_scene_edit_may_materialize_missing_reference",
    }


def test_action_failure_retries_action_only_and_retains_attempts(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    scene = _SceneBackend(_selection(candidates))
    executor = _Executor([[False, False, False, False]])
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=scene,
        coordinator=_Coordinator(["bound"]),
        action_executor=executor,
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(max_action_attempts=3),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
    )

    assert not result.succeeded
    assert result.failure_class == "action_execution"
    assert scene.seeds == [0]
    assert executor.calls == 3
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["attempts"][0]["action_attempts"]) == 3


def test_action_retry_stops_after_first_success(tmp_path: Path) -> None:
    candidates = _candidate_set()
    executor = _Executor(
        [
            [False, False, False, False],
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=_SceneBackend(_selection(candidates)),
        coordinator=_Coordinator(["bound"]),
        action_executor=executor,
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(max_action_attempts=3),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
    )

    assert result.succeeded
    assert executor.calls == 2
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert [item["status"] for item in manifest["attempts"][0]["action_attempts"]] == [
        "failed",
        "succeeded",
    ]


def test_existing_edit_binding_conflict_does_not_invent_scene_repair(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    scene = _SceneBackend(_selection(candidates), input_kind="gym_project")
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=scene,
        coordinator=_Coordinator(["unsatisfied"]),
        action_executor=_Executor([[True, False, False, False]]),
    )

    result = workflow.run(
        _request(tmp_path, existing=True, edit=True),
        workflow_cfg=TaskEngineWorkflowCfg(max_scene_attempts=2),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
    )

    assert result.status == "input_conflict"
    assert result.failure_class == "input_conflict"
    assert scene.seeds == [0]


def test_image_binding_conflict_does_not_regenerate_scene(tmp_path: Path) -> None:
    candidates = _candidate_set()
    scene = _SceneBackend(_selection(candidates))
    workflow = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=scene,
        coordinator=_Coordinator(["unsatisfied"]),
        action_executor=_Executor([[True, False, False, False]]),
    )

    result = workflow.run(
        _request(tmp_path),
        workflow_cfg=TaskEngineWorkflowCfg(max_scene_attempts=2),
        execution_cfg=TaskEngineExecutionCfg(num_envs=4),
    )

    assert result.status == "input_conflict"
    assert result.failure_class == "input_conflict"
    assert scene.seeds == [0]
