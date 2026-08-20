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

"""Parallel Task Engine workflow with bounded, fully audited recovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Final

from embodichain.gen_sim.action_engine.agent import ActionAgent
from embodichain.gen_sim.action_engine.unbound import ActionCapabilityError
from embodichain.gen_sim.action_engine.runtime import (
    EXECUTION_REPORT_FILENAME,
    validate_execution_report,
)
from embodichain.gen_sim.scene_engine.errors import SceneServiceError

from .agent import TaskAgent
from .config import (
    TaskEngineExecutionCfg,
    TaskEnginePlanningCfg,
    TaskEngineWorkflowCfg,
    load_task_engine_config,
)
from .contracts import canonical_hash
from .orchestration.artifacts import ArtifactTransaction
from .orchestration.coordinator import PreparationResult, TaskEngineCoordinator
from .orchestration.scene_adapter import CandidateSelection, SceneAdapter
from .scene_backend import (
    SceneAnalysis,
    SceneEngineBackend,
    SceneRemediableError,
    SceneRevision,
)
from .state_machine import (
    TaskEngineState,
    WorkflowStage,
    complete_stage,
    fail_stage,
    initial_state,
    start_stage,
)
from .workflow_contracts import TaskRunRequest, validate_task_run_request

__all__ = [
    "TASK_ENGINE_RUN_MANIFEST_SCHEMA",
    "ActionExecutor",
    "SubprocessActionExecutor",
    "TaskEngineRunResult",
    "TaskEngineWorkflow",
]

TASK_ENGINE_RUN_MANIFEST_SCHEMA: Final = "embodichain.task-engine-run/v1"
ActionExecutor = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class TaskEngineRunResult:
    """Published outcome of one isolated cross-engine workflow run."""

    status: str
    output_dir: Path
    manifest_path: Path
    state_path: Path
    final_bundle: Path | None
    failure_class: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether real simulator execution met the configured policy."""
        return self.status == "succeeded"


class SubprocessActionExecutor:
    """Execute a prepared bundle through Task Engine's private runner."""

    def __call__(
        self,
        bundle: str | Path,
        output_root: str | Path,
        *,
        seed: int,
        num_envs: int,
        dataset_saving: bool = False,
    ) -> Mapping[str, Any]:
        """Run one simulator attempt and preserve its report and trajectory.

        Args:
            bundle: Prepared Action Engine bundle.
            output_root: Fresh directory for this execution attempt.
            seed: Action Engine random seed.
            num_envs: Number of vectorized scene replicas.
            dataset_saving: Whether to enable the Gym project's dataset recorder.

        Returns:
            Validated Action Engine execution report.
        """
        bundle_root = Path(bundle).expanduser().resolve()
        attempt_root = Path(output_root).expanduser().resolve()
        attempt_root.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            "-u",
            "-m",
            "embodichain.gen_sim.task_engine._bundle_runner",
            "--bundle",
            bundle_root.as_posix(),
            "--num_envs",
            str(num_envs),
            "--seed",
            str(seed),
            "--headless",
        ]
        if not dataset_saving:
            command.append("--filter_dataset_saving")
        log_path = attempt_root / "action.log"
        print(
            "[Task Engine] Starting "
            f"{attempt_root.name}: seed={seed}, num_envs={num_envs}, "
            f"dataset_saving={dataset_saving}",
            flush=True,
        )
        completed = _run_streaming_process(command, log_path)
        print(
            f"[Task Engine] Completed {attempt_root.name}: "
            f"returncode={completed.returncode}",
            flush=True,
        )
        report_path = bundle_root / EXECUTION_REPORT_FILENAME
        process_record = {
            "command": command,
            "returncode": completed.returncode,
            "combined_log": log_path.name,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        _write_json(attempt_root / "process.json", process_record)
        if not report_path.is_file():
            raise RuntimeError(
                "Action execution did not publish execution_report.json; "
                f"returncode={completed.returncode}."
            )
        report = validate_execution_report(_read_json(report_path))
        shutil.copy2(report_path, attempt_root / EXECUTION_REPORT_FILENAME)
        trajectory_copy = _copy_trajectory_record(report, attempt_root)
        if report["action_count"] > 0 and trajectory_copy is None:
            raise RuntimeError(
                "Action execution report did not expose a readable trajectory record."
            )
        _write_json(
            attempt_root / "execution_attempt.json",
            {
                "seed": seed,
                "num_envs": num_envs,
                "dataset_saving": dataset_saving,
                "returncode": completed.returncode,
                "trajectory_copy": trajectory_copy,
                "report": report,
            },
        )
        return report


def _run_streaming_process(
    command: list[str],
    log_path: str | Path,
) -> subprocess.CompletedProcess[str]:
    """Run a child while teeing its combined output to the terminal and disk.

    Args:
        command: Argument vector passed directly to the child process.
        log_path: File receiving the exact combined stdout and stderr bytes.

    Returns:
        Completed process metadata with a decoded copy of the combined output.
    """
    resolved_log = Path(log_path).expanduser().resolve()
    resolved_log.parent.mkdir(parents=True, exist_ok=True)
    captured = bytearray()
    process: subprocess.Popen[bytes] | None = None
    try:
        with resolved_log.open("wb") as log_stream:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    break
                captured.extend(chunk)
                log_stream.write(chunk)
                log_stream.flush()
                _write_terminal_chunk(chunk)
            returncode = process.wait()
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    output = captured.decode("utf-8", errors="replace")
    return subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout=output,
        stderr="",
    )


def _write_terminal_chunk(chunk: bytes) -> None:
    """Best-effort write of raw child output to the parent terminal."""
    try:
        stream = getattr(sys.stdout, "buffer", None)
        if stream is not None:
            stream.write(chunk)
            stream.flush()
            return
        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
        sys.stdout.flush()
    except (BrokenPipeError, OSError, ValueError):
        return


class TaskEngineWorkflow:
    """Run Scene and Action work concurrently under Task Engine ownership."""

    def __init__(
        self,
        *,
        task_agent: TaskAgent | None = None,
        scene_adapter: SceneAdapter | None = None,
        action_agent: ActionAgent | None = None,
        coordinator: TaskEngineCoordinator | None = None,
        scene_backend: SceneEngineBackend | None = None,
        action_executor: ActionExecutor | None = None,
    ) -> None:
        self.task_agent = task_agent or TaskAgent()
        self.scene_adapter = scene_adapter or SceneAdapter()
        self.action_agent = action_agent or ActionAgent()
        self.coordinator = coordinator or TaskEngineCoordinator(
            task_agent=self.task_agent,
            scene_adapter=self.scene_adapter,
            action_agent=self.action_agent,
        )
        self.scene_backend = scene_backend or SceneEngineBackend()
        self.action_executor = action_executor or SubprocessActionExecutor()

    def run(
        self,
        request: TaskRunRequest | Mapping[str, Any],
        *,
        workflow_cfg: TaskEngineWorkflowCfg | None = None,
        planning_cfg: TaskEnginePlanningCfg | None = None,
        execution_cfg: TaskEngineExecutionCfg | None = None,
        config_path: str | Path | None = None,
        model: str | None = None,
        vlm_model: str | None = None,
        base_seed: int = 0,
        dataset_saving: bool = False,
        run_id: str | None = None,
        created_at: datetime | None = None,
        overwrite: bool = False,
        execute: bool = True,
    ) -> TaskEngineRunResult:
        """Run all stages and publish success only after simulator acceptance.

        Args:
            request: One of the four image/project plus optional-edit inputs.
            workflow_cfg: Optional retry and concurrency configuration.
            planning_cfg: Optional interpretation and bundle generation defaults.
            execution_cfg: Optional vectorized success policy.
            config_path: YAML used for omitted workflow or execution config.
            model: Optional Task and grounding model override.
            vlm_model: Optional Action Engine VLM override.
            base_seed: First audited scene and action attempt seed.
            dataset_saving: Whether Action attempts may initialize dataset recording.
            run_id: Optional externally allocated run identifier.
            created_at: Optional timezone-aware run creation timestamp.
            overwrite: Whether to atomically replace an existing run directory.
            execute: Whether to execute the prepared bundle in the simulator.

        Returns:
            Published run status, manifest, state audit, and final bundle path.
        """
        normalized = validate_task_run_request(request)
        if not isinstance(dataset_saving, bool):
            raise TypeError("dataset_saving must be a boolean.")
        if workflow_cfg is None or planning_cfg is None or execution_cfg is None:
            loaded_workflow, loaded_planning, loaded_execution = (
                load_task_engine_config(config_path)
            )
            workflow_cfg = workflow_cfg or loaded_workflow
            planning_cfg = planning_cfg or loaded_planning
            execution_cfg = execution_cfg or loaded_execution
        effective_candidate_count = planning_cfg.candidate_count
        effective_run_id = str(run_id or Path(normalized["output_dir"]).name).strip()
        if not effective_run_id or Path(effective_run_id).name != effective_run_id:
            raise ValueError("run_id must be one non-empty path component.")
        effective_created_at = created_at or datetime.now().astimezone()
        if (
            effective_created_at.tzinfo is None
            or effective_created_at.utcoffset() is None
        ):
            raise ValueError("created_at must include a timezone.")
        run_metadata = {
            "run_id": effective_run_id,
            "created_at": effective_created_at.isoformat(),
            "dataset_saving": bool(dataset_saving),
        }
        state = initial_state(normalized)
        attempts: list[dict[str, Any]] = []
        output_dir = Path(normalized["output_dir"])

        with ArtifactTransaction(output_dir, overwrite=overwrite) as transaction:
            staging = transaction.staging_dir
            assert staging is not None
            analysis_root = staging / "scene_analysis"
            state = start_stage(state, WorkflowStage.TASK_CANDIDATES)
            state = start_stage(state, WorkflowStage.SCENE_PREPARATION)
            with ThreadPoolExecutor(
                max_workers=workflow_cfg.max_parallel_workers,
                thread_name_prefix="task-engine-input",
            ) as executor:
                candidate_future = executor.submit(
                    self.task_agent.generate,
                    normalized["task_id"],
                    normalized["task_instruction"],
                    model,
                    effective_candidate_count,
                )
                analysis_future = executor.submit(
                    self.scene_backend.analyze,
                    normalized,
                    analysis_root,
                )
                try:
                    candidate_set = candidate_future.result()
                except Exception as exc:
                    analysis_future.cancel()
                    state = fail_stage(
                        state,
                        WorkflowStage.TASK_CANDIDATES,
                        reason=str(exc),
                    )
                    return self._publish(
                        transaction,
                        staging,
                        normalized,
                        workflow_cfg,
                        planning_cfg,
                        execution_cfg,
                        run_metadata,
                        state,
                        attempts,
                        status="failed",
                        failure_class="task_generation",
                    )
                state = complete_stage(state, WorkflowStage.TASK_CANDIDATES)
                try:
                    analysis = analysis_future.result()
                except Exception as exc:
                    state = fail_stage(
                        state,
                        WorkflowStage.SCENE_PREPARATION,
                        reason=str(exc),
                    )
                    return self._publish(
                        transaction,
                        staging,
                        normalized,
                        workflow_cfg,
                        planning_cfg,
                        execution_cfg,
                        run_metadata,
                        state,
                        attempts,
                        status="failed",
                        failure_class="scene_analysis",
                    )
            state = complete_stage(state, WorkflowStage.SCENE_PREPARATION)

            state = start_stage(state, WorkflowStage.CANDIDATE_SELECTION)
            try:
                selection = self.scene_backend.select(
                    analysis,
                    candidate_set,
                    self.scene_adapter,
                    force_most_likely=True,
                )
            except Exception as exc:
                state = fail_stage(
                    state,
                    WorkflowStage.CANDIDATE_SELECTION,
                    reason=str(exc),
                )
                return self._publish(
                    transaction,
                    staging,
                    normalized,
                    workflow_cfg,
                    planning_cfg,
                    execution_cfg,
                    run_metadata,
                    state,
                    attempts,
                    status="input_conflict",
                    failure_class="candidate_selection",
                )
            _write_json(
                staging / "initial_binding_report.json", selection.binding_report
            )
            if selection.selected_candidate is None:
                if normalized["scene_edit_prompt"] is None:
                    state = fail_stage(
                        state,
                        WorkflowStage.CANDIDATE_SELECTION,
                        reason=str(selection.binding_report["selection_reason"]),
                    )
                    return self._publish(
                        transaction,
                        staging,
                        normalized,
                        workflow_cfg,
                        planning_cfg,
                        execution_cfg,
                        run_metadata,
                        state,
                        attempts,
                        status="input_conflict",
                        failure_class="unbound_scene_reference",
                    )
                provisional = _highest_vote_candidate(candidate_set)
                selection = replace(
                    selection,
                    selected_candidate=deepcopy(provisional),
                )
                _write_json(
                    staging / "provisional_candidate.json",
                    {
                        "candidate_id": provisional["candidate_id"],
                        "reason": "explicit_scene_edit_may_materialize_missing_reference",
                        "binding_status": selection.binding_report["status"],
                    },
                )
            state = complete_stage(state, WorkflowStage.CANDIDATE_SELECTION)

            state = start_stage(state, WorkflowStage.UNBOUND_ACTION)
            if normalized["scene_edit_prompt"] is not None:
                state = start_stage(state, WorkflowStage.SCENE_EDIT)
            else:
                state = start_stage(state, WorkflowStage.SCENE_FINALIZATION)

            unbound_plan: Mapping[str, Any] | None = None
            unbound_failures: list[dict[str, Any]] = []
            unbound_error: Exception | None = None
            scene_error: Exception | None = None
            inspection_error = False
            preparation_error: Exception | None = None
            preparation: PreparationResult | None = None
            scene_attempt_limit = (
                1
                if analysis.input_kind == "gym_project"
                and normalized["scene_edit_prompt"] is None
                else workflow_cfg.max_scene_attempts
            )
            for scene_index in range(1, scene_attempt_limit + 1):
                inspection_error = False
                scene_seed = int(base_seed) + scene_index - 1
                attempt_root = staging / "attempts" / f"scene_{scene_index:04d}"
                attempt_root.mkdir(parents=True)
                attempt = {
                    "scene_attempt": scene_index,
                    "scene_seed": scene_seed,
                    "status": "running",
                    "scene_revision": None,
                    "final_inspection": None,
                    "unbound_action_plan": None,
                    "final_unbound_action_plan": None,
                    "unbound_transition": None,
                    "unbound_failures": [],
                    "preparation": None,
                    "planning_attempts": [],
                    "action_attempts": [],
                    "parallel_errors": [],
                    "error": None,
                }
                attempts.append(attempt)
                revision: SceneRevision | None = None
                try:
                    if unbound_plan is None:
                        with ThreadPoolExecutor(
                            max_workers=workflow_cfg.max_parallel_workers,
                            thread_name_prefix="task-engine-parallel",
                        ) as executor:
                            scene_future = executor.submit(
                                self.scene_backend.materialize,
                                analysis,
                                normalized,
                                attempt_root / "scene_revision",
                                seed=scene_seed,
                            )
                            draft_future = executor.submit(
                                self._draft_with_fallback,
                                candidate_set,
                                selection,
                            )
                            try:
                                revision = scene_future.result()
                            except Exception as exc:
                                scene_error = exc
                                revision = None
                            try:
                                unbound_plan, unbound_failures = draft_future.result()
                            except Exception as exc:
                                unbound_error = exc
                            if unbound_error is not None:
                                raise unbound_error
                            state = complete_stage(state, WorkflowStage.UNBOUND_ACTION)
                            if scene_error is not None:
                                raise scene_error
                            assert revision is not None
                    else:
                        revision = self.scene_backend.materialize(
                            analysis,
                            normalized,
                            attempt_root / "scene_revision",
                            seed=scene_seed,
                        )
                    scene_error = None
                except Exception as exc:
                    if revision is not None:
                        attempt["scene_revision"] = _revision_record(revision)
                        state = _complete_materialized_scene(
                            state,
                            has_edit=normalized["scene_edit_prompt"] is not None,
                        )
                    if unbound_error is not None:
                        attempt["status"] = "unbound_action_failed"
                        attempt["error"] = _error_record(unbound_error)
                        if scene_error is not None:
                            attempt["parallel_errors"].append(
                                {
                                    "branch": "scene",
                                    **_error_record(scene_error),
                                }
                            )
                        _write_json(attempt_root / "attempt.json", attempt)
                        break
                    scene_error = exc
                    if unbound_plan is not None:
                        attempt["unbound_action_plan"] = deepcopy(dict(unbound_plan))
                        attempt["unbound_failures"] = deepcopy(unbound_failures)
                        _write_json(
                            attempt_root / "unbound_action_plan.json", unbound_plan
                        )
                    attempt["status"] = "scene_failed"
                    attempt["error"] = _error_record(exc)
                    _write_json(attempt_root / "attempt.json", attempt)
                    if (
                        scene_index < scene_attempt_limit
                        and _is_scene_remediable_error(exc)
                    ):
                        continue
                    break

                attempt["scene_revision"] = _revision_record(revision)
                attempt["unbound_action_plan"] = deepcopy(dict(unbound_plan))
                attempt["unbound_failures"] = deepcopy(unbound_failures)
                _write_json(attempt_root / "unbound_action_plan.json", unbound_plan)
                state = _complete_materialized_scene(
                    state,
                    has_edit=normalized["scene_edit_prompt"] is not None,
                )
                if state.stages[WorkflowStage.FINAL_INSPECTION].value == "pending":
                    state = start_stage(state, WorkflowStage.FINAL_INSPECTION)
                try:
                    final_inspection = self.scene_backend.inspect(
                        revision,
                        attempt_root / "final_scene_inspection.json",
                    )
                except Exception as exc:
                    scene_error = exc
                    inspection_error = True
                    attempt["status"] = "scene_inspection_failed"
                    attempt["error"] = _error_record(exc)
                    _write_json(attempt_root / "attempt.json", attempt)
                    if (
                        scene_index < scene_attempt_limit
                        and _is_scene_remediable_error(exc)
                    ):
                        continue
                    break
                attempt["final_inspection"] = deepcopy(dict(final_inspection))
                if state.stages[WorkflowStage.FINAL_INSPECTION].value == "running":
                    state = complete_stage(state, WorkflowStage.FINAL_INSPECTION)

                bundle_root = attempt_root / "bundle"
                try:
                    preparation = self.coordinator.prepare(
                        normalized["task_id"],
                        normalized["task_instruction"],
                        revision.source,
                        bundle_root,
                        model=model,
                        candidate_count=effective_candidate_count,
                        planning_mode=planning_cfg.planning_mode,
                        vlm_model=vlm_model,
                        max_episodes=planning_cfg.max_episodes,
                        max_episode_steps=planning_cfg.max_episode_steps,
                        candidate_set=candidate_set,
                        force_most_likely=True,
                        final_inspection=final_inspection,
                        unbound_action_plan=unbound_plan,
                    )
                except Exception as exc:
                    preparation_error = exc
                    attempt["status"] = "preparation_error"
                    attempt["error"] = _error_record(exc)
                    _write_json(attempt_root / "attempt.json", attempt)
                    break
                attempt["preparation"] = preparation.status
                attempt["planning_attempts"] = deepcopy(
                    list(preparation.planning_attempts)
                )
                if preparation.status == "bound":
                    attempt["status"] = "prepared"
                    _write_json(attempt_root / "attempt.json", attempt)
                    break
                attempt["status"] = "preparation_failed"
                attempt["error"] = {
                    "type": "PreparationFailure",
                    "message": preparation.status,
                }
                _write_json(attempt_root / "attempt.json", attempt)
                if not _scene_remediable(
                    preparation,
                    analysis=analysis,
                    request=normalized,
                ):
                    break

            if preparation is None or preparation.status != "bound":
                failure_class = (
                    "action_capability"
                    if unbound_error is not None
                    or isinstance(preparation_error, ActionCapabilityError)
                    else (
                        "preparation_error"
                        if preparation_error is not None
                        else _preparation_failure_class(
                            preparation,
                            scene_error=scene_error,
                            analysis=analysis,
                            request=normalized,
                        )
                    )
                )
                failed_stage = (
                    WorkflowStage.FINAL_INSPECTION
                    if inspection_error
                    else (
                        WorkflowStage.UNBOUND_ACTION
                        if unbound_error is not None
                        else _failure_stage(failure_class, normalized)
                    )
                )
                if state.stages[failed_stage].value in {
                    "pending",
                    "running",
                    "succeeded",
                }:
                    state = fail_stage(
                        state,
                        failed_stage,
                        reason=(
                            str(unbound_error)
                            if unbound_error is not None
                            else (
                                str(preparation_error)
                                if preparation_error is not None
                                else (
                                    str(scene_error)
                                    if scene_error is not None
                                    else (
                                        preparation.status
                                        if preparation is not None
                                        else failure_class
                                    )
                                )
                            )
                        ),
                    )
                return self._publish(
                    transaction,
                    staging,
                    normalized,
                    workflow_cfg,
                    planning_cfg,
                    execution_cfg,
                    run_metadata,
                    state,
                    attempts,
                    status=(
                        "input_conflict"
                        if failure_class == "input_conflict"
                        else "failed"
                    ),
                    failure_class=failure_class,
                )

            final_candidate_id = preparation.selected_candidate_id
            if not isinstance(final_candidate_id, str) or not final_candidate_id:
                raise ValueError(
                    "A bound preparation must select one non-empty candidate ID."
                )
            selected_attempt = attempts[-1]
            final_unbound = getattr(preparation, "unbound_action_plan", None)
            if (
                final_unbound is None
                and final_candidate_id != unbound_plan["candidate_id"]
            ):
                final_candidate = next(
                    item
                    for item in candidate_set["candidates"]
                    if item["candidate_id"] == final_candidate_id
                )
                final_unbound = self.action_agent.draft(final_candidate)
            elif final_unbound is None:
                final_unbound = unbound_plan
            if str(final_unbound.get("candidate_id")) != final_candidate_id:
                raise ValueError(
                    "Final UnboundActionPlan candidate does not match preparation."
                )
            selected_attempt["final_unbound_action_plan"] = deepcopy(
                dict(final_unbound)
            )
            selected_attempt["unbound_transition"] = {
                "initial_candidate_id": str(unbound_plan["candidate_id"]),
                "initial_hash": canonical_hash(unbound_plan),
                "final_candidate_id": final_candidate_id,
                "final_hash": canonical_hash(final_unbound),
                "changed": final_unbound != unbound_plan,
            }
            _write_json(
                preparation.output_dir.parent / "final_unbound_action_plan.json",
                final_unbound,
            )
            _write_json(
                preparation.output_dir.parent / "attempt.json",
                selected_attempt,
            )

            for stage in (
                WorkflowStage.FINAL_BINDING,
                WorkflowStage.STATIC_FEASIBILITY,
                WorkflowStage.GROUNDED_ACTION,
            ):
                state = start_stage(state, stage)
                state = complete_stage(state, stage)
            if not execute:
                final_root = staging / "final"
                final_bundle = final_root / "bundle"
                final_root.mkdir()
                shutil.copytree(preparation.output_dir, final_bundle)
                selected_attempt["status"] = "prepared"
                _write_json(
                    preparation.output_dir.parent / "attempt.json",
                    selected_attempt,
                )
                _write_json(
                    final_root / "selection.json",
                    {
                        "scene_attempt": selected_attempt["scene_attempt"],
                        "candidate_id": final_candidate_id,
                        "action_attempt": None,
                        "execution_report": None,
                    },
                )
                return self._publish(
                    transaction,
                    staging,
                    normalized,
                    workflow_cfg,
                    planning_cfg,
                    execution_cfg,
                    run_metadata,
                    state,
                    attempts,
                    status="prepared",
                    failure_class=None,
                    final_bundle=final_bundle,
                )
            state = start_stage(state, WorkflowStage.EXECUTION)

            successful_report: Mapping[str, Any] | None = None
            successful_action_root: Path | None = None
            success_terms = _bundle_success_terms(preparation.output_dir)
            for action_index in range(1, workflow_cfg.max_action_attempts + 1):
                action_seed = int(base_seed) + action_index - 1
                action_root = (
                    preparation.output_dir.parent
                    / "action_attempts"
                    / f"action_{action_index:04d}"
                )
                action_record: dict[str, Any] = {
                    "action_attempt": action_index,
                    "seed": action_seed,
                    "status": "running",
                    "successful_environments": 0,
                    "required_successes": execution_cfg.required_successes,
                    "error": None,
                }
                try:
                    report = self.action_executor(
                        preparation.output_dir,
                        action_root,
                        seed=action_seed,
                        num_envs=execution_cfg.num_envs,
                        dataset_saving=bool(dataset_saving),
                    )
                    successes = _environment_successes(
                        report,
                        required_semantic_steps=success_terms,
                    )
                    if len(successes) != execution_cfg.num_envs:
                        raise ValueError(
                            "Execution report environment count does not match "
                            "TaskEngineExecutionCfg.num_envs."
                        )
                    action_record["successful_environments"] = sum(successes)
                    accepted = (
                        str(report.get("status")) not in {"rejected", "aborted"}
                        and sum(successes) >= execution_cfg.required_successes
                    )
                    action_record["status"] = "succeeded" if accepted else "failed"
                    _write_json(action_root / "task_engine_attempt.json", action_record)
                    selected_attempt["action_attempts"].append(deepcopy(action_record))
                    if accepted:
                        successful_report = deepcopy(dict(report))
                        successful_action_root = action_root
                        break
                except Exception as exc:
                    action_record["status"] = "failed"
                    action_record["error"] = _error_record(exc)
                    action_root.mkdir(parents=True, exist_ok=True)
                    _write_json(action_root / "task_engine_attempt.json", action_record)
                    selected_attempt["action_attempts"].append(deepcopy(action_record))
                _write_json(
                    preparation.output_dir.parent / "attempt.json", selected_attempt
                )

            if successful_report is None:
                selected_attempt["status"] = "execution_failed"
                _write_json(
                    preparation.output_dir.parent / "attempt.json", selected_attempt
                )
                state = fail_stage(
                    state,
                    WorkflowStage.EXECUTION,
                    reason="All bounded Action Engine execution attempts failed.",
                )
                return self._publish(
                    transaction,
                    staging,
                    normalized,
                    workflow_cfg,
                    planning_cfg,
                    execution_cfg,
                    run_metadata,
                    state,
                    attempts,
                    status="failed",
                    failure_class="action_execution",
                )

            selected_attempt["status"] = "succeeded"
            _write_json(
                preparation.output_dir.parent / "attempt.json", selected_attempt
            )
            state = complete_stage(state, WorkflowStage.EXECUTION)
            final_root = staging / "final"
            final_bundle = final_root / "bundle"
            final_root.mkdir()
            shutil.copytree(preparation.output_dir, final_bundle)
            _write_json(
                final_root / "selection.json",
                {
                    "scene_attempt": selected_attempt["scene_attempt"],
                    "candidate_id": final_candidate_id,
                    "action_attempt": int(successful_action_root.name.split("_")[-1]),
                    "execution_report": successful_report,
                    "success_spec_steps": list(success_terms),
                },
            )
            return self._publish(
                transaction,
                staging,
                normalized,
                workflow_cfg,
                planning_cfg,
                execution_cfg,
                run_metadata,
                state,
                attempts,
                status="succeeded",
                failure_class=None,
                final_bundle=final_bundle,
            )

    def _draft_with_fallback(
        self,
        candidate_set: Mapping[str, Any],
        selection: CandidateSelection,
    ) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
        selected_id = selection.selected_candidate_id
        resolved_ids = {
            str(item["candidate_id"])
            for item in selection.binding_report["candidates"]
            if item["status"] == "resolved"
        }
        ordered = [selected_id] + [
            str(item["candidate_id"])
            for item in candidate_set["candidates"]
            if item["candidate_id"] != selected_id
            and item["candidate_id"] in resolved_ids
        ]
        failures = []
        for candidate_id in ordered:
            candidate = next(
                item
                for item in candidate_set["candidates"]
                if item["candidate_id"] == candidate_id
            )
            try:
                return self.action_agent.draft(candidate), failures
            except ActionCapabilityError:
                raise
            except (TypeError, ValueError) as exc:
                failures.append(
                    {
                        "candidate_id": candidate_id,
                        "stage": "unbound_action",
                        "draft": deepcopy(candidate["draft"]),
                        "error": _error_record(exc),
                    }
                )
        raise ValueError(
            "No selected task candidate can be represented by Action Engine."
        )

    @staticmethod
    def _publish(
        transaction: ArtifactTransaction,
        staging: Path,
        request: Mapping[str, Any],
        workflow_cfg: TaskEngineWorkflowCfg,
        planning_cfg: TaskEnginePlanningCfg,
        execution_cfg: TaskEngineExecutionCfg,
        run_metadata: Mapping[str, Any],
        state: TaskEngineState,
        attempts: Sequence[Mapping[str, Any]],
        *,
        status: str,
        failure_class: str | None,
        final_bundle: Path | None = None,
    ) -> TaskEngineRunResult:
        state_path = staging / "workflow_state.json"
        manifest_path = staging / "run_manifest.json"
        _write_json(state_path, state.to_dict())
        _write_json(
            manifest_path,
            {
                "schema_version": TASK_ENGINE_RUN_MANIFEST_SCHEMA,
                "run_id": run_metadata["run_id"],
                "created_at": run_metadata["created_at"],
                "output_root": Path(request["output_dir"]).parent.as_posix(),
                "run_dir": Path(request["output_dir"]).as_posix(),
                "status": status,
                "failure_class": failure_class,
                "request": deepcopy(dict(request)),
                "configuration": {
                    "workflow": {
                        "max_parallel_workers": workflow_cfg.max_parallel_workers,
                        "max_scene_attempts": workflow_cfg.max_scene_attempts,
                        "max_action_attempts": workflow_cfg.max_action_attempts,
                    },
                    "planning": {
                        "candidate_count": planning_cfg.candidate_count,
                        "planning_mode": planning_cfg.planning_mode,
                        "max_episodes": planning_cfg.max_episodes,
                        "max_episode_steps": planning_cfg.max_episode_steps,
                    },
                    "execution": {
                        "num_envs": execution_cfg.num_envs,
                        "success_policy": execution_cfg.success_policy,
                        "min_successful_envs": execution_cfg.min_successful_envs,
                        "dataset_saving": bool(run_metadata["dataset_saving"]),
                    },
                },
                "attempts": deepcopy(list(attempts)),
                "final_bundle": (
                    None if final_bundle is None else final_bundle.as_posix()
                ),
            },
        )
        published = transaction.commit()
        return TaskEngineRunResult(
            status=status,
            output_dir=published,
            manifest_path=published / manifest_path.name,
            state_path=published / state_path.name,
            final_bundle=(
                None if final_bundle is None else published / "final" / "bundle"
            ),
            failure_class=failure_class,
        )


def _complete_materialized_scene(
    state: TaskEngineState,
    *,
    has_edit: bool,
) -> TaskEngineState:
    if has_edit and state.stages[WorkflowStage.SCENE_EDIT].value == "running":
        state = complete_stage(state, WorkflowStage.SCENE_EDIT)
        state = start_stage(state, WorkflowStage.SCENE_FINALIZATION)
    if state.stages[WorkflowStage.SCENE_FINALIZATION].value == "running":
        state = complete_stage(state, WorkflowStage.SCENE_FINALIZATION)
    return state


def _scene_remediable(
    preparation: PreparationResult,
    *,
    analysis: SceneAnalysis,
    request: Mapping[str, Any],
) -> bool:
    if preparation.status != "infeasible":
        return False
    report = preparation.feasibility_report
    if not isinstance(report, Mapping) or report.get("remediation_class") != (
        "scene_remediable"
    ):
        return False
    if analysis.input_kind == "image":
        return True
    return request["scene_edit_prompt"] is not None


def _is_scene_remediable_error(error: Exception) -> bool:
    """Return whether one typed Scene failure may create a new attempt."""
    return isinstance(error, (SceneRemediableError, SceneServiceError))


def _preparation_failure_class(
    preparation: PreparationResult | None,
    *,
    scene_error: Exception | None,
    analysis: SceneAnalysis,
    request: Mapping[str, Any],
) -> str:
    if scene_error is not None:
        return "scene_materialization"
    if preparation is None:
        return "scene_materialization"
    if preparation.status == "planning_failed":
        return "action_capability"
    if preparation.status in {"ambiguous", "unsatisfied"}:
        return "input_conflict"
    if preparation.status == "infeasible":
        report = preparation.feasibility_report
        remediation = (
            str(report.get("remediation_class"))
            if isinstance(report, Mapping)
            else "terminal"
        )
        if remediation == "action_capability":
            return "action_capability"
        if remediation == "input_conflict":
            return "input_conflict"
        if remediation != "scene_remediable":
            return "terminal_feasibility"
        if (
            analysis.input_kind == "gym_project"
            and request["scene_edit_prompt"] is None
        ):
            return "read_only_scene_infeasible"
        return "scene_infeasible"
    return "preparation"


def _failure_stage(
    failure_class: str,
    request: Mapping[str, Any],
) -> WorkflowStage:
    if failure_class == "action_capability":
        return WorkflowStage.GROUNDED_ACTION
    if failure_class == "preparation_error":
        return WorkflowStage.FINAL_BINDING
    if failure_class == "input_conflict":
        return WorkflowStage.FINAL_BINDING
    if failure_class in {
        "scene_infeasible",
        "read_only_scene_infeasible",
        "terminal_feasibility",
    }:
        return WorkflowStage.STATIC_FEASIBILITY
    if failure_class == "scene_materialization":
        return (
            WorkflowStage.SCENE_EDIT
            if request["scene_edit_prompt"] is not None
            else WorkflowStage.SCENE_FINALIZATION
        )
    return WorkflowStage.GROUNDED_ACTION


def _environment_successes(
    report: Mapping[str, Any],
    *,
    required_semantic_steps: Sequence[str] = (),
) -> list[bool]:
    environments = report.get("environments")
    if not isinstance(environments, Sequence) or isinstance(environments, (str, bytes)):
        raise ValueError("Execution report environments must be a sequence.")
    values = []
    for item in environments:
        if not isinstance(item, Mapping) or not isinstance(item.get("success"), bool):
            raise ValueError("Every execution environment requires boolean success.")
        success = bool(item["success"])
        if required_semantic_steps:
            semantics = item.get("semantic_success")
            if not isinstance(semantics, Mapping):
                success = False
            else:
                success = success and all(
                    semantics.get(step_id) is True
                    for step_id in required_semantic_steps
                )
        values.append(success)
    if not values:
        raise ValueError("Execution report must contain at least one environment.")
    return values


def _bundle_success_terms(bundle: Path) -> tuple[str, ...]:
    path = bundle / "grounded_task_plan.json"
    if not path.is_file():
        return ()
    try:
        value = _read_json(path)
        success_spec = value.get("success_spec")
        terms = success_spec.get("terms") if isinstance(success_spec, Mapping) else None
        strict = isinstance(value.get("schema_version"), str)
        if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
            if strict:
                raise ValueError("GroundedTaskPlan has no valid SuccessSpec terms.")
            return ()
        result = tuple(
            str(item["step_id"])
            for item in terms
            if isinstance(item, Mapping) and isinstance(item.get("step_id"), str)
        )
        if len(result) != len(terms) or (strict and not result):
            if strict:
                raise ValueError("GroundedTaskPlan SuccessSpec terms are invalid.")
            return ()
        return result
    except OSError:
        return ()


def _highest_vote_candidate(candidate_set: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = candidate_set.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("TaskCandidateSet.candidates must be a sequence.")
    values = [item for item in candidates if isinstance(item, Mapping)]
    if not values:
        raise ValueError("TaskCandidateSet requires at least one candidate.")
    return max(
        values,
        key=lambda item: (
            int(item.get("vote_count", 0)),
            str(item.get("candidate_id", "")),
        ),
    )


def _copy_trajectory_record(report: Mapping[str, Any], output_root: Path) -> str | None:
    raw = report.get("record_dir")
    if not isinstance(raw, str) or not raw:
        return None
    source = Path(raw).expanduser().resolve()
    if not source.is_dir():
        return None
    destination = output_root / "trajectory"
    if source == destination or destination in source.parents:
        return source.as_posix()
    shutil.copytree(source, destination)
    return destination.as_posix()


def _revision_record(revision: SceneRevision) -> dict[str, Any]:
    return {
        "source": revision.source.as_posix(),
        "output_root": (
            None if revision.output_root is None else revision.output_root.as_posix()
        ),
        "revision_id": revision.revision_id,
        "seed": revision.seed,
        "edit_plan": deepcopy(revision.edit_plan),
        "source_fingerprint": (
            None
            if revision.source_fingerprint is None
            else revision.source_fingerprint.to_dict()
        ),
    }


def _error_record(error: Exception) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "failure_type": (
            "scene_remediable" if _is_scene_remediable_error(error) else "terminal"
        ),
        "message": str(error),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
