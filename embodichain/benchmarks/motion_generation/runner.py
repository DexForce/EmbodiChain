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

"""Generic lifecycle runner for motion-generation benchmark tracks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

from . import planners as _builtin_planners  # noqa: F401 - registry side effects
from . import robots as _builtin_robots  # noqa: F401 - registry side effects
from . import scenarios as _builtin_scenarios  # noqa: F401 - registry side effects
from .aggregation import aggregate_results
from .artifacts import (
    TrialJsonlWriter,
    create_run_directory,
    environment_metadata,
    write_case_manifest,
    write_json,
    write_resolved_suite,
)
from .config import PlannerSpecCfg, SuiteCfg, stable_hash
from .embodiments import (
    EmbodimentResolution,
    check_embodiment_capabilities,
    resolve_embodiment,
)
from .metrics import timed_call
from .models import (
    BenchmarkCase,
    PlannerMetadata,
    TrialPhase,
    TrialRecord,
)
from .planners.base import (
    PlannerAdapter,
    PlannerContext,
    _case_motion_capability,
)
from .registry import (
    create_planner_adapter,
    create_robot_provider,
    create_scenario_provider,
)
from .reporting import write_markdown_report
from .scenarios.base import ScenarioEvaluation, ScenarioProvider
from .scenarios.free_space import FreeSpaceScenario
from .video import (
    VideoRecordCfg,
    should_record_case,
    summarize_video_recording,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from embodichain.lab.sim.objects import Robot

__all__ = ["BenchmarkRunResult", "BenchmarkRunner", "resolve_device"]

_T = TypeVar("_T")


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Paths and aggregate data produced by one completed run."""

    run_dir: Path
    report_path: Path
    trials_path: Path
    records: tuple[TrialRecord, ...]
    aggregates: dict[str, list[dict[str, object]]]


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` while rejecting an unavailable explicit CUDA request."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(requested)


def _capture(callable_fn: "Callable[[], _T]") -> tuple[_T | None, Exception | None]:
    """Return a callable result or its ordinary exception for timed recording."""
    try:
        return callable_fn(), None
    except Exception as exc:  # noqa: BLE001 - failures are benchmark outcomes
        return None, exc


class BenchmarkRunner:
    """Generate fixed cases, execute adapters, aggregate, and report results."""

    def __init__(
        self,
        suite: SuiteCfg,
        planner_specs: list[PlannerSpecCfg],
        *,
        device: str = "auto",
        headless: bool = True,
        output_root: str | Path = "outputs/benchmarks",
        video: VideoRecordCfg | None = None,
        adapters: Mapping[str, type[PlannerAdapter]] | None = None,
    ) -> None:
        self.adapters = dict(adapters or {})
        for name, adapter in self.adapters.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Adapter names must be non-empty strings.")
            if not isinstance(adapter, type) or not issubclass(adapter, PlannerAdapter):
                raise TypeError(f"Adapter {name!r} must be a PlannerAdapter subclass.")
        # One copy preserves shared planner references within the owned config.
        self.suite, self.planner_specs = deepcopy((suite, planner_specs))
        suite = self.suite
        self.device = resolve_device(device)
        self.headless = headless
        self.output_root = Path(output_root)
        self.video = VideoRecordCfg() if video is None else video
        self.embodiment_resolution: EmbodimentResolution = resolve_embodiment(
            suite.embodiment,
            suite.robot,
            base_dir=Path.cwd(),
        )
        if suite.embodiment is None:
            self.robot_provider = create_robot_provider(suite.robot)
        else:
            from .robots.base import EmbodimentRobotProvider

            self.robot_provider = EmbodimentRobotProvider(
                self.embodiment_resolution, suite.embodiment.resource_id
            )
            suite.robot = deepcopy(self.robot_provider.spec)
            for track in suite.tracks:
                self.robot_provider.configure_track(track)
        self.control_part = self.robot_provider.control_part
        self.records: list[TrialRecord] = []
        self.cases: list[BenchmarkCase] = []
        self.metadata: dict[str, PlannerMetadata] = {}
        self.notes: list[str] = []
        self._run_dir: Path | None = None
        self._video_paths: list[str] = []

    def _create_simulation(self, batch_size: int) -> tuple[SimulationManager, "Robot"]:
        """Create one uninitialized suite-selected robot for a fixed batch size."""
        sim = SimulationManager(
            SimulationManagerCfg(
                headless=self.headless,
                sim_device=str(self.device),
                num_envs=batch_size,
                arena_space=2.0,
            )
        )
        try:
            robot = self.robot_provider.add_robot(sim)
        except BaseException:
            try:
                sim.destroy(exit_process=False)
            finally:
                SimulationManager.flush_cleanup_queue()
            raise
        return sim, robot

    def _append(self, writer: TrialJsonlWriter, record: TrialRecord) -> None:
        """Retain and immediately persist one raw record."""
        self.records.append(record)
        writer.append(record)

    @staticmethod
    def _base_record(
        metadata: PlannerMetadata,
        case: BenchmarkCase,
        phase: TrialPhase,
        *,
        repeat: int = -1,
    ) -> dict[str, object]:
        """Build fields shared by all lifecycle records."""
        return {
            "suite_version": case.suite_version,
            "track": case.track,
            "scenario_id": case.scenario_id,
            "case_id": case.case_id,
            "algorithm_id": metadata.algorithm_id,
            "algorithm_role": metadata.algorithm_role,
            "model_revision": metadata.model_revision,
            "planner_config_hash": metadata.config_hash,
            "seed": case.seed,
            "repeat": repeat,
            "batch_size": case.batch_size,
            "waypoint_count": case.num_waypoints,
            "path_shape": case.path_shape,
            "start_state_bin": case.start_state_bin,
            "robot_id": case.robot_id,
            "skill_id": case.skill_id,
            "object_id": case.object_id,
            "task_difficulty": case.task_difficulty,
            "primary_success": case.primary_success,
            "phase": phase,
            "domain": case.domain,
            "track_group": case.track_group,
            "embodiment_id": case.embodiment_id,
        }

    def _record_unavailable(
        self,
        writer: TrialJsonlWriter,
        metadata: PlannerMetadata,
        case: BenchmarkCase,
        reason: str,
        *,
        failure_code: str,
        record_metadata: dict[str, object] | None = None,
    ) -> None:
        """Record an unsupported/unavailable planner without counting a failure."""
        self._append(
            writer,
            TrialRecord(
                **self._base_record(metadata, case, TrialPhase.AVAILABILITY),
                status="unsupported",
                failure_code=failure_code,
                failure_message=reason,
                metadata={} if record_metadata is None else dict(record_metadata),
            ),
        )
        note = f"{metadata.algorithm_id} skipped for B={case.batch_size}: {reason}"
        self.notes.append(note)
        print(f"SKIPPED: {note}")

    @staticmethod
    def _trial_metadata(
        provider: ScenarioProvider,
        case: BenchmarkCase,
        metadata: PlannerMetadata,
    ) -> dict[str, object]:
        """Record timing provenance for the path that actually owns motion."""
        values = dict(provider.trial_metadata(case))
        values.setdefault("timing_is_solved", metadata.timing_is_solved)
        limits = metadata.parameters.get("resolved_motion_limits")
        if limits is not None:
            values["resolved_motion_limits"] = deepcopy(limits)
        return values

    def _record_timed_lifecycle(
        self,
        writer: TrialJsonlWriter,
        metadata: PlannerMetadata,
        case: BenchmarkCase,
        phase: TrialPhase,
        callable_fn: "Callable[[], object]",
        *,
        record_metadata: dict[str, object] | None = None,
    ) -> tuple[object | None, Exception | None]:
        """Measure a construct/prepare operation and persist its outcome."""
        measured = timed_call(lambda: _capture(callable_fn))
        result, error = measured.result
        status = "error" if error is not None else "ok"
        phase_metadata = dict(result) if isinstance(result, dict) else {}
        if error is not None and record_metadata is not None:
            phase_metadata.update(record_metadata)
        self._append(
            writer,
            TrialRecord(
                **self._base_record(metadata, case, phase),
                status=status,
                failure_code="planner_exception" if error is not None else None,
                failure_message=str(error) if error is not None else None,
                cost_time_ms=measured.cost_time_ms,
                cpu_delta_mb=measured.cpu_delta_mb,
                gpu_delta_mb=measured.gpu_delta_mb,
                peak_gpu_mb=measured.peak_gpu_mb,
                metadata=phase_metadata,
            ),
        )
        if error is not None:
            self.notes.append(
                f"{metadata.algorithm_id} {phase.value} failed for "
                f"B={case.batch_size}: {error}"
            )
        return result, error

    def _record_lifecycle_failures(
        self,
        writer: TrialJsonlWriter,
        metadata: PlannerMetadata,
        cases: list[BenchmarkCase],
        phase: TrialPhase,
        *,
        failure_code: str,
        failure_message: str,
        record_metadata: dict[str, dict[str, object]],
    ) -> None:
        """Replicate a shared lifecycle failure across every affected case."""
        for case in cases:
            self._append(
                writer,
                TrialRecord(
                    **self._base_record(metadata, case, phase),
                    status="error",
                    failure_code=failure_code,
                    failure_message=failure_message,
                    metadata=dict(record_metadata[case.case_id]),
                ),
            )

    def _run_plan_call(
        self,
        writer: TrialJsonlWriter,
        sim: SimulationManager,
        robot: "Robot",
        adapter: PlannerAdapter,
        metadata: PlannerMetadata,
        case: BenchmarkCase,
        provider: ScenarioProvider,
        phase: TrialPhase,
        repeat: int,
    ) -> None:
        """Time one plan, validate outside timing, and persist the record."""
        record_metadata = self._trial_metadata(provider, case, metadata)
        provider.reset_case(sim, robot, case, self.control_part)
        measured = timed_call(
            lambda: _capture(lambda: provider.plan_case(adapter, case))
        )
        result, error = measured.result
        failure_code = None
        failure_message = None
        status = "ok"
        evaluation: ScenarioEvaluation | None = None
        if error is not None:
            status = "error"
            failure_code = "planner_exception"
            failure_message = str(error)
            outcomes = provider.failure_outcomes(case, failure_code)
        elif (contract_error := provider.plan_contract_error(result)) is not None:
            status = "error"
            failure_code = "planner_contract_error"
            failure_message = contract_error
            outcomes = provider.failure_outcomes(case, failure_code)
        elif phase in (TrialPhase.WARMUP, TrialPhase.COLD):
            # Cold/warmup timing must not pay for FK validation that is unused
            # by aggregation.
            outcomes = ()
        else:
            try:
                candidate_evaluation = provider.evaluate_case(
                    result,
                    case,
                    robot,
                    self.control_part,
                    self.suite,
                    planning_time_ms=measured.cost_time_ms,
                    timing_is_solved=bool(record_metadata["timing_is_solved"]),
                )
                conflicts = {
                    key: (record_metadata[key], value)
                    for key, value in candidate_evaluation.metadata.items()
                    if key in record_metadata and record_metadata[key] != value
                }
                if conflicts:
                    raise ValueError(
                        "Scenario evaluation changed pre-plan trial metadata: "
                        f"{conflicts}."
                    )
                record_metadata.update(candidate_evaluation.metadata)
                evaluation = candidate_evaluation
                outcomes = evaluation.outcomes
            except Exception as exc:  # noqa: BLE001 - metric failure is recorded
                status = "error"
                failure_code = "metric_evaluation_error"
                failure_message = str(exc)
                outcomes = provider.failure_outcomes(case, failure_code)

        if status == "error":
            outcomes = tuple(
                replace(
                    outcome,
                    timed_motion_valid=(
                        False if record_metadata["timing_is_solved"] else None
                    ),
                )
                for outcome in outcomes
            )

        video_path = self._maybe_record_replay(
            provider, result, case, evaluation, metadata.algorithm_id, phase
        )
        if video_path is not None:
            record_metadata["video_path"] = str(video_path)

        self._append(
            writer,
            TrialRecord(
                **self._base_record(metadata, case, phase, repeat=repeat),
                status=status,
                failure_code=failure_code,
                failure_message=failure_message,
                cost_time_ms=measured.cost_time_ms,
                cpu_delta_mb=measured.cpu_delta_mb,
                gpu_delta_mb=measured.gpu_delta_mb,
                peak_gpu_mb=measured.peak_gpu_mb,
                execution_time_ms=(
                    None if evaluation is None else evaluation.execution_time_ms
                ),
                end_to_end_time_ms=(
                    None if evaluation is None else evaluation.end_to_end_time_ms
                ),
                trajectory_duration_s=(
                    None if evaluation is None else evaluation.trajectory_duration_s
                ),
                trajectory_waypoints=(
                    None if evaluation is None else evaluation.trajectory_waypoints
                ),
                metadata=record_metadata,
                outcomes=outcomes,
            ),
        )
        if phase is not TrialPhase.WARMUP:
            print(
                f"  {metadata.algorithm_id:<16} B={case.batch_size:>3d} "
                f"W={case.num_waypoints} {case.skill_id:<20} "
                f"{phase.value:<8} {measured.cost_time_ms:>10.3f} ms "
                f"status={status}"
            )

    def _maybe_record_replay(
        self,
        provider: ScenarioProvider,
        result: object,
        case: BenchmarkCase,
        evaluation: ScenarioEvaluation | None,
        algorithm_id: str,
        phase: TrialPhase,
    ) -> Path | None:
        """Record one measured Atomic Skill replay outside planner timing."""
        if phase is not TrialPhase.MEASURED or not self.video.enabled:
            return None
        success = bool(
            evaluation is not None
            and evaluation.outcomes
            and all(bool(outcome.task_success) for outcome in evaluation.outcomes)
        )
        if not should_record_case(self.video, len(self._video_paths), success):
            return None
        if self._run_dir is None:
            raise RuntimeError("Benchmark run directory is not initialized.")
        output_dir = (
            self.video.output_dir
            if self.video.output_dir is not None
            else self._run_dir / "videos"
        )
        video_path = provider.record_replay(
            result,
            case,
            evaluation,
            output_dir=output_dir,
            algorithm_id=algorithm_id,
            video=self.video,
        )
        if video_path is not None:
            self._video_paths.append(str(video_path))
        return video_path

    def _run_adapter(
        self,
        writer: TrialJsonlWriter,
        sim: SimulationManager,
        robot: "Robot",
        spec: PlannerSpecCfg,
        cases: list[BenchmarkCase],
        required_capabilities: frozenset[str],
        provider: ScenarioProvider | None = None,
    ) -> None:
        """Execute one adapter over every case for a fixed simulator batch."""
        provider = provider or FreeSpaceScenario()
        context = PlannerContext(
            robot=robot,
            control_part=self.control_part,
            device=self.device,
            sample_interval=self.suite.protocol.sample_interval,
            robot_id=self.suite.robot.id,
            joint_velocity_limit_rad_s=self.suite.protocol.joint_velocity_limit_rad_s,
            joint_acceleration_limit_rad_s2=(
                self.suite.protocol.joint_acceleration_limit_rad_s2
            ),
            joint_jerk_limit_rad_s3=self.suite.protocol.joint_jerk_limit_rad_s3,
        )
        adapter = (
            create_planner_adapter(spec, context, adapters=self.adapters)
            if self.adapters
            else create_planner_adapter(spec, context)
        )
        metadata = adapter.metadata
        self.metadata.setdefault(metadata.algorithm_id, metadata)
        supported_cases = []
        for case in cases:
            embodiment_decision = check_embodiment_capabilities(
                self.embodiment_resolution,
                case.required_capabilities,
            )
            if embodiment_decision.status != "supported":
                self._record_unavailable(
                    writer,
                    metadata,
                    case,
                    embodiment_decision.reason or "unsupported embodiment capability",
                    failure_code="unsupported_capability",
                    record_metadata=self._trial_metadata(provider, case, metadata),
                )
                continue
            planner_required_capabilities = set(required_capabilities)
            missing_planner_capabilities = sorted(
                planner_required_capabilities - set(adapter.capabilities)
            )
            if missing_planner_capabilities:
                self._record_unavailable(
                    writer,
                    metadata,
                    case,
                    "missing required capabilities on planner adapter: "
                    + ", ".join(missing_planner_capabilities),
                    failure_code="unsupported_capability",
                    record_metadata=self._trial_metadata(provider, case, metadata),
                )
                continue
            motion_capability, modality_reason = _case_motion_capability(case)
            if (
                motion_capability is None
                or motion_capability not in adapter.capabilities
            ):
                self._record_unavailable(
                    writer,
                    metadata,
                    case,
                    modality_reason
                    or f"case requires planner capability {motion_capability!r}",
                    failure_code="unsupported_capability",
                    record_metadata=self._trial_metadata(provider, case, metadata),
                )
                continue
            supported, reason = adapter.supports_case(case)
            if supported:
                supported_cases.append(case)
                continue
            self._record_unavailable(
                writer,
                metadata,
                case,
                reason or "case is outside planner capacity",
                failure_code="unsupported_capacity",
                record_metadata=self._trial_metadata(provider, case, metadata),
            )
        if not supported_cases:
            return
        first_case = supported_cases[0]
        lifecycle_metadata = {
            case.case_id: self._trial_metadata(provider, case, metadata)
            for case in supported_cases
        }
        available, reason = adapter.availability()
        if not available:
            for case in supported_cases:
                self._record_unavailable(
                    writer,
                    metadata,
                    case,
                    reason or "runtime unavailable",
                    failure_code="runtime_unavailable",
                    record_metadata=lifecycle_metadata[case.case_id],
                )
            return

        _, build_error = self._record_timed_lifecycle(
            writer,
            metadata,
            first_case,
            TrialPhase.CONSTRUCT,
            adapter.build,
            record_metadata=lifecycle_metadata[first_case.case_id],
        )
        if build_error is not None:
            self._record_lifecycle_failures(
                writer,
                metadata,
                supported_cases[1:],
                TrialPhase.CONSTRUCT,
                failure_code="planner_exception",
                failure_message=str(build_error),
                record_metadata=lifecycle_metadata,
            )
            adapter.close()
            return
        # Adapters resolve robot-owned limits during construction. Persist the
        # actual per-batch budgets rather than only the unresolved suite request.
        metadata = adapter.metadata
        self.metadata[metadata.algorithm_id] = metadata
        lifecycle_metadata = {
            case.case_id: self._trial_metadata(provider, case, metadata)
            for case in supported_cases
        }
        scenario_prepared = False
        try:
            if adapter.separate_prepare:
                _, prepare_error = self._record_timed_lifecycle(
                    writer,
                    metadata,
                    first_case,
                    TrialPhase.PREPARE,
                    lambda: adapter.prepare_cases(supported_cases),
                    record_metadata=lifecycle_metadata[first_case.case_id],
                )
                if prepare_error is not None:
                    self._record_lifecycle_failures(
                        writer,
                        metadata,
                        supported_cases[1:],
                        TrialPhase.PREPARE,
                        failure_code="planner_exception",
                        failure_message=str(prepare_error),
                        record_metadata=lifecycle_metadata,
                    )
                    return

            try:
                provider.prepare_planner(adapter, first_case)
                scenario_prepared = True
            except Exception as exc:  # noqa: BLE001 - recorded benchmark failure
                self._record_lifecycle_failures(
                    writer,
                    metadata,
                    supported_cases,
                    TrialPhase.PREPARE,
                    failure_code="scenario_prepare_error",
                    failure_message=str(exc),
                    record_metadata=lifecycle_metadata,
                )
                self.notes.append(
                    f"{metadata.algorithm_id} scenario prepare failed for "
                    f"B={first_case.batch_size}: {exc}"
                )
                return

            self._run_plan_call(
                writer,
                sim,
                robot,
                adapter,
                metadata,
                first_case,
                provider,
                TrialPhase.COLD,
                repeat=-1,
            )
            for case in supported_cases:
                for warmup_index in range(self.suite.protocol.warmup_trials):
                    self._run_plan_call(
                        writer,
                        sim,
                        robot,
                        adapter,
                        metadata,
                        case,
                        provider,
                        TrialPhase.WARMUP,
                        repeat=warmup_index,
                    )
                for repeat in range(self.suite.protocol.measured_trials):
                    self._run_plan_call(
                        writer,
                        sim,
                        robot,
                        adapter,
                        metadata,
                        case,
                        provider,
                        TrialPhase.MEASURED,
                        repeat=repeat,
                    )
        finally:
            try:
                if scenario_prepared:
                    provider.close_planner(adapter)
            finally:
                adapter.close()

    def run(self) -> BenchmarkRunResult:
        """Run the suite and write all required artifacts."""
        run_dir = create_run_directory(self.output_root, self.suite.name)
        self._run_dir = run_dir
        write_resolved_suite(run_dir / "resolved_suite.yaml", self.suite)
        write_json(run_dir / "environment.json", environment_metadata())
        writer = TrialJsonlWriter(run_dir / "trials.jsonl")
        planner_config_hashes = {
            spec.id: stable_hash(spec.config) for spec in self.planner_specs
        }

        print("=" * 60)
        print("Motion Generation Benchmark")
        print("=" * 60)
        enabled_tracks = self.suite.evaluation_tracks()
        if self.suite.embodiment is not None:
            for track in enabled_tracks:
                self.robot_provider.configure_track(track)
        print(
            f"suite={self.suite.suite_version} device={self.device} "
            f"tracks={','.join(track.id for track in enabled_tracks)} "
            f"planners={','.join(spec.id for spec in self.planner_specs)}"
        )

        for track in enabled_tracks:
            provider = create_scenario_provider(track.scenario)
            for batch_size in provider.batch_sizes(self.suite, track):
                sim: SimulationManager | None = None
                runtime_configured = False
                try:
                    sim, robot = self._create_simulation(batch_size)
                    runtime_configured = True
                    provider.create_runtime_entities(sim, self.suite, track)
                    # GPU physics topology must be complete before its first
                    # update; scenario entities cannot be appended afterwards.
                    sim.update(step=1)
                    provider.configure_runtime(
                        sim,
                        robot,
                        self.suite,
                        track,
                        self.control_part,
                    )
                    cases = provider.generate_cases(
                        self.suite, track, robot, self.control_part, batch_size
                    )
                    domain = (
                        track.domain.to_identity() if track.domain is not None else None
                    )
                    if domain is not None and any(
                        case.domain not in (None, domain) for case in cases
                    ):
                        raise ValueError("Provider domain conflicts with track.domain.")
                    track_group = track.group_id or track.id
                    resolved_cases = []
                    for case in cases:
                        if (
                            case.robot_id != self.suite.robot.id
                            or case.embodiment_id
                            not in (None, self.embodiment_resolution.embodiment_id)
                        ):
                            raise ValueError(
                                "Provider case conflicts with suite embodiment."
                            )
                        case_domain = case.domain or domain
                        required = set(case.required_capabilities)
                        if case_domain is not None:
                            required.update(case_domain.required_capabilities)
                        resolved_cases.append(
                            replace(
                                case,
                                domain=case_domain,
                                track_group=track_group,
                                embodiment_id=case.embodiment_id
                                or self.embodiment_resolution.embodiment_id,
                                required_capabilities=frozenset(required),
                            )
                        )
                    cases = resolved_cases
                    self.cases.extend(cases)
                    # Persist each frozen population before evaluating candidates.
                    write_case_manifest(
                        run_dir / "case_manifest.json",
                        self.cases,
                        suite=self.suite,
                        planner_config_hashes=planner_config_hashes,
                        embodiment_resolution=self.embodiment_resolution,
                    )
                    for spec in self.planner_specs:
                        self._run_adapter(
                            writer,
                            sim,
                            robot,
                            spec,
                            cases,
                            provider.required_capabilities,
                            provider,
                        )
                finally:
                    try:
                        if runtime_configured:
                            provider.close_runtime()
                    finally:
                        if sim is not None:
                            # Let the caller continue after simulator teardown.
                            try:
                                sim.destroy(exit_process=False)
                            finally:
                                SimulationManager.flush_cleanup_queue()

        write_case_manifest(
            run_dir / "case_manifest.json",
            self.cases,
            suite=self.suite,
            planner_config_hashes=planner_config_hashes,
            embodiment_resolution=self.embodiment_resolution,
        )
        metadata = [
            self.metadata[spec.id]
            for spec in self.planner_specs
            if spec.id in self.metadata
        ]
        aggregates = aggregate_results(
            self.records,
            metadata,
            self.cases,
            self.suite.protocol.measured_trials,
            track_scenarios={track.id: track.scenario for track in enabled_tracks},
        )
        write_json(
            run_dir / "aggregates.json",
            {
                "report_schema_version": self.suite.protocol.report_schema_version,
                **aggregates,
            },
        )
        enabled_scenarios = {track.scenario for track in enabled_tracks}
        track_notes: list[str] = []
        if "free_space" in enabled_scenarios:
            track_notes.append(
                "Motion Planning validates geometry and reports sampled dynamics "
                "separately for solved timing. Collision, physical execution, and "
                "task completion are not evaluated by free_space."
            )
        if "atomic_skill" in enabled_scenarios:
            track_notes.extend(
                [
                    "Atomic Skill cost_time_ms measures AtomicActionEngine.compile only; "
                    "execution_time_ms is common physics-replay wall time and "
                    "end_to_end_time_ms is their sum.",
                    "trajectory_duration_s is available only for solved trajectory timing; "
                    "task_completion_time_s is simulated replay time through the "
                    "stability hold for successful tasks.",
                ]
            )
        track_notes.extend(summarize_video_recording(self.video, self._video_paths))
        report_path = write_markdown_report(
            run_dir / "report.md",
            self.suite,
            aggregates,
            notes=[
                "CPU/GPU memory values are process/PyTorch allocator deltas around timed calls.",
                "Ordered Cartesian matching uses position/rotation thresholds of "
                f"{self.suite.protocol.position_threshold_m:g} m / "
                f"{self.suite.protocol.rotation_threshold_rad:g} rad. Joint tasks "
                "use ordered joint-waypoint error and keep Cartesian errors as diagnostics.",
                "Continuous error and path metrics average only geometry_valid outcomes; "
                "read these success-conditioned values alongside n_valid and success rates.",
                *track_notes,
                "Success rates give each mandatory case equal weight after averaging its "
                "environment/repeat outcomes. Unsupported cases are N/A; missing supported "
                "results count as failures. coverage_rate measures outcome completeness.",
                "Ranking latency uses nearest-rank p95 of per-case mean warm cost_time_ms, "
                "omitting missing cases: latency_p95_ms for Motion Planning and "
                "compile_latency_p95_ms for Atomic Skills. Time & Memory retains trial-level "
                "latency by batch_size/waypoint_count.",
                "planner_construct_ms / backend_prepare_ms are one-time batch costs. "
                "cold_plan_ms appears only on the row matching that batch's first measured "
                "case; other waypoint rows show N/A.",
                *self.notes,
            ],
        )
        print("=" * 60)
        print("Benchmarks complete.")
        print("=" * 60)
        print(f"Markdown report saved: {report_path}")
        return BenchmarkRunResult(
            run_dir=run_dir,
            report_path=report_path,
            trials_path=writer.path,
            records=tuple(self.records),
            aggregates=aggregates,
        )
