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

from dataclasses import dataclass
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
from .config import PlannerSpecCfg, SuiteCfg
from .metrics import timed_call
from .models import (
    BenchmarkCase,
    PlannerMetadata,
    TrialPhase,
    TrialRecord,
)
from .planners.base import PlannerAdapter, PlannerContext
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
    ) -> None:
        self.suite = suite
        self.planner_specs = planner_specs
        self.device = resolve_device(device)
        self.headless = headless
        self.output_root = Path(output_root)
        self.video = VideoRecordCfg() if video is None else video
        self.robot_provider = create_robot_provider(suite.robot)
        self.control_part = self.robot_provider.control_part
        self.records: list[TrialRecord] = []
        self.cases: list[BenchmarkCase] = []
        self.metadata: dict[str, PlannerMetadata] = {}
        self.notes: list[str] = []
        self._run_dir: Path | None = None
        self._video_paths: list[str] = []

    def _create_simulation(self, batch_size: int) -> tuple[SimulationManager, "Robot"]:
        """Create one isolated suite-selected robot for a fixed batch size."""
        sim = SimulationManager(
            SimulationManagerCfg(
                headless=self.headless,
                sim_device=str(self.device),
                num_envs=batch_size,
                arena_space=2.0,
            )
        )
        robot = self.robot_provider.add_robot(sim)
        sim.update(step=1)
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
        }

    def _record_unavailable(
        self,
        writer: TrialJsonlWriter,
        metadata: PlannerMetadata,
        case: BenchmarkCase,
        reason: str,
        *,
        failure_code: str,
    ) -> None:
        """Record an unsupported/unavailable planner without counting a failure."""
        self._append(
            writer,
            TrialRecord(
                **self._base_record(metadata, case, TrialPhase.AVAILABILITY),
                status="unsupported",
                failure_code=failure_code,
                failure_message=reason,
            ),
        )
        note = f"{metadata.algorithm_id} skipped for B={case.batch_size}: {reason}"
        self.notes.append(note)
        print(f"SKIPPED: {note}")

    def _record_timed_lifecycle(
        self,
        writer: TrialJsonlWriter,
        metadata: PlannerMetadata,
        case: BenchmarkCase,
        phase: TrialPhase,
        callable_fn: "Callable[[], object]",
    ) -> tuple[object | None, Exception | None]:
        """Measure a construct/prepare operation and persist its outcome."""
        measured = timed_call(lambda: _capture(callable_fn))
        result, error = measured.result
        status = "error" if error is not None else "ok"
        phase_metadata = result if isinstance(result, dict) else {}
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
                evaluation = provider.evaluate_case(
                    result,
                    case,
                    robot,
                    self.control_part,
                    self.suite,
                    planning_time_ms=measured.cost_time_ms,
                )
                outcomes = evaluation.outcomes
            except Exception as exc:  # noqa: BLE001 - metric failure is recorded
                status = "error"
                failure_code = "metric_evaluation_error"
                failure_message = str(exc)
                outcomes = provider.failure_outcomes(case, failure_code)

        record_metadata = {} if evaluation is None else dict(evaluation.metadata)
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
        """Record one measured Atomic Task replay outside planner timing."""
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
        )
        adapter = create_planner_adapter(spec, context)
        metadata = adapter.metadata
        self.metadata.setdefault(metadata.algorithm_id, metadata)
        first_case = cases[0]
        missing = sorted(required_capabilities - adapter.capabilities)
        if missing:
            self._record_unavailable(
                writer,
                metadata,
                first_case,
                f"missing required capabilities: {', '.join(missing)}",
                failure_code="unsupported_capability",
            )
            return
        available, reason = adapter.availability()
        if not available:
            self._record_unavailable(
                writer,
                metadata,
                first_case,
                reason or "runtime unavailable",
                failure_code="runtime_unavailable",
            )
            return

        _, build_error = self._record_timed_lifecycle(
            writer,
            metadata,
            first_case,
            TrialPhase.CONSTRUCT,
            adapter.build,
        )
        if build_error is not None:
            adapter.close()
            return
        scenario_prepared = False
        try:
            if adapter.separate_prepare:
                _, prepare_error = self._record_timed_lifecycle(
                    writer,
                    metadata,
                    first_case,
                    TrialPhase.PREPARE,
                    lambda: adapter.prepare(first_case),
                )
                if prepare_error is not None:
                    return

            try:
                provider.prepare_planner(adapter, first_case)
                scenario_prepared = True
            except Exception as exc:  # noqa: BLE001 - recorded benchmark failure
                self._append(
                    writer,
                    TrialRecord(
                        **self._base_record(metadata, first_case, TrialPhase.PREPARE),
                        status="error",
                        failure_code="scenario_prepare_error",
                        failure_message=str(exc),
                    ),
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
            for case in cases:
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
            if scenario_prepared:
                provider.close_planner(adapter)
            adapter.close()

    def run(self) -> BenchmarkRunResult:
        """Run the suite and write all required artifacts."""
        run_dir = create_run_directory(self.output_root, self.suite.name)
        self._run_dir = run_dir
        write_resolved_suite(run_dir / "resolved_suite.yaml", self.suite)
        write_json(run_dir / "environment.json", environment_metadata())
        writer = TrialJsonlWriter(run_dir / "trials.jsonl")

        print("=" * 60)
        print("Motion Generation Benchmark")
        print("=" * 60)
        enabled_tracks = self.suite.enabled_tracks()
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
                    provider.configure_runtime(
                        sim,
                        robot,
                        self.suite,
                        track,
                        self.control_part,
                    )
                    runtime_configured = True
                    cases = provider.generate_cases(
                        self.suite, track, robot, self.control_part, batch_size
                    )
                    self.cases.extend(cases)
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
                    if runtime_configured:
                        provider.close_runtime()
                    if sim is not None:
                        # Benchmarks must aggregate and report after simulator
                        # teardown; the SimulationManager default exits the whole
                        # process, so opt into deferred in-process cleanup here.
                        sim.destroy(exit_process=False)
                        SimulationManager.flush_cleanup_queue()

        write_case_manifest(run_dir / "case_manifest.json", self.cases)
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
        )
        write_json(run_dir / "aggregates.json", aggregates)
        enabled_scenarios = {track.scenario for track in enabled_tracks}
        track_notes: list[str] = []
        if "free_space" in enabled_scenarios:
            track_notes.append(
                "Collision, dynamic, execution, and task metrics are N/A in "
                "free-space-common v1."
            )
        if "atomic_task" in enabled_scenarios:
            track_notes.extend(
                [
                    "Atomic Task cost_time_ms measures AtomicActionEngine.compile only; "
                    "execution_time_ms is common physics-replay wall time and "
                    "end_to_end_time_ms is their sum.",
                    "trajectory_duration_s is planner-native nominal duration; "
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
                "External position/rotation thresholds (default 0.01 m / 0.1 rad) are "
                "feasibility gates for ordered_waypoints_reached / motion_valid. "
                "Read final_*_err and waypoint_*_p95 for finer precision on the "
                "motion-valid subset.",
                "Continuous error and path metrics (final_*_err, waypoint_*_p95, "
                "path lengths, path_efficiency) average only motion_valid outcomes "
                "(success-conditioned / survivor-biased). Always read them with n_valid; "
                "a high path_efficiency on n_valid=2 is not comparable to n_valid=200.",
                *track_notes,
                "Leaderboard and Success-table boolean rates "
                "(overall_success_rate / success_rate / motion_valid_rate / "
                "planning_success_rate / ordered_waypoint_success_rate) are macro averages "
                "over mandatory cases (equal case weight after within-case env/repeat "
                "micro-average). Missing cases contribute 0.0. coverage_rate remains "
                "outcome-count completeness.",
                "Leaderboard latency_p95_ms is a case-macro tiebreaker: mean warm "
                "cost_time_ms within each case, then nearest-rank p95 across cases "
                "(equal case weight; missing cases omitted). Stratified absolute latency "
                "stays in Time & Memory (warm_plan_ms_p50/p95 by batch_size/waypoint_count).",
                "cold_plan_ms is reported only on the Time & Memory row whose waypoint_count "
                "matches the first real case measured for that batch; other waypoint rows "
                "show N/A. planner_construct_ms / backend_prepare_ms are one-time batch costs.",
                "Waypoint continuous errors use the same threshold-greedy arrival matching "
                "as ordered_waypoints_reached / motion_valid.",
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
