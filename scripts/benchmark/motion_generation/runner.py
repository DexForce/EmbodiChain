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
from embodichain.lab.sim.planners.utils import PlanResult
from embodichain.lab.sim.robots import FrankaPandaCfg

from . import planners as _builtin_planners  # noqa: F401 - registry side effects
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
from .metrics import compute_case_outcomes, timed_call
from .metrics.trajectory import make_failure_outcomes
from .models import (
    BenchmarkCase,
    PlannerMetadata,
    TrialPhase,
    TrialRecord,
)
from .planners.base import PlannerAdapter, PlannerContext
from .registry import create_planner_adapter, create_scenario_provider
from .reporting import write_markdown_report

if TYPE_CHECKING:
    from collections.abc import Callable

    from embodichain.lab.sim.objects import Robot

__all__ = ["BenchmarkRunResult", "BenchmarkRunner", "resolve_device"]

_T = TypeVar("_T")
_CONTROL_PART = "arm"
_ROBOT_UID = "benchmark_franka_panda"


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
    ) -> None:
        self.suite = suite
        self.planner_specs = planner_specs
        self.device = resolve_device(device)
        self.headless = headless
        self.output_root = Path(output_root)
        self.records: list[TrialRecord] = []
        self.cases: list[BenchmarkCase] = []
        self.metadata: dict[str, PlannerMetadata] = {}
        self.notes: list[str] = []

    def _create_simulation(self, batch_size: int) -> tuple[SimulationManager, "Robot"]:
        """Create one isolated Franka simulator for a fixed batch size."""
        sim = SimulationManager(
            SimulationManagerCfg(
                headless=self.headless,
                sim_device=str(self.device),
                num_envs=batch_size,
                arena_space=2.0,
            )
        )
        robot = sim.add_robot(
            cfg=FrankaPandaCfg.from_dict({"uid": _ROBOT_UID, "robot_type": "panda"})
        )
        sim.update(step=1)
        return sim, robot

    @staticmethod
    def _set_case_start(
        sim: SimulationManager, robot: "Robot", case: BenchmarkCase
    ) -> None:
        """Restore current and target robot state outside the timed region."""
        robot.set_qpos(case.start_qpos, name=_CONTROL_PART, target=False)
        robot.set_qpos(case.start_qpos, name=_CONTROL_PART, target=True)
        robot.clear_dynamics()
        sim.update(step=1)

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
        phase: TrialPhase,
        repeat: int,
    ) -> None:
        """Time one plan, validate outside timing, and persist the record."""
        self._set_case_start(sim, robot, case)
        measured = timed_call(lambda: _capture(lambda: adapter.plan(case)))
        result, error = measured.result
        failure_code = None
        failure_message = None
        status = "ok"
        if error is not None:
            status = "error"
            failure_code = "planner_exception"
            failure_message = str(error)
            outcomes = make_failure_outcomes(case.batch_size, failure_code)
        elif not isinstance(result, PlanResult):
            status = "error"
            failure_code = "planner_contract_error"
            failure_message = f"Expected PlanResult, got {type(result).__name__}."
            outcomes = make_failure_outcomes(case.batch_size, failure_code)
        elif phase in (TrialPhase.WARMUP, TrialPhase.COLD):
            # Cold/warmup timing must not pay for FK validation that is unused
            # by aggregation.
            outcomes = ()
        else:
            try:
                outcomes = compute_case_outcomes(
                    result,
                    case,
                    robot,
                    _CONTROL_PART,
                    validation_samples=self.suite.protocol.validation_samples,
                    position_threshold_m=self.suite.protocol.position_threshold_m,
                    rotation_threshold_rad=self.suite.protocol.rotation_threshold_rad,
                    joint_limit_tolerance_rad=(
                        self.suite.protocol.joint_limit_tolerance_rad
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - metric failure is recorded
                status = "error"
                failure_code = "metric_evaluation_error"
                failure_message = str(exc)
                outcomes = make_failure_outcomes(case.batch_size, failure_code)

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
                outcomes=outcomes,
            ),
        )
        if phase is not TrialPhase.WARMUP:
            print(
                f"  {metadata.algorithm_id:<16} B={case.batch_size:>3d} "
                f"W={case.num_waypoints} {case.path_shape:<16} "
                f"{phase.value:<8} {measured.cost_time_ms:>10.3f} ms "
                f"status={status}"
            )

    def _run_adapter(
        self,
        writer: TrialJsonlWriter,
        sim: SimulationManager,
        robot: "Robot",
        spec: PlannerSpecCfg,
        cases: list[BenchmarkCase],
        required_capabilities: frozenset[str],
    ) -> None:
        """Execute one adapter over every case for a fixed simulator batch."""
        context = PlannerContext(
            robot=robot,
            control_part=_CONTROL_PART,
            device=self.device,
            sample_interval=self.suite.protocol.sample_interval,
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

            self._run_plan_call(
                writer,
                sim,
                robot,
                adapter,
                metadata,
                first_case,
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
                        TrialPhase.MEASURED,
                        repeat=repeat,
                    )
        finally:
            adapter.close()

    def run(self) -> BenchmarkRunResult:
        """Run the suite and write all required artifacts."""
        run_dir = create_run_directory(self.output_root, self.suite.name)
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
                try:
                    sim, robot = self._create_simulation(batch_size)
                    cases = provider.generate_cases(
                        self.suite, track, robot, _CONTROL_PART, batch_size
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
                        )
                finally:
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
                "Collision, dynamic, execution, and task metrics are N/A in free-space-common v1.",
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
