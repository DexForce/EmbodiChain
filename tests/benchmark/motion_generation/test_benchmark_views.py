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

"""Focused contracts for separate planning and atomic benchmark views."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from embodichain.benchmarks.motion_generation.aggregation import aggregate_results
from embodichain.benchmarks.motion_generation.config import SuiteCfg
from embodichain.benchmarks.motion_generation.contracts import EvaluationDomain
from embodichain.benchmarks.motion_generation.metrics.trajectory import (
    compute_case_outcomes,
    make_failure_outcomes,
)
from embodichain.benchmarks.motion_generation.models import (
    AlgorithmRole,
    BenchmarkCase,
    CaseOutcome,
    PlannerMetadata,
    TrialPhase,
    TrialRecord,
)
from embodichain.benchmarks.motion_generation.reporting import write_markdown_report


def _case(track: str = "neutral", *, atomic: bool = False) -> BenchmarkCase:
    """Build a frozen case whose friendly name cannot identify its family."""
    pose = torch.eye(4).reshape(1, 1, 4, 4)
    pose[..., 0, 3] = 0.1
    return BenchmarkCase(
        suite_version="test_v3",
        track=track,
        scenario_id="ambiguous",
        case_id=f"{track}-case",
        seed=7,
        batch_size=1,
        num_waypoints=1,
        path_shape="direct",
        start_state_bin="nominal",
        start_qpos=torch.zeros(1, 1),
        target_waypoints=pose,
        reference_qpos=torch.tensor([[[0.1]]]),
        primary_success="physical_skill_success" if atomic else "geometry_valid",
        domain=EvaluationDomain(id="nominal", version="v1", kind="nominal"),
    )


def _outcome(*, timed: bool | None = True, physical: bool | None = None) -> CaseOutcome:
    """Produce one externally valid path independently of its clock."""
    return replace(
        make_failure_outcomes(1, "unused")[0],
        planning_success=True,
        finite=True,
        ordered_waypoints_reached=True,
        motion_valid=timed is not False,
        completed_waypoint_ratio=1.0,
        final_translation_err_mm=2.0,
        joint_path_length_rad=0.1,
        timed_motion_valid=timed,
        execution_success=physical,
        physical_skill_success=physical,
        task_success=physical,
        failure_code=None,
    )


def _metadata(algorithm: str, *, timed: bool = True) -> PlannerMetadata:
    """Describe a configured algorithm and its default clock capability."""
    return PlannerMetadata(
        algorithm_id=algorithm,
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        adapter="fake",
        config_hash="test",
        capabilities=frozenset(),
        timing_is_solved=timed,
    )


def _record(
    case: BenchmarkCase,
    algorithm: str = "solved",
    *,
    outcome: CaseOutcome | None = None,
    timed: bool = True,
) -> TrialRecord:
    """Build a measured record retaining the explicit timing capability."""
    return TrialRecord(
        suite_version=case.suite_version,
        track=case.track,
        scenario_id=case.scenario_id,
        case_id=case.case_id,
        algorithm_id=algorithm,
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        model_revision="test",
        planner_config_hash="test",
        seed=case.seed,
        repeat=0,
        batch_size=1,
        waypoint_count=1,
        path_shape=case.path_shape,
        start_state_bin=case.start_state_bin,
        phase=TrialPhase.MEASURED,
        primary_success=case.primary_success,
        domain=case.domain,
        cost_time_ms=4.0,
        execution_time_ms=8.0,
        end_to_end_time_ms=12.0,
        trajectory_duration_s=1.0,
        metadata={"timing_is_solved": timed},
        outcomes=(
            _outcome(timed=True if timed else None) if outcome is None else outcome,
        ),
    )


class _Robot:
    """Expose one linear joint for geometric and timing validation."""

    device = torch.device("cpu")

    def get_qpos_limits(self, *, name: str) -> torch.Tensor:
        return torch.tensor([[[-1.0, 1.0]]])

    def get_qvel_limits(self, *, name: str) -> torch.Tensor:
        return torch.tensor([[0.2]])

    def compute_batch_fk(
        self, *, qpos: torch.Tensor, name: str, to_matrix: bool
    ) -> torch.Tensor:
        pose = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        pose[..., 0, 3] = qpos[..., 0]
        return pose


def test_geometry_and_timing_verdicts_use_distinct_contracts() -> None:
    result = SimpleNamespace(
        positions=torch.tensor([[[0.0], [0.1]]]),
        dt=torch.tensor([[0.0, 0.1]]),
        success=True,
    )
    kwargs = dict(
        validation_samples=2,
        position_threshold_m=0.001,
        rotation_threshold_rad=0.001,
        joint_limit_tolerance_rad=0.001,
    )
    solved = compute_case_outcomes(
        result, _case(), _Robot(), "arm", timing_is_solved=True, **kwargs
    )[0]
    nominal = compute_case_outcomes(
        result, _case(), _Robot(), "arm", timing_is_solved=False, **kwargs
    )[0]
    assert solved.geometry_valid is nominal.geometry_valid is True
    assert solved.timed_motion_valid is False
    assert nominal.timed_motion_valid is None
    assert solved.motion_valid is False and nominal.motion_valid is True


def test_geometry_is_recomputed_when_raw_outcomes_are_round_tripped() -> None:
    outcome = _outcome(timed=None)
    raw = _record(_case(), outcome=outcome).to_dict()["outcomes"][0]
    raw.pop("end_to_end_success")
    assert CaseOutcome(**raw).geometry_valid is True
    assert CaseOutcome(**raw).timed_motion_valid is None
    assert replace(outcome, joint_limit_violation=True).geometry_valid is False


def test_missing_failed_and_nominal_timing_do_not_receive_timed_credit() -> None:
    case = _case()
    failed = replace(
        _record(case), repeat=1, outcomes=make_failure_outcomes(1, "planning_failed")
    )
    records = [_record(case), failed, _record(case, "nominal", timed=False)]
    result = aggregate_results(
        records,
        [_metadata("solved"), _metadata("nominal", timed=False)],
        [case],
        2,
        track_scenarios={case.track: "free_space"},
    )
    timed = {row["algorithm"]: row for row in result["motion_timing_leaderboard"]}
    assert timed["solved"]["overall_success_rate"] == pytest.approx(0.5)
    assert timed["nominal"]["overall_success_rate"] is None
    assert timed["nominal"]["rank"] is None
    assert timed["nominal"]["eligible"] is False
    missing = aggregate_results(
        [_record(case)],
        [_metadata("solved")],
        [case],
        2,
        track_scenarios={case.track: "free_space"},
    )
    assert missing["motion_timing_leaderboard"][0][
        "overall_success_rate"
    ] == pytest.approx(0.5)


def test_geometry_error_population_includes_a_dynamic_failure() -> None:
    case = _case()
    result = aggregate_results(
        [_record(case, outcome=_outcome(timed=False))],
        [_metadata("solved")],
        [case],
        1,
        track_scenarios={case.track: "free_space"},
    )
    row = result["motion_planning_success_and_metrics"][0]
    assert row["n_valid"] == 1
    assert row["final_pos_err_mm"] == pytest.approx(2.0)
    assert row["geometry_valid_rate"] == 1.0
    assert row["timed_motion_valid_rate"] == 0.0


@pytest.mark.parametrize("unsupported", [False, True])
def test_effective_algorithm_preserves_each_source_population(
    unsupported: bool,
) -> None:
    case = _case(atomic=True)
    metadata = [_metadata("source_a"), _metadata("source_b", timed=False)]
    attribution = {
        "selected_planner_backend_used": False,
        "effective_motion_algorithm": "exact_ik",
        "timing_is_solved": True,
    }
    success = replace(
        _record(case, "source_a", outcome=_outcome(physical=True)), metadata=attribution
    )
    absent = replace(
        _record(case, "source_b"),
        phase=TrialPhase.AVAILABILITY if unsupported else TrialPhase.PREPARE,
        status="unsupported" if unsupported else "error",
        failure_code="unsupported_capacity" if unsupported else "prepare_failed",
        metadata=attribution,
        outcomes=(),
    )
    result = aggregate_results(
        [success, absent],
        metadata,
        [case],
        1,
        track_scenarios={case.track: "atomic_skill"},
    )
    row = next(
        row
        for row in result["atomic_skills_success_and_metrics"]
        if row["algorithm"] == "exact_ik"
    )
    domain = next(
        row for row in result["generalization"] if row["algorithm"] == "exact_ik"
    )
    rank = next(
        row
        for row in result["atomic_skills_leaderboard"]
        if row["algorithm"] == "exact_ik"
    )
    expected = 1.0 if unsupported else 0.5
    assert (
        row["success_rate"]
        == domain["success_rate"]
        == rank["overall_success_rate"]
        == pytest.approx(expected)
    )
    assert row["coverage_rate"] == domain["coverage_rate"] == pytest.approx(0.5)
    assert row["timed_motion_valid_rate"] == pytest.approx(expected)


@pytest.mark.parametrize("atomic", [False, True])
def test_unsupported_algorithm_has_no_rates_or_rank_and_renders_na(
    tmp_path, atomic: bool
) -> None:
    """Unsupported results remain distinct from measured zero-success results."""
    case = _case(atomic=atomic)
    unsupported = replace(
        _record(case, "unsupported"),
        phase=TrialPhase.AVAILABILITY,
        status="unsupported",
        failure_code="unsupported_capability",
        outcomes=(),
    )
    failure = _record(
        case, "failed", outcome=make_failure_outcomes(1, "planning_failed")[0]
    )
    success = _record(case, "solved", outcome=_outcome(physical=atomic))
    result = aggregate_results(
        [unsupported, failure, success],
        [_metadata(name) for name in ("unsupported", "failed", "solved")],
        [case],
        1,
        track_scenarios={case.track: "atomic_skill" if atomic else "free_space"},
    )
    view = "atomic_skills" if atomic else "motion_planning"
    row = next(
        row
        for row in result[f"{view}_success_and_metrics"]
        if row["algorithm"] == "unsupported"
    )
    assert row["coverage_rate"] == 0.0
    for field in (
        "success_rate",
        "planning_success_rate",
        "geometry_valid_rate",
        "timed_motion_valid_rate",
        "motion_valid_rate",
        "ordered_waypoint_success_rate",
        "joint_violation_rate",
        "physical_skill_success_rate",
    ):
        assert row[field] is None, field

    rows = result[f"{view}_leaderboard"]
    assert [row["algorithm"] for row in rows] == [
        "solved",
        "failed",
        "unsupported",
    ]
    assert [row["overall_success_rate"] for row in rows] == [1.0, 0.0, None]
    assert [row["rank"] for row in rows] == [1, 2, None]
    assert rows[-1]["eligible"] is False
    assert rows[-1]["planning_success_rate"] is None

    report = write_markdown_report(
        tmp_path / "report.md", SuiteCfg(), result
    ).read_text()
    headers: list[str] = []
    checked_tables = 0
    for line in report.splitlines():
        if line.startswith("| "):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if "algorithm" in cells:
                headers = cells
            elif "unsupported" in cells and "success_rate" in headers:
                rendered = dict(zip(headers, cells))
                assert rendered["success_rate"] == "N/A"
                assert rendered["planning_success_rate"] == "N/A"
                assert rendered["geometry_valid_rate"] == "N/A"
                assert rendered["coverage_rate"] == "0.00%"
                checked_tables += 1
    assert checked_tables == 1


@pytest.mark.parametrize(
    ("failure_code", "expected_success"),
    [
        ("unsupported_capability", 1.0),
        ("unsupported_capacity", 1.0),
        ("runtime_unavailable", 0.5),
        ("prepare_failed", 0.5),
        (None, 0.5),
    ],
)
def test_only_explicitly_unsupported_cases_leave_success_denominator(
    failure_code: str | None, expected_success: float
) -> None:
    """Missing cases and runtime errors retain their weight beside successes."""
    supported = _case()
    absent = replace(supported, case_id="absent-case")
    records = [_record(supported)]
    if failure_code is not None:
        prepare_failed = failure_code == "prepare_failed"
        records.append(
            replace(
                _record(absent),
                phase=TrialPhase.PREPARE if prepare_failed else TrialPhase.AVAILABILITY,
                status="error" if prepare_failed else "unsupported",
                failure_code=failure_code,
                outcomes=(),
            )
        )
    result = aggregate_results(
        records,
        [_metadata("solved")],
        [supported, absent],
        1,
        track_scenarios={supported.track: "free_space"},
    )
    for table, success_field in (
        ("motion_planning_success_and_metrics", "success_rate"),
        ("motion_planning_leaderboard", "overall_success_rate"),
    ):
        row = result[table][0]
        assert row[success_field] == pytest.approx(expected_success)
        assert row["planning_success_rate"] == pytest.approx(expected_success)
        assert row["coverage_rate"] == pytest.approx(0.5)


def test_explicit_scenario_mapping_separates_scopes_and_mixed_reports(tmp_path) -> None:
    motion = _case("looks_atomic")
    atomic = _case("looks_free_space", atomic=True)
    result = aggregate_results(
        [_record(motion), _record(atomic, outcome=_outcome(physical=True))],
        [_metadata("solved")],
        [motion, atomic],
        1,
        track_scenarios={motion.track: "free_space", atomic.track: "atomic_skill"},
    )
    assert set(result) == {
        "report_views",
        "generalization",
        "motion_timing_leaderboard",
        "motion_planning_time_and_memory",
        "motion_planning_success_and_metrics",
        "motion_planning_leaderboard",
        "atomic_skills_time_and_memory",
        "atomic_skills_success_and_metrics",
        "atomic_skills_leaderboard",
    }
    planning_time = result["motion_planning_time_and_memory"][0]
    atomic_time = result["atomic_skills_time_and_memory"][0]
    assert (
        planning_time["planning_time_ms"] == 4.0
        and planning_time["compile_time_ms"] is None
    )
    assert (
        atomic_time["compile_time_ms"] == 4.0
        and atomic_time["planning_time_ms"] is None
    )
    assert atomic_time["warm_plan_ms_p95"] is None
    assert (
        atomic_time["execution_time_ms"] == 8.0
        and atomic_time["end_to_end_time_ms"] == 12.0
    )
    assert result["atomic_skills_leaderboard"][0]["latency_p95_ms"] is None
    assert result["atomic_skills_leaderboard"][0]["compile_latency_p95_ms"] == 4.0
    report = write_markdown_report(
        tmp_path / "report.md", SuiteCfg(), result
    ).read_text()
    assert "## Motion Planning" in report and "## Atomic Skills" in report
    assert report.count("\n| ---") == 6
    assert "nominal" in report.lower()


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({}, "Missing report scenario"),
        ({"atomic_skill": "custom"}, "Unsupported report scenarios"),
    ],
)
def test_report_scenario_mapping_is_explicit(
    mapping: dict[str, str], message: str
) -> None:
    case = _case("atomic_skill")
    with pytest.raises(ValueError, match=message):
        aggregate_results(
            [_record(case)],
            [_metadata("solved")],
            [case],
            1,
            track_scenarios=mapping,
        )


def test_report_rejects_unclassified_aggregate_payload(tmp_path) -> None:
    with pytest.raises(ValueError, match="scenario-owned report_views"):
        write_markdown_report(tmp_path / "report.md", SuiteCfg(), {})


def test_effective_solved_clock_failure_without_measured_rows_is_zero() -> None:
    case = _case(atomic=True)
    failure = replace(
        _record(case, "nominal", timed=False),
        phase=TrialPhase.PREPARE,
        status="error",
        outcomes=(),
        metadata={
            "timing_is_solved": True,
            "selected_planner_backend_used": False,
            "effective_motion_algorithm": "exact_ik",
        },
    )
    result = aggregate_results(
        [failure],
        [_metadata("nominal", timed=False)],
        [case],
        2,
        track_scenarios={case.track: "atomic_skill"},
    )
    row = next(
        row
        for row in result["atomic_skills_success_and_metrics"]
        if row["algorithm"] == "exact_ik"
    )
    assert row["timed_motion_valid_rate"] == 0.0


def test_performance_distinguishes_a_planner_named_like_an_effective_algorithm() -> (
    None
):
    case = _case(atomic=True)
    selected = replace(_record(case, "exact_ik"), cost_time_ms=100.0)
    bypass = replace(
        _record(case, "source"),
        cost_time_ms=4.0,
        metadata={
            "selected_planner_backend_used": False,
            "effective_motion_algorithm": "exact_ik",
            "timing_is_solved": True,
        },
    )
    result = aggregate_results(
        [selected, bypass],
        [_metadata("exact_ik"), _metadata("source")],
        [case],
        1,
        track_scenarios={case.track: "atomic_skill"},
    )
    rows = [
        row
        for row in result["atomic_skills_time_and_memory"]
        if row["algorithm"] == "exact_ik"
    ]
    assert len(rows) == 2
    assert {row["algorithm_role"]: row["cost_time_ms"] for row in rows} == {
        "primary_baseline": 100.0,
        "effective_motion_algorithm": 4.0,
    }
    assert all(row["num_trials"] == 1 for row in rows)


def test_effective_motion_cost_does_not_include_selected_backend_lifecycle() -> None:
    case = _case(atomic=True)
    attribution = {
        "selected_planner_backend_used": False,
        "effective_motion_algorithm": "exact_ik",
        "timing_is_solved": True,
    }
    measured = replace(_record(case, "source"), metadata=attribution)
    construct = replace(
        measured, phase=TrialPhase.CONSTRUCT, cost_time_ms=100.0, outcomes=()
    )
    prepare = replace(
        measured, phase=TrialPhase.PREPARE, cost_time_ms=50.0, outcomes=()
    )
    result = aggregate_results(
        [construct, prepare, measured],
        [_metadata("source")],
        [case],
        1,
        track_scenarios={case.track: "atomic_skill"},
    )
    row = next(
        row
        for row in result["atomic_skills_time_and_memory"]
        if row["algorithm"] == "exact_ik"
    )
    assert row["planner_construct_ms"] is None
    assert row["backend_prepare_ms"] is None
    assert row["compile_time_ms"] == 4.0
