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

"""Physical evidence, stage denominators and independently reported outcomes."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from embodichain.benchmarks.motion_generation.aggregation import aggregate_results
from embodichain.benchmarks.motion_generation.artifacts import TrialJsonlWriter
from embodichain.benchmarks.motion_generation.config import SuiteCfg, TrackCfg
from embodichain.benchmarks.motion_generation.contracts import (
    PhysicalEvaluation,
    StageOutcome,
    validate_physical_evaluations,
)
from embodichain.benchmarks.motion_generation.evaluation import (
    STANDARD_VERSION,
    SKILL_CRITERIA,
    StageEvidence,
    evaluate_skill_stage_batch,
    evaluate_skill_stages,
    validate_standard_evaluation,
)
from embodichain.benchmarks.motion_generation.models import (
    AlgorithmRole,
    BenchmarkCase,
    CaseOutcome,
    PlannerMetadata,
    TrialPhase,
    TrialRecord,
    with_physical_evaluation,
)
from embodichain.benchmarks.motion_generation.reporting import write_markdown_report


def _evidence(window: str, **measurements: float | bool) -> StageEvidence:
    return StageEvidence(window, "simulation_state", measurements, "test_observation")


def _primitive(error: float, env_index: int = 0) -> PhysicalEvaluation:
    return evaluate_skill_stages(
        "move_end_effector",
        {
            "reached": _evidence(
                "terminal_settle",
                tcp_position_error_m=error,
                tcp_rotation_error_rad=0.0,
            )
        },
        env_index=env_index,
    )


def _outcome(env_index: int = 0) -> CaseOutcome:
    return CaseOutcome(
        env_index=env_index,
        planning_success=True,
        finite=True,
        ordered_waypoints_reached=True,
        motion_valid=True,
        completed_waypoint_ratio=1.0,
        final_translation_err_mm=0.0,
        final_rotation_err_deg=0.0,
        waypoint_translation_err_mm_mean=0.0,
        waypoint_translation_err_mm_p95=0.0,
        waypoint_translation_err_mm_max=0.0,
        waypoint_rotation_err_deg_mean=0.0,
        waypoint_rotation_err_deg_p95=0.0,
        waypoint_rotation_err_deg_max=0.0,
        joint_limit_violation=False,
        max_normalized_joint_violation=0.0,
        joint_path_length_rad=0.0,
        cartesian_path_length_m=0.0,
        path_efficiency=1.0,
        execution_success=True,
    )


def _case(batch_size: int = 1, case_id: str = "case") -> BenchmarkCase:
    return BenchmarkCase(
        suite_version="standard_test_v1",
        track="primitive",
        scenario_id="atomic_skill",
        case_id=case_id,
        seed=11,
        batch_size=batch_size,
        num_waypoints=1,
        path_shape="direct",
        start_state_bin="nominal",
        start_qpos=torch.zeros(batch_size, 7),
        target_waypoints=torch.eye(4).repeat(batch_size, 1, 1, 1),
        reference_qpos=torch.zeros(batch_size, 1, 7),
        skill_id="move_end_effector",
        primary_success="task_success",
    )


def _record(case: BenchmarkCase, outcomes: tuple[CaseOutcome, ...]) -> TrialRecord:
    return TrialRecord(
        suite_version=case.suite_version,
        track=case.track,
        scenario_id=case.scenario_id,
        case_id=case.case_id,
        algorithm_id="baseline",
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        model_revision="test",
        planner_config_hash="abc",
        seed=case.seed,
        repeat=0,
        batch_size=case.batch_size,
        waypoint_count=1,
        path_shape=case.path_shape,
        start_state_bin=case.start_state_bin,
        phase=TrialPhase.MEASURED,
        skill_id=case.skill_id,
        primary_success=case.primary_success,
        outcomes=outcomes,
    )


def _aggregates(
    cases: list[BenchmarkCase], records: list[TrialRecord], repeats: int = 1
) -> dict:
    metadata = [
        PlannerMetadata(
            "baseline",
            AlgorithmRole.PRIMARY_BASELINE,
            "ik_interpolate",
            "abc",
            frozenset({"atomic_action"}),
        )
    ]
    return aggregate_results(
        records,
        metadata,
        cases,
        repeats,
        track_scenarios={case.track: "atomic_skill" for case in cases},
    )


@pytest.mark.parametrize(
    "error,passed", [(0.0029, True), (0.003, True), (0.0031, False), (-0.001, False)]
)
def test_measured_primitive_boundary(error: float, passed: bool) -> None:
    result = _primitive(error)
    assert result.task_success is passed
    validate_standard_evaluation("move_end_effector", result)


def test_joint_criterion_uses_l2_measurement_not_max_joint_error() -> None:
    result = evaluate_skill_stages(
        "move_joints",
        {
            "reached": _evidence("terminal_settle", joint_error_l2_rad=0.006),
        },
        env_index=0,
    )
    assert result.failure_code == "task_goal_miss"


def test_batched_stage_evidence_preserves_environment_order() -> None:
    results = evaluate_skill_stage_batch(
        "move_end_effector",
        (
            {
                "reached": _evidence(
                    "terminal_settle",
                    tcp_position_error_m=0.002,
                    tcp_rotation_error_rad=0.0,
                )
            },
            {
                "reached": _evidence(
                    "terminal_settle",
                    tcp_position_error_m=0.004,
                    tcp_rotation_error_rad=0.0,
                )
            },
        ),
    )

    assert [result.env_index for result in results] == [0, 1]
    assert [result.task_success for result in results] == [True, False]
    for result in results:
        validate_standard_evaluation("move_end_effector", result)


def test_lift_then_drop_fails_closing_stage() -> None:
    result = evaluate_skill_stages(
        "pick_up",
        {
            "grasped": _evidence("grasp_end", grasp_established=True),
            "lifted": _evidence(
                "lift_end", lift_delta_m=0.05, min_height_above_support_m=0.0
            ),
            "held": _evidence(
                "terminal_settle",
                object_retained=True,
                min_height_above_support_m=-0.16,
            ),
        },
        env_index=0,
    )
    assert [stage.status for stage in result.stages] == ["passed", "passed", "failed"]
    assert result.failure_stage == "held"
    assert result.failure_code == "object_dropped"


def test_later_observations_do_not_repair_failed_release() -> None:
    result = evaluate_skill_stages(
        "place",
        {
            "released": _evidence("release_end", release_established=False),
            "placed": _evidence(
                "terminal_settle",
                position_error_m=0.0,
                orientation_error_rad=0.0,
                min_height_above_support_m=0.0,
            ),
            "stable": _evidence(
                "settle_interval",
                position_drift_m=0.0,
                rotation_drift_rad=0.0,
                min_height_above_support_m=0.0,
            ),
        },
        env_index=0,
    )
    assert result.failure_code == "release_failure"
    assert [stage.status for stage in result.stages] == [
        "failed",
        "not_reached",
        "not_reached",
    ]


def test_missing_grasp_is_unavailable_even_when_final_pose_looks_good() -> None:
    result = evaluate_skill_stages(
        "pick_up",
        {
            "lifted": _evidence(
                "lift_end", lift_delta_m=0.1, min_height_above_support_m=0.0
            ),
        },
        env_index=0,
    )
    assert result.failure_code == "evidence_unavailable"
    assert [stage.status for stage in result.stages] == [
        "unavailable",
        "not_reached",
        "not_reached",
    ]
    validate_standard_evaluation("pick_up", result)


def test_wrong_sampling_window_is_not_silently_accepted() -> None:
    with pytest.raises(ValueError, match="requires window"):
        evaluate_skill_stages(
            "move_end_effector",
            {
                "reached": _evidence("actuation_end", tcp_position_error_m=0.0),
            },
            env_index=0,
        )


@pytest.mark.parametrize("ratio,passed", [(-1.0, False), (0.79, False), (0.8, True)])
def test_press_uses_signed_peak_stroke(ratio: float, passed: bool) -> None:
    result = evaluate_skill_stages(
        "press",
        {
            "contacted": _evidence("actuation_interval", contact_established=True),
            "pressed": _evidence("actuation_interval", peak_signed_stroke_ratio=ratio),
        },
        env_index=0,
    )
    assert result.task_success is passed


def test_knob_joint_motion_does_not_substitute_for_executed_tcp_twist() -> None:
    result = evaluate_skill_stages(
        "twist",
        {
            "grasped": _evidence("grasp_end", grasp_established=True),
            "twisted": _evidence("actuation_end", knob_angle_error_rad=0.0),
        },
        env_index=0,
    )
    assert result.stages[1].status == "unavailable"


@pytest.mark.parametrize(
    "height,passed", [(-0.10, True), (0.03, True), (-0.101, False), (0.031, False)]
)
def test_handover_signed_height_bounds(height: float, passed: bool) -> None:
    result = evaluate_skill_stages(
        "hand_over",
        {
            "grasped": _evidence("grasp_end", grasp_established=True),
            "transferred": _evidence(
                "transfer_end",
                both_grasps_established=True,
                min_height_above_support_m=0.0,
            ),
            "handed_over": _evidence(
                "handover_end",
                giver_released=True,
                receiver_retained=True,
                min_height_above_support_m=0.0,
            ),
            "receiver_released": _evidence("terminal_settle", release_established=True),
            "placed": _evidence(
                "terminal_settle",
                xy_error_m=0.03,
                signed_z_error_m=height,
                min_height_above_support_m=0.0,
            ),
        },
        env_index=0,
    )
    assert result.task_success is passed


@pytest.mark.parametrize(
    "mutation",
    ["missing_stage", "fake_pass", "no_provenance", "wrong_version", "inapplicable"],
)
def test_claimed_standard_cannot_bypass_criteria(mutation: str) -> None:
    result = _primitive(0.004)
    if mutation == "missing_stage":
        result = replace(result, stages=(StageOutcome("different", "passed"),))
    elif mutation == "fake_pass":
        result = replace(
            result,
            stages=(replace(result.stages[0], status="passed", failure_code=None),),
        )
    elif mutation == "no_provenance":
        result = replace(
            result, stages=(replace(result.stages[0], evidence_source=None),)
        )
    elif mutation == "wrong_version":
        result = replace(result, protocol_version="unknown")
    else:
        result = replace(result, stages=(StageOutcome("reached", "not_applicable"),))
    with pytest.raises(ValueError):
        validate_standard_evaluation("move_end_effector", result)


def test_physical_success_is_independent_of_motion_and_execution() -> None:
    original = replace(
        _outcome(),
        motion_valid=False,
        execution_success=False,
        failure_code="joint_limit_violation",
    )
    outcome = with_physical_evaluation(
        original, _primitive(0.002), skill_id="move_end_effector"
    )
    assert outcome.physical_skill_success is True
    assert outcome.end_to_end_success is False
    assert outcome.task_success is False
    assert outcome.failure_code == "joint_limit_violation"
    assert outcome.physical_failure_code is None


def test_batch_cannot_mix_rows_or_protocol_versions() -> None:
    with pytest.raises(ValueError, match="environment row"):
        with_physical_evaluation(
            _outcome(1), _primitive(0.002), skill_id="move_end_effector"
        )
    with pytest.raises(ValueError, match="version must match"):
        validate_physical_evaluations(
            (_primitive(0.002), replace(_primitive(0.002, 1), protocol_version=None)),
            batch_size=2,
        )


def test_jsonl_and_report_preserve_layers_and_stage_denominators(
    tmp_path: Path,
) -> None:
    case = _case(3)
    physical = (
        _primitive(0.002),
        evaluate_skill_stages("move_end_effector", {}, env_index=1),
        _primitive(0.004, 2),
    )
    outcomes = tuple(
        with_physical_evaluation(_outcome(i), value, skill_id=case.skill_id)
        for i, value in enumerate(physical)
    )
    record = _record(case, outcomes)
    raw = tmp_path / "trials.jsonl"
    TrialJsonlWriter(raw).append(record)
    payload = json.loads(raw.read_text())
    assert payload["outcomes"][0]["evaluation_protocol"] == STANDARD_VERSION
    assert payload["outcomes"][0]["end_to_end_success"] is True
    assert payload["outcomes"][1]["physical_failure_code"] == "evidence_unavailable"
    aggregates = _aggregates([case], [record])
    metrics = aggregates["atomic_skills_success_and_metrics"][0]
    assert metrics["physical_skill_success_rate"] == pytest.approx(1 / 3)
    assert metrics["stage_success_rates"]["reached"] == pytest.approx(1 / 2)
    assert metrics["stage_counts"]["reached"] == {
        "reached": 2,
        "passed": 1,
        "failed": 1,
        "not_reached": 0,
        "not_applicable": 0,
        "unavailable": 1,
    }
    suite = SuiteCfg(tracks=[TrackCfg(id=case.track, scenario="atomic_skill")])
    report = write_markdown_report(tmp_path / "report.md", suite, aggregates)
    text = report.read_text()
    assert "physical_skill_success_rate" in text
    assert "stage_counts" in text
    assert STANDARD_VERSION in text
    assert sum(line.startswith("| ---") for line in text.splitlines()) == 3


def test_unreached_stage_does_not_reduce_conditional_success() -> None:
    case = _case(2)
    outcomes = (
        replace(
            _outcome(),
            stages=(
                StageOutcome("grasped", "passed"),
                StageOutcome("lifted", "passed"),
            ),
        ),
        replace(
            _outcome(1),
            stages=(
                StageOutcome("grasped", "failed", "miss"),
                StageOutcome("lifted", "not_reached"),
            ),
        ),
    )
    metrics = _aggregates([case], [_record(case, outcomes)])[
        "atomic_skills_success_and_metrics"
    ][0]
    assert metrics["stage_success_rates"] == {"grasped": 0.5, "lifted": 1.0}
    assert metrics["stage_counts"]["lifted"]["reached"] == 1


def test_all_unreached_stage_rate_is_null() -> None:
    case = _case()
    outcome = replace(_outcome(), stages=(StageOutcome("lifted", "not_reached"),))
    metrics = _aggregates([case], [_record(case, (outcome,))])[
        "atomic_skills_success_and_metrics"
    ][0]
    assert metrics["stage_success_rates"] == {"lifted": None}


def test_physical_rates_weight_cases_equally_and_penalize_missing_cases() -> None:
    small, large = _case(), _case(8, "large")
    passed = with_physical_evaluation(
        _outcome(), _primitive(0.002), skill_id=small.skill_id
    )
    metrics = _aggregates([small, large], [_record(small, (passed,))])[
        "atomic_skills_success_and_metrics"
    ]
    # Different B values have separate metric rows; the absent row is not a pass.
    assert metrics[0]["physical_skill_success_rate"] == 1.0
    assert metrics[1]["physical_skill_success_rate"] is None
    large = replace(
        large,
        batch_size=1,
        start_qpos=small.start_qpos,
        target_waypoints=small.target_waypoints,
        reference_qpos=small.reference_qpos,
    )
    metrics = _aggregates([small, large], [_record(small, (passed,))])[
        "atomic_skills_success_and_metrics"
    ][0]
    assert metrics["physical_skill_success_rate"] == 0.5
    assert metrics["physical_evaluation_coverage_rate"] == 0.5


def test_legacy_results_cannot_be_advertised_as_standard_physical_success() -> None:
    case = _case()
    metrics = _aggregates(
        [case], [_record(case, (replace(_outcome(), task_success=True),))]
    )["atomic_skills_success_and_metrics"][0]
    assert metrics["physical_skill_success_rate"] is None
    assert metrics["end_to_end_success_rate"] is None
    assert metrics["evaluation_protocols"] == ["legacy/unversioned"]


def test_unknown_skill_and_projected_evidence_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown standard skill"):
        evaluate_skill_stages("undeclared", {}, env_index=0)
    with pytest.raises(ValueError, match="execution measurements"):
        StageEvidence(
            "grasp_end", "projected_context", {"grasp_established": True}, "symbolic"
        )


def test_catalog_preserves_full_skill_stage_declarations() -> None:
    assert len(SKILL_CRITERIA) == 15
    assert tuple(s.stage_id for s in SKILL_CRITERIA["coordinated_pickment"]) == (
        "dual_grasped",
        "lifted",
        "moved",
        "settled",
        "held",
    )


def test_missing_repeat_lowers_physical_and_overall_success() -> None:
    case = _case()
    passed = with_physical_evaluation(
        _outcome(), _primitive(0.002), skill_id=case.skill_id
    )
    aggregates = _aggregates([case], [_record(case, (passed,))], repeats=2)
    metrics = aggregates["atomic_skills_success_and_metrics"][0]
    assert metrics["physical_skill_success_rate"] == 0.5
    assert metrics["end_to_end_success_rate"] == 0.5
    assert metrics["physical_evaluation_coverage_rate"] == 0.5
    assert aggregates["atomic_skills_leaderboard"][0]["overall_success_rate"] == 0.5


def test_suite_can_explicitly_rank_physical_success() -> None:
    case = replace(_case(), primary_success="physical_skill_success")
    physical = with_physical_evaluation(
        replace(_outcome(), motion_valid=False),
        _primitive(0.002),
        skill_id=case.skill_id,
    )
    aggregates = _aggregates([case], [_record(case, (physical,))])
    metrics = aggregates["atomic_skills_success_and_metrics"][0]
    assert metrics["physical_skill_success_rate"] == 1.0
    assert metrics["end_to_end_success_rate"] == 0.0
    assert aggregates["atomic_skills_leaderboard"][0]["overall_success_rate"] == 1.0
