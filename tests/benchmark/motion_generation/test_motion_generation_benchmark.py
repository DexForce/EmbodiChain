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

"""Unit tests for the free-space motion-generation benchmark architecture."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.planners.curobo.curobo_planner import CuroboPlanner
from embodichain.lab.sim.planners.utils import MoveType, PlanResult
from scripts.benchmark.motion_generation.aggregation import aggregate_results
from scripts.benchmark.motion_generation.config import load_suite
from scripts.benchmark.motion_generation.metrics.trajectory import (
    compute_case_outcomes,
    compute_waypoint_errors,
    match_ordered_waypoints,
)
from scripts.benchmark.motion_generation import (
    scenarios as _scenarios,
)  # noqa: F401
from scripts.benchmark.motion_generation.models import (
    AlgorithmRole,
    BenchmarkCase,
    CaseOutcome,
    PlannerMetadata,
    TrialPhase,
    TrialRecord,
)
from scripts.benchmark.motion_generation.registry import (
    create_scenario_provider,
    scenario_provider_names,
)
from scripts.benchmark.motion_generation.reporting import (
    write_markdown_report,
)
from scripts.benchmark.motion_generation.run_benchmark import (
    _apply_overrides,
)


def _translated_pose(x: float) -> torch.Tensor:
    pose = torch.eye(4)
    pose[0, 3] = x
    return pose


def test_compute_waypoint_errors_uses_ordered_trajectory_hits():
    waypoints = torch.stack(
        [
            torch.eye(4),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ]
    )
    errors = compute_waypoint_errors([torch.eye(4), waypoints[1]], waypoints)
    assert errors["mean_waypoint_pos_err_mm"] == pytest.approx(0.0)
    assert errors["max_waypoint_pos_err_mm"] == pytest.approx(0.0)


def test_ordered_waypoints_reject_out_of_order_hits():
    waypoints = torch.stack([_translated_pose(0.1), _translated_pose(0.0)])
    trajectory = torch.stack([_translated_pose(0.0), _translated_pose(0.1)])

    result = match_ordered_waypoints(
        trajectory,
        waypoints,
        position_threshold_m=1.0e-4,
        rotation_threshold_rad=1.0e-4,
    )

    assert result["ordered_waypoints_reached"] is False
    assert result["completed_waypoint_ratio"] == pytest.approx(0.5)


def test_ordered_waypoint_requires_position_and_rotation_at_same_sample():
    target = torch.eye(4)
    target[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    position_only = torch.eye(4)
    rotation_only = target.clone()
    rotation_only[0, 3] = 0.2

    result = match_ordered_waypoints(
        torch.stack([position_only, rotation_only]),
        target.unsqueeze(0),
        position_threshold_m=0.01,
        rotation_threshold_rad=0.01,
    )

    assert result["ordered_waypoints_reached"] is False
    assert result["arrival_indices"] == []


class _MetricRobot:
    device = torch.device("cpu")

    def get_qpos_limits(self, name: str):  # noqa: ARG002
        limits = torch.tensor([[-1.0, 1.0]]).repeat(7, 1)
        return limits.unsqueeze(0)

    def compute_batch_fk(
        self, qpos: torch.Tensor, name: str, to_matrix: bool
    ):  # noqa: ARG002
        poses = torch.eye(4).repeat(qpos.shape[0], qpos.shape[1], 1, 1)
        poses[..., :3, 3] = qpos[..., :3]
        return poses


def test_motion_valid_is_independent_of_planner_reported_success():
    case = _case()
    case.target_waypoints[0, 0, 0, 3] = 0.1
    positions = torch.zeros(1, 2, 7)
    positions[0, 1, 0] = 0.1

    outcomes = compute_case_outcomes(
        PlanResult(success=False, positions=positions),
        case,
        _MetricRobot(),
        "arm",
        validation_samples=8,
        position_threshold_m=1.0e-4,
        rotation_threshold_rad=1.0e-4,
        joint_limit_tolerance_rad=1.0e-5,
    )

    assert outcomes[0].planning_success is False
    assert outcomes[0].motion_valid is True
    assert outcomes[0].failure_code is None
    assert outcomes[0].planner_failure_code == "planner_reported_failure"


def test_top_failure_ignores_planner_internal_codes_when_motion_valid():
    case = _case()
    case.target_waypoints[0, 0, 0, 3] = 0.1
    positions = torch.zeros(1, 2, 7)
    positions[0, 1, 0] = 0.1
    outcomes = compute_case_outcomes(
        PlanResult(success=False, positions=positions),
        case,
        _MetricRobot(),
        "arm",
        validation_samples=8,
        position_threshold_m=1.0e-4,
        rotation_threshold_rad=1.0e-4,
        joint_limit_tolerance_rad=1.0e-5,
    )
    metadata = [
        PlannerMetadata(
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="curobo",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]
    measured = TrialRecord(
        suite_version="test_v1",
        track="free-space-common",
        scenario_id="reach",
        case_id="case-1",
        algorithm_id="curobo",
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        model_revision="curobo-v2",
        planner_config_hash="abc",
        seed=11,
        repeat=0,
        batch_size=1,
        waypoint_count=1,
        path_shape="direct",
        start_state_bin="nominal",
        phase=TrialPhase.MEASURED,
        cost_time_ms=10.0,
        outcomes=outcomes,
    )

    aggregates = aggregate_results([measured], metadata, [case], measured_trials=1)
    row = aggregates["success_and_metrics"][0]

    assert row["success_rate"] == pytest.approx(1.0)
    assert row["planning_success_rate"] == pytest.approx(0.0)
    assert row["top_failure"] is None
    assert row["start_state_bin"] == "nominal"


def test_nmg_precision_and_external_accuracy_are_independently_configurable():
    suite = load_suite("smoke")
    _apply_overrides(
        suite,
        position_threshold_m=0.02,
        rotation_threshold_rad=0.10,
        nmg_pos_eps=0.03,
        nmg_rot_eps=0.20,
    )
    nmg = next(spec for spec in suite.planners if spec.id == "nmg")

    assert suite.protocol.position_threshold_m == pytest.approx(0.02)
    assert suite.protocol.rotation_threshold_rad == pytest.approx(0.10)
    assert nmg.config["pos_eps"] == pytest.approx(0.03)
    assert nmg.config["rot_eps"] == pytest.approx(0.20)


@pytest.mark.parametrize("override", [{"nmg_pos_eps": 0.0}, {"nmg_rot_eps": -0.1}])
def test_nmg_precision_rejects_non_positive_values(override):
    suite = load_suite("smoke")

    with pytest.raises(ValueError, match="NMG"):
        _apply_overrides(suite, **override)


class _FakeRobot:
    device = torch.device("cpu")

    def get_qpos_limits(self, name: str):  # noqa: ARG002
        lower = torch.tensor([-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8])
        upper = torch.tensor([2.8, 1.7, 2.8, -0.05, 2.8, 3.7, 2.8])
        return torch.stack([lower, upper], dim=-1).unsqueeze(0)

    def compute_fk(
        self, qpos: torch.Tensor, name: str, to_matrix: bool
    ):  # noqa: ARG002
        poses = torch.eye(4).repeat(qpos.shape[0], 1, 1)
        poses[:, :3, 3] = qpos[:, :3]
        return poses


def test_suite_loads_tracks_and_keeps_mutable_free_space_config():
    suite = load_suite("smoke")

    assert [track.id for track in suite.enabled_tracks()] == ["free-space-common"]
    assert suite.enabled_tracks()[0].scenario == "free_space"
    assert "free_space" in scenario_provider_names()
    provider = create_scenario_provider("free_space")
    assert provider.batch_sizes(suite, suite.enabled_tracks()[0]) == [1]

    suite.free_space.batch_sizes = [1, 8]
    suite.validate_benchmark()
    assert suite.enabled_tracks()[0].config["batch_sizes"] == [1, 8]


def test_free_space_manifest_is_seed_stable_and_algorithm_independent():
    suite = load_suite("smoke")
    robot = _FakeRobot()
    provider = create_scenario_provider("free_space")
    track = suite.enabled_tracks()[0]

    first = provider.generate_cases(suite, track, robot, "arm", batch_size=1)
    second = provider.generate_cases(suite, track, robot, "arm", batch_size=1)

    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert torch.equal(first[0].start_qpos, second[0].start_qpos)
    assert torch.equal(first[0].target_waypoints, second[0].target_waypoints)
    assert {case.track for case in first} == {"free-space-common"}


def test_free_space_cases_use_one_start_state_bin_each():
    suite = load_suite("coverage")
    suite.free_space.batch_sizes = [2]
    suite.free_space.waypoint_counts = [1]
    suite.free_space.path_shapes = ["direct"]
    suite.free_space.seeds = [11]
    suite.free_space.start_state_bins = ["nominal", "near_limit"]
    track = suite.enabled_tracks()[0]
    cases = create_scenario_provider("free_space").generate_cases(
        suite, track, _FakeRobot(), "arm", batch_size=2
    )

    assert [case.start_state_bin for case in cases] == ["nominal", "near_limit"]
    assert len({case.case_id for case in cases}) == 2


def test_success_metrics_are_stratified_by_start_state_bin():
    metadata = [
        PlannerMetadata(
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="curobo",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]
    cases = [
        BenchmarkCase(
            suite_version="test_v1",
            track="free-space-common",
            scenario_id="reach",
            case_id="nominal-case",
            seed=11,
            batch_size=1,
            num_waypoints=1,
            path_shape="direct",
            start_state_bin="nominal",
            start_qpos=torch.zeros(1, 7),
            target_waypoints=torch.eye(4).reshape(1, 1, 4, 4),
            reference_qpos=torch.zeros(1, 1, 7),
        ),
        BenchmarkCase(
            suite_version="test_v1",
            track="free-space-common",
            scenario_id="reach",
            case_id="limit-case",
            seed=11,
            batch_size=1,
            num_waypoints=1,
            path_shape="direct",
            start_state_bin="near_limit",
            start_qpos=torch.zeros(1, 7),
            target_waypoints=torch.eye(4).reshape(1, 1, 4, 4),
            reference_qpos=torch.zeros(1, 1, 7),
        ),
    ]
    records = [
        TrialRecord(
            suite_version="test_v1",
            track="free-space-common",
            scenario_id="reach",
            case_id=cases[0].case_id,
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            model_revision="curobo-v2",
            planner_config_hash="abc",
            seed=11,
            repeat=0,
            batch_size=1,
            waypoint_count=1,
            path_shape="direct",
            start_state_bin="nominal",
            phase=TrialPhase.MEASURED,
            cost_time_ms=10.0,
            outcomes=(_outcome(),),
        ),
        TrialRecord(
            suite_version="test_v1",
            track="free-space-common",
            scenario_id="reach",
            case_id=cases[1].case_id,
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            model_revision="curobo-v2",
            planner_config_hash="abc",
            seed=11,
            repeat=0,
            batch_size=1,
            waypoint_count=1,
            path_shape="direct",
            start_state_bin="near_limit",
            phase=TrialPhase.MEASURED,
            cost_time_ms=12.0,
            outcomes=(
                CaseOutcome(
                    env_index=0,
                    planning_success=False,
                    finite=True,
                    ordered_waypoints_reached=False,
                    motion_valid=False,
                    completed_waypoint_ratio=0.0,
                    final_translation_err_mm=None,
                    final_rotation_err_deg=None,
                    waypoint_translation_err_mm_mean=None,
                    waypoint_translation_err_mm_p95=None,
                    waypoint_translation_err_mm_max=None,
                    waypoint_rotation_err_deg_mean=None,
                    waypoint_rotation_err_deg_p95=None,
                    waypoint_rotation_err_deg_max=None,
                    joint_limit_violation=False,
                    max_normalized_joint_violation=0.0,
                    joint_path_length_rad=None,
                    cartesian_path_length_m=None,
                    path_efficiency=None,
                    failure_code="waypoint_miss",
                ),
            ),
        ),
    ]

    rows = aggregate_results(records, metadata, cases, measured_trials=1)[
        "success_and_metrics"
    ]
    by_bin = {row["start_state_bin"]: row for row in rows}

    assert set(by_bin) == {"nominal", "near_limit"}
    assert by_bin["nominal"]["success_rate"] == pytest.approx(1.0)
    assert by_bin["near_limit"]["success_rate"] == pytest.approx(0.0)
    assert by_bin["near_limit"]["top_failure"] == "waypoint_miss"


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        suite_version="test_v1",
        track="free-space-common",
        scenario_id="reach",
        case_id="case-1",
        seed=11,
        batch_size=1,
        num_waypoints=1,
        path_shape="direct",
        start_state_bin="nominal",
        start_qpos=torch.zeros(1, 7),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4),
        reference_qpos=torch.zeros(1, 1, 7),
    )


def _outcome() -> CaseOutcome:
    return CaseOutcome(
        env_index=0,
        planning_success=True,
        finite=True,
        ordered_waypoints_reached=True,
        motion_valid=True,
        completed_waypoint_ratio=1.0,
        final_translation_err_mm=1.0,
        final_rotation_err_deg=2.0,
        waypoint_translation_err_mm_mean=1.0,
        waypoint_translation_err_mm_p95=1.0,
        waypoint_translation_err_mm_max=1.0,
        waypoint_rotation_err_deg_mean=2.0,
        waypoint_rotation_err_deg_p95=2.0,
        waypoint_rotation_err_deg_max=2.0,
        joint_limit_violation=False,
        max_normalized_joint_violation=0.0,
        joint_path_length_rad=0.2,
        cartesian_path_length_m=0.1,
        path_efficiency=1.0,
    )


def _record(phase: TrialPhase, cost: float) -> TrialRecord:
    return TrialRecord(
        suite_version="test_v1",
        track="free-space-common",
        scenario_id="reach",
        case_id="case-1",
        algorithm_id="curobo",
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        model_revision="curobo-v2",
        planner_config_hash="abc",
        seed=11,
        repeat=0,
        batch_size=1,
        waypoint_count=1,
        path_shape="direct",
        start_state_bin="nominal",
        phase=phase,
        cost_time_ms=cost,
        cpu_delta_mb=1.0,
        gpu_delta_mb=2.0,
        peak_gpu_mb=3.0,
        outcomes=(_outcome(),) if phase is TrialPhase.MEASURED else (),
    )


def test_aggregation_excludes_warmup_and_keeps_unavailable_algorithm():
    metadata = [
        PlannerMetadata(
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="curobo",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        ),
        PlannerMetadata(
            algorithm_id="nmg",
            algorithm_role=AlgorithmRole.CANDIDATE,
            adapter="neural_stub",
            config_hash="def",
            capabilities=frozenset({"eef_waypoint"}),
        ),
    ]
    aggregates = aggregate_results(
        [_record(TrialPhase.WARMUP, 999.0), _record(TrialPhase.MEASURED, 10.0)],
        metadata,
        [_case()],
        measured_trials=1,
    )

    perf = next(
        row for row in aggregates["time_and_memory"] if row["algorithm"] == "curobo"
    )
    assert perf["cost_time_ms"] == pytest.approx(10.0)
    leaderboard = aggregates["leaderboard"]
    assert {row["algorithm"] for row in leaderboard} == {"curobo", "nmg"}
    nmg = next(row for row in leaderboard if row["algorithm"] == "nmg")
    assert nmg["eligible"] is False
    assert nmg["coverage_rate"] == pytest.approx(0.0)


def test_leaderboard_uses_case_macro_average_not_env_micro_average():
    """B=64 failures must not dominate a B=1 success on the leaderboard."""
    metadata = [
        PlannerMetadata(
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="curobo",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]
    small = BenchmarkCase(
        suite_version="test_v1",
        track="free-space-common",
        scenario_id="reach",
        case_id="case-b1",
        seed=11,
        batch_size=1,
        num_waypoints=1,
        path_shape="direct",
        start_state_bin="nominal",
        start_qpos=torch.zeros(1, 7),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4),
        reference_qpos=torch.zeros(1, 1, 7),
    )
    large = BenchmarkCase(
        suite_version="test_v1",
        track="free-space-common",
        scenario_id="reach",
        case_id="case-b64",
        seed=23,
        batch_size=64,
        num_waypoints=1,
        path_shape="direct",
        start_state_bin="nominal",
        start_qpos=torch.zeros(64, 7),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4).expand(64, 1, 4, 4).clone(),
        reference_qpos=torch.zeros(64, 1, 7),
    )
    failed = CaseOutcome(
        env_index=0,
        planning_success=False,
        finite=True,
        ordered_waypoints_reached=False,
        motion_valid=False,
        completed_waypoint_ratio=0.0,
        final_translation_err_mm=None,
        final_rotation_err_deg=None,
        waypoint_translation_err_mm_mean=None,
        waypoint_translation_err_mm_p95=None,
        waypoint_translation_err_mm_max=None,
        waypoint_rotation_err_deg_mean=None,
        waypoint_rotation_err_deg_p95=None,
        waypoint_rotation_err_deg_max=None,
        joint_limit_violation=False,
        max_normalized_joint_violation=0.0,
        joint_path_length_rad=None,
        cartesian_path_length_m=None,
        path_efficiency=None,
        failure_code="waypoint_miss",
    )
    records = [
        TrialRecord(
            suite_version="test_v1",
            track="free-space-common",
            scenario_id="reach",
            case_id=small.case_id,
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            model_revision="curobo-v2",
            planner_config_hash="abc",
            seed=small.seed,
            repeat=0,
            batch_size=1,
            waypoint_count=1,
            path_shape="direct",
            start_state_bin="nominal",
            phase=TrialPhase.MEASURED,
            cost_time_ms=10.0,
            outcomes=(_outcome(),),
        ),
        TrialRecord(
            suite_version="test_v1",
            track="free-space-common",
            scenario_id="reach",
            case_id=large.case_id,
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            model_revision="curobo-v2",
            planner_config_hash="abc",
            seed=large.seed,
            repeat=0,
            batch_size=64,
            waypoint_count=1,
            path_shape="direct",
            start_state_bin="nominal",
            phase=TrialPhase.MEASURED,
            cost_time_ms=40.0,
            outcomes=tuple(
                replace(failed, env_index=env_index) for env_index in range(64)
            ),
        ),
    ]

    row = aggregate_results(records, metadata, [small, large], measured_trials=1)[
        "leaderboard"
    ][0]

    # Env micro-average would be 1/65 ≈ 0.015; case macro-average is 0.5.
    assert row["overall_success_rate"] == pytest.approx(0.5)
    assert row["motion_valid_rate"] == pytest.approx(0.5)
    assert row["planning_success_rate"] == pytest.approx(0.5)
    assert row["coverage_rate"] == pytest.approx(1.0)
    assert row["eligible"] is True


def test_cold_plan_ms_only_attaches_to_matching_waypoint_row():
    """Cold latency from W=1 must not be copied onto W=5 Time & Memory rows."""
    metadata = [
        PlannerMetadata(
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="curobo",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]
    case_w1 = BenchmarkCase(
        suite_version="test_v1",
        track="free-space-common",
        scenario_id="reach",
        case_id="case-w1",
        seed=11,
        batch_size=1,
        num_waypoints=1,
        path_shape="direct",
        start_state_bin="nominal",
        start_qpos=torch.zeros(1, 7),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4),
        reference_qpos=torch.zeros(1, 1, 7),
    )
    case_w5 = BenchmarkCase(
        suite_version="test_v1",
        track="free-space-common",
        scenario_id="reach",
        case_id="case-w5",
        seed=11,
        batch_size=1,
        num_waypoints=5,
        path_shape="direct",
        start_state_bin="nominal",
        start_qpos=torch.zeros(1, 7),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4).expand(1, 5, 4, 4).clone(),
        reference_qpos=torch.zeros(1, 5, 7),
    )

    def _measured(case: BenchmarkCase, cost: float) -> TrialRecord:
        return TrialRecord(
            suite_version="test_v1",
            track="free-space-common",
            scenario_id="reach",
            case_id=case.case_id,
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            model_revision="curobo-v2",
            planner_config_hash="abc",
            seed=case.seed,
            repeat=0,
            batch_size=case.batch_size,
            waypoint_count=case.num_waypoints,
            path_shape=case.path_shape,
            start_state_bin=case.start_state_bin,
            phase=TrialPhase.MEASURED,
            cost_time_ms=cost,
            outcomes=(_outcome(),),
        )

    records = [
        TrialRecord(
            suite_version="test_v1",
            track="free-space-common",
            scenario_id="reach",
            case_id=case_w1.case_id,
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            model_revision="curobo-v2",
            planner_config_hash="abc",
            seed=case_w1.seed,
            repeat=-1,
            batch_size=1,
            waypoint_count=1,
            path_shape="direct",
            start_state_bin="nominal",
            phase=TrialPhase.COLD,
            cost_time_ms=123.0,
        ),
        TrialRecord(
            suite_version="test_v1",
            track="free-space-common",
            scenario_id="reach",
            case_id=case_w1.case_id,
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            model_revision="curobo-v2",
            planner_config_hash="abc",
            seed=case_w1.seed,
            repeat=-1,
            batch_size=1,
            waypoint_count=1,
            path_shape="direct",
            start_state_bin="nominal",
            phase=TrialPhase.CONSTRUCT,
            cost_time_ms=50.0,
        ),
        _measured(case_w1, 10.0),
        _measured(case_w5, 20.0),
    ]

    rows = aggregate_results(records, metadata, [case_w1, case_w5], measured_trials=1)[
        "time_and_memory"
    ]
    by_waypoints = {row["waypoint_count"]: row for row in rows}

    assert by_waypoints[1]["cold_plan_ms"] == pytest.approx(123.0)
    assert by_waypoints[5]["cold_plan_ms"] is None
    # One-time construct cost remains visible on every batch row.
    assert by_waypoints[1]["planner_construct_ms"] == pytest.approx(50.0)
    assert by_waypoints[5]["planner_construct_ms"] == pytest.approx(50.0)


def test_report_contains_exactly_three_markdown_tables(tmp_path):
    suite = load_suite("smoke")
    aggregates = {
        "time_and_memory": [],
        "success_and_metrics": [],
        "leaderboard": [],
    }

    report = write_markdown_report(tmp_path / "report.md", suite, aggregates)
    text = report.read_text(encoding="utf-8")

    assert text.count("\n| ---") == 3
    assert text.count("## Time & Memory") == 1
    assert text.count("## Success & Other Metrics") == 1
    assert text.count("## Leaderboard") == 1


def test_curobo_prepare_backend_exposes_actual_graph_mode():
    planner = object.__new__(CuroboPlanner)
    planner.robot = Mock(num_instances=8)
    planner.cfg = Mock(world=Mock(multi_env=False))
    backend = Mock(
        control_part="arm",
        batch_size=8,
        planning_mode=MoveType.EEF_MOVE,
        use_cuda_graph=False,
    )
    planner._get_backend = Mock(return_value=backend)

    result = planner.prepare_backend(
        control_part="arm", batch_size=8, move_type=MoveType.EEF_MOVE
    )

    planner._get_backend.assert_called_once_with("arm", 8, MoveType.EEF_MOVE)
    assert result["use_cuda_graph"] is False
    assert result["batch_size"] == 8


def test_metric_rows_use_success_rate_and_null_peak_gpu():
    metadata = [
        PlannerMetadata(
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="curobo",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]
    measured = TrialRecord(
        suite_version="test_v1",
        track="free-space-common",
        scenario_id="reach",
        case_id="case-1",
        algorithm_id="curobo",
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        model_revision="curobo-v2",
        planner_config_hash="abc",
        seed=11,
        repeat=0,
        batch_size=1,
        waypoint_count=1,
        path_shape="direct",
        start_state_bin="nominal",
        phase=TrialPhase.MEASURED,
        cost_time_ms=10.0,
        peak_gpu_mb=None,
        outcomes=(_outcome(),),
    )
    aggregates = aggregate_results([measured], metadata, [_case()], measured_trials=1)

    assert aggregates["success_and_metrics"][0]["success_rate"] == pytest.approx(1.0)
    assert "motion_valid_rate" not in aggregates["success_and_metrics"][0]
    assert aggregates["time_and_memory"][0]["peak_gpu_mb"] is None
    assert aggregates["leaderboard"][0]["peak_gpu_mb"] is None


def test_toppra_adapter_close_releases_planner():
    from scripts.benchmark.motion_generation.config import PlannerSpecCfg
    from scripts.benchmark.motion_generation.planners.base import PlannerContext
    from scripts.benchmark.motion_generation.planners.toppra import ToppraAdapter

    planner = Mock()
    adapter = ToppraAdapter(
        PlannerSpecCfg(
            id="toppra",
            adapter="toppra",
            role=AlgorithmRole.DIAGNOSTIC_BASELINE.value,
        ),
        PlannerContext(
            robot=Mock(),
            control_part="arm",
            device=torch.device("cpu"),
            sample_interval=40,
        ),
    )
    adapter.motion_generator = Mock(planner=planner)

    adapter.close()

    planner.close.assert_called_once_with()
    assert adapter.motion_generator is None
