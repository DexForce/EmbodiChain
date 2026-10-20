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

"""Tests for versioned benchmark domains and recomputable statistics."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from embodichain.benchmarks.motion_generation.aggregation import aggregate_results
from embodichain.benchmarks.motion_generation.config import SuiteCfg
from embodichain.benchmarks.motion_generation.contracts import EvaluationDomain
from embodichain.benchmarks.motion_generation.generalization import (
    aggregate_generalization,
)
from embodichain.benchmarks.motion_generation.models import (
    AlgorithmRole,
    BenchmarkCase,
    CaseOutcome,
    PlannerMetadata,
    TrialPhase,
    TrialRecord,
)


def _case(case_id: str, domain: EvaluationDomain) -> BenchmarkCase:
    return BenchmarkCase(
        suite_version="generalization_v1",
        track=f"pick@{domain.id}",
        track_group="pick",
        scenario_id="pick_up",
        case_id=case_id,
        seed=11,
        batch_size=1,
        num_waypoints=1,
        path_shape="direct",
        start_state_bin="nominal",
        start_qpos=torch.zeros(1, 7),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4),
        reference_qpos=torch.zeros(1, 1, 7),
        skill_id="pick_up",
        primary_success="task_success",
        domain=domain,
    )


def _outcome(
    success: bool,
    *,
    failure_stage: str | None = None,
) -> CaseOutcome:
    return CaseOutcome(
        env_index=0,
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
        task_success=success,
        failure_stage=failure_stage,
    )


def _record(case: BenchmarkCase, outcome: CaseOutcome) -> TrialRecord:
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
        waypoint_count=case.num_waypoints,
        path_shape=case.path_shape,
        start_state_bin=case.start_state_bin,
        phase=TrialPhase.MEASURED,
        robot_id=case.robot_id,
        skill_id=case.skill_id,
        primary_success=case.primary_success,
        domain=case.domain,
        outcomes=(outcome,),
    )


def test_suite_expands_domains_without_changing_the_case_provider_schema() -> None:
    suite = SuiteCfg.from_dict(
        {
            "planners": [{"id": "baseline", "adapter": "ik_interpolate"}],
            "embodiment": {
                "component": "components/embodiments/franka_panda.yaml",
                "overrides": {"simulation": {"uid": "test"}},
            },
            "tracks": [
                {
                    "id": "pick",
                    "scenario": "free_space",
                    "domains": [
                        {"id": "nominal", "version": "v1", "kind": "nominal"},
                        {
                            "id": "objects",
                            "version": "v1",
                            "kind": "held_out",
                            "objects": ["unseen_box"],
                        },
                    ],
                }
            ],
        }
    )

    tracks = suite.evaluation_tracks()

    assert [track.id for track in tracks] == ["pick@nominal", "pick@objects"]
    assert [track.group_id for track in tracks] == ["pick", "pick"]
    assert tracks[1].domain.objects == ["unseen_box"]
    assert suite.resolved_embodiment_id.endswith("franka_panda.yaml")


def test_suite_rejects_colliding_expanded_track_ids() -> None:
    suite = SuiteCfg.from_dict(
        {
            "planners": [{"id": "baseline", "adapter": "ik_interpolate"}],
            "tracks": [
                {
                    "id": "foo",
                    "scenario": "free_space",
                    "domains": [
                        {"id": "x", "version": "v1", "kind": "nominal"},
                        {"id": "y", "version": "v1", "kind": "robustness"},
                    ],
                },
                {"id": "foo@x", "scenario": "free_space"},
            ],
        }
    )

    with pytest.raises(ValueError, match="Expanded benchmark track ids"):
        suite.evaluation_tracks()


def test_suite_validates_atomic_domain_overrides_before_execution() -> None:
    suite = SuiteCfg.from_dict(
        {
            "planners": [{"id": "baseline", "adapter": "ik_interpolate"}],
            "tracks": [
                {
                    "id": "atomic",
                    "scenario": "atomic_skill",
                    "config": {"batch_sizes": [1], "seeds": [1]},
                    "domains": [
                        {
                            "id": "bad",
                            "version": "v1",
                            "kind": "robustness",
                            "overrides": {
                                "randomization": {
                                    "enabled": True,
                                    "mode": "none",
                                    "pose_batch_size": 2,
                                    "object_translation_jitter_m": [0.1, 0.0, 0.0],
                                }
                            },
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="mode='none'"):
        suite.evaluation_tracks()


def test_suite_preserves_multiple_free_space_track_configs() -> None:
    common = {
        "waypoint_counts": [1],
        "path_shapes": ["direct"],
        "start_state_bins": ["nominal"],
        "seeds": [1],
        "target_types": ["eef"],
    }
    suite = SuiteCfg.from_dict(
        {
            "planners": [{"id": "baseline", "adapter": "ik_interpolate"}],
            "tracks": [
                {
                    "id": "small",
                    "scenario": "free_space",
                    "config": {**common, "batch_sizes": [1]},
                },
                {
                    "id": "large",
                    "scenario": "free_space",
                    "config": {**common, "batch_sizes": [8]},
                },
            ],
        }
    )

    assert [track.config["batch_sizes"] for track in suite.tracks] == [[1], [8]]


def test_suite_ignores_disabled_free_space_tracks_when_syncing_defaults() -> None:
    suite = SuiteCfg.from_dict(
        {
            "planners": [{"id": "baseline", "adapter": "ik_interpolate"}],
            "tracks": [
                {
                    "id": "enabled",
                    "scenario": "free_space",
                    "config": {"batch_sizes": [8]},
                },
                {
                    "id": "disabled",
                    "scenario": "free_space",
                    "enabled": False,
                },
            ],
        }
    )

    assert suite.free_space.batch_sizes == [8]
    assert suite.tracks[0].config["batch_sizes"] == [8]
    assert [track.id for track in suite.evaluation_tracks()] == ["enabled"]


def test_generalization_excludes_unsupported_cases_but_reports_coverage() -> None:
    nominal = EvaluationDomain(id="nominal", version="v1", kind="nominal")
    held_out = EvaluationDomain(id="objects", version="v1", kind="held_out")
    nominal_case = _case("nominal-case", nominal)
    held_out_case = replace(_case("held-out-case", held_out), track="pick@objects")
    metadata = [
        PlannerMetadata(
            algorithm_id="baseline",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="ik_interpolate",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]
    unsupported = TrialRecord(
        **{
            **_record(held_out_case, _outcome(False)).__dict__,
            "phase": TrialPhase.AVAILABILITY,
            "status": "unsupported",
            "failure_code": "unsupported_capability",
            "outcomes": (),
        }
    )
    rows = aggregate_generalization(
        [_record(nominal_case, _outcome(True)), unsupported],
        metadata,
        [nominal_case, held_out_case],
        measured_trials=1,
    )

    held_out_row = next(row for row in rows if row["domain_kind"] == "held_out")

    assert held_out_row["coverage_rate"] == pytest.approx(0.0)
    assert held_out_row["success_rate"] is None
    assert held_out_row["unsupported_cases"] == 1
    assert held_out_row["generalization_eligible"] is False


def test_generalization_counts_missing_repeats_as_failures() -> None:
    domain = EvaluationDomain(id="objects", version="v1", kind="held_out")
    case = _case("held-out-case", domain)
    metadata = [
        PlannerMetadata(
            algorithm_id="baseline",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="ik_interpolate",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]

    rows = aggregate_generalization(
        [_record(case, _outcome(True))],
        metadata,
        [case],
        measured_trials=2,
    )

    assert rows[0]["coverage_rate"] == pytest.approx(0.5)
    assert rows[0]["success_rate"] == pytest.approx(0.5)
    assert rows[0]["domain_mean_success_rate"] == pytest.approx(0.5)
    assert rows[0]["domain_worst_success_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize("effective_algorithm", [False, True])
@pytest.mark.parametrize(
    "nominal_state",
    ["absent", "missing", "partial", "unsupported", "failed", "complete"],
)
def test_generalization_requires_a_complete_nominal_baseline(
    nominal_state: str, effective_algorithm: bool
) -> None:
    """A complete OOD run cannot borrow eligibility from an incomplete baseline."""
    nominal = _case(
        "nominal", EvaluationDomain(id="nominal", kind="nominal", version="v1")
    )
    held_out = _case(
        "objects", EvaluationDomain(id="objects", kind="held_out", version="v1")
    )
    cases = [held_out]
    records = [replace(_record(held_out, _outcome(True)), repeat=i) for i in range(2)]
    if nominal_state != "absent":
        cases.append(nominal)
    if nominal_state in {"partial", "failed", "complete"}:
        repeats = 1 if nominal_state == "partial" else 2
        records.extend(
            replace(_record(nominal, _outcome(nominal_state != "failed")), repeat=i)
            for i in range(repeats)
        )
    elif nominal_state == "unsupported":
        records.append(
            replace(
                _record(nominal, _outcome(False)),
                phase=TrialPhase.AVAILABILITY,
                status="unsupported",
                failure_code="unsupported_capability",
                outcomes=(),
            )
        )
    algorithm = "baseline"
    if effective_algorithm:
        algorithm = "sequential_ik_interpolation"
        records = [
            replace(
                record,
                metadata={
                    "selected_planner_backend_used": False,
                    "effective_motion_algorithm": algorithm,
                },
            )
            for record in records
        ]
    metadata = [
        PlannerMetadata(
            algorithm_id="baseline",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="ik_interpolate",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]

    rows = aggregate_generalization(records, metadata, cases, measured_trials=2)
    row = next(
        row
        for row in rows
        if row["algorithm"] == algorithm and row["domain_kind"] == "held_out"
    )

    assert row["coverage_rate"] == pytest.approx(1.0)
    assert row["success_rate"] == pytest.approx(1.0)
    assert row["generalization_eligible"] is (nominal_state in {"failed", "complete"})
    if nominal_state == "complete":
        assert row["domain_retention"] == pytest.approx(1.0)
    else:
        # Missing baseline evidence must not inflate retention; a measured zero
        # baseline is valid for eligibility but cannot be used as a divisor.
        assert row["domain_retention"] is None


@pytest.mark.parametrize("held_out_repeats", [None, 0, 1])
def test_nominal_results_alone_do_not_qualify_for_generalization(
    held_out_repeats: int | None,
) -> None:
    """OOD populations must exist and complete their expected measured repeats."""
    nominal = _case(
        "nominal", EvaluationDomain(id="nominal", kind="nominal", version="v1")
    )
    held_out = _case(
        "objects", EvaluationDomain(id="objects", kind="held_out", version="v1")
    )
    cases = [nominal]
    records = [replace(_record(nominal, _outcome(True)), repeat=i) for i in range(2)]
    if held_out_repeats is not None:
        cases.append(held_out)
        records.extend(
            replace(_record(held_out, _outcome(True)), repeat=i)
            for i in range(held_out_repeats)
        )
    metadata = [
        PlannerMetadata(
            algorithm_id="baseline",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="ik_interpolate",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]

    rows = aggregate_generalization(records, metadata, cases, measured_trials=2)

    assert all(row["generalization_eligible"] is False for row in rows)


def test_generalization_reports_row_level_spatial_bucket_rates() -> None:
    """Near/mid/far shell rows retain independent success summaries."""
    domain = EvaluationDomain(id="workspace_shell", version="v1", kind="robustness")
    case = replace(
        _case("shell-case", domain),
        track="pick@workspace_shell",
        batch_size=3,
        start_qpos=torch.zeros(3, 7),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4).repeat(3, 1, 1, 1),
        reference_qpos=torch.zeros(3, 1, 7),
        case_parameters={"randomization": {"spatial_buckets": ["near", "mid", "far"]}},
    )
    record = replace(
        _record(case, _outcome(True)),
        batch_size=3,
        outcomes=(
            replace(_outcome(True), env_index=0),
            replace(_outcome(False, failure_stage="physical_release"), env_index=1),
            replace(_outcome(True), env_index=2),
        ),
    )
    metadata = [
        PlannerMetadata(
            algorithm_id="baseline",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="ik_interpolate",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]

    row = aggregate_generalization([record], metadata, [case], measured_trials=1)[0]

    assert row["spatial_bucket_success_rates"] == {
        "far": 1.0,
        "mid": 0.0,
        "near": 1.0,
    }
    assert row["spatial_bucket_counts"] == {"far": 1, "mid": 1, "near": 1}
    assert row["spatial_bucket_failure_stages"] == {"mid": {"physical_release": 1}}


def test_generalization_reports_bypassed_cases_under_effective_algorithm() -> None:
    domain = EvaluationDomain(id="objects", version="v1", kind="held_out")
    attributed_case = replace(_case("planner-case", domain), track="pick@objects")
    bypassed_case = replace(
        attributed_case,
        case_id="ik-interpolation-case",
        seed=23,
        case_parameters={"randomization": {"spatial_buckets": ["near"]}},
    )
    attributed_record = _record(attributed_case, _outcome(False))
    bypassed_record = replace(
        _record(bypassed_case, _outcome(True)),
        metadata={
            "selected_planner_backend_used": False,
            "effective_motion_algorithm": "sequential_ik_interpolation",
        },
    )
    metadata = [
        PlannerMetadata(
            algorithm_id="baseline",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="ik_interpolate",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]

    aggregates = aggregate_results(
        [attributed_record, bypassed_record],
        metadata,
        [attributed_case, bypassed_case],
        measured_trials=1,
        track_scenarios={
            case.track: "atomic_skill" for case in [attributed_case, bypassed_case]
        },
    )
    rows = aggregates["generalization"]

    assert len(rows) == 2
    planner_row = next(row for row in rows if row["algorithm"] == "baseline")
    assert planner_row["cases"] == 2
    assert planner_row["attributed_cases"] == 1
    assert planner_row["applicable_cases"] == 1
    assert planner_row["bypassed_cases"] == 1
    assert planner_row["coverage_rate"] == pytest.approx(1.0)
    assert planner_row["success_rate"] == pytest.approx(0.0)
    effective_row = next(
        row for row in rows if row["algorithm"] == "sequential_ik_interpolation"
    )
    assert effective_row["algorithm_role"] == "effective_motion_algorithm"
    assert effective_row["cases"] == 1
    assert effective_row["attributed_cases"] == 1
    assert effective_row["bypassed_cases"] == 0
    assert effective_row["coverage_rate"] == pytest.approx(1.0)
    assert effective_row["success_rate"] == pytest.approx(1.0)
    assert effective_row["spatial_bucket_success_rates"] == {"near": 1.0}
    assert effective_row["spatial_bucket_counts"] == {"near": 1}
    effective_metrics = next(
        row
        for row in aggregates["atomic_skills_success_and_metrics"]
        if row["algorithm"] == "sequential_ik_interpolation"
    )
    assert effective_metrics["algorithm_role"] == "effective_motion_algorithm"
    assert effective_metrics["skill"] == "pick_up"
    assert effective_metrics["success_rate"] == pytest.approx(1.0)
    leaderboard = next(
        row
        for row in aggregates["atomic_skills_leaderboard"]
        if row["algorithm"] == "baseline"
    )
    assert leaderboard["attributed_cases"] == 1
    assert leaderboard["bypassed_cases"] == 1
    bypassed_rows = aggregate_generalization(
        [bypassed_record], metadata, [bypassed_case], measured_trials=1
    )
    assert len(bypassed_rows) == 2
    bypassed_planner_row = next(
        row for row in bypassed_rows if row["algorithm"] == "baseline"
    )
    assert bypassed_planner_row["cases"] == 1
    assert bypassed_planner_row["attributed_cases"] == 0
    assert bypassed_planner_row["applicable_cases"] == 0
    assert bypassed_planner_row["bypassed_cases"] == 1
    assert bypassed_planner_row["coverage_rate"] == pytest.approx(0.0)
    assert bypassed_planner_row["success_rate"] is None
    assert bypassed_planner_row["generalization_eligible"] is False
    bypassed_effective_row = next(
        row
        for row in bypassed_rows
        if row["algorithm"] == "sequential_ik_interpolation"
    )
    assert bypassed_effective_row["success_rate"] == pytest.approx(1.0)
    assert bypassed_effective_row["generalization_eligible"] is False


def test_effective_algorithm_combines_source_planners_once() -> None:
    """One effective row combines repeats from every bypassed source planner."""
    domain = EvaluationDomain(id="workspace_shell", version="v1", kind="robustness")
    case = replace(
        _case("shell-case", domain),
        track="pick@workspace_shell",
        case_parameters={"randomization": {"spatial_buckets": ["near"]}},
    )
    attribution = {
        "selected_planner_backend_used": False,
        "effective_motion_algorithm": "sequential_ik_interpolation",
    }
    baseline_record = replace(_record(case, _outcome(True)), metadata=attribution)
    candidate_record = replace(
        _record(case, _outcome(False, failure_stage="contact")),
        algorithm_id="candidate",
        algorithm_role=AlgorithmRole.CANDIDATE,
        metadata=attribution,
    )
    metadata = [
        PlannerMetadata(
            algorithm_id="baseline",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="ik_interpolate",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        ),
        PlannerMetadata(
            algorithm_id="candidate",
            algorithm_role=AlgorithmRole.CANDIDATE,
            adapter="curobo",
            config_hash="def",
            capabilities=frozenset({"eef_waypoint"}),
        ),
    ]

    aggregates = aggregate_results(
        [baseline_record, candidate_record],
        metadata,
        [case],
        measured_trials=1,
        track_scenarios={case.track: "atomic_skill"},
    )

    effective_rows = [
        row
        for row in aggregates["generalization"]
        if row["algorithm"] == "sequential_ik_interpolation"
    ]
    assert len(effective_rows) == 1
    assert effective_rows[0]["coverage_rate"] == pytest.approx(1.0)
    assert effective_rows[0]["success_rate"] == pytest.approx(0.5)
    assert effective_rows[0]["spatial_bucket_success_rates"] == {"near": 0.5}
    assert effective_rows[0]["spatial_bucket_counts"] == {"near": 2}
    effective_metrics = [
        row
        for row in aggregates["atomic_skills_success_and_metrics"]
        if row["algorithm"] == "sequential_ik_interpolation"
    ]
    assert len(effective_metrics) == 1
    assert effective_metrics[0]["coverage_rate"] == pytest.approx(1.0)
    assert effective_metrics[0]["success_rate"] == pytest.approx(0.5)


def test_spatial_bucket_rates_count_missing_repeats_as_failures() -> None:
    """Missing expected rows remain in every bucket's success denominator."""
    domain = EvaluationDomain(id="workspace_shell", version="v1", kind="robustness")
    case = replace(
        _case("shell-case", domain),
        track="pick@workspace_shell",
        batch_size=3,
        start_qpos=torch.zeros(3, 7),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4).repeat(3, 1, 1, 1),
        reference_qpos=torch.zeros(3, 1, 7),
        case_parameters={"randomization": {"spatial_buckets": ["near", "mid", "far"]}},
    )
    record = replace(
        _record(case, _outcome(True)),
        batch_size=3,
        outcomes=(
            replace(_outcome(True), env_index=0),
            replace(_outcome(True), env_index=2),
        ),
    )
    metadata = [
        PlannerMetadata(
            algorithm_id="baseline",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="ik_interpolate",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]

    row = aggregate_generalization([record], metadata, [case], measured_trials=2)[0]

    assert row["spatial_bucket_success_rates"] == {
        "far": 0.5,
        "mid": 0.0,
        "near": 0.5,
    }
    assert row["spatial_bucket_counts"] == {"far": 2, "mid": 2, "near": 2}
    assert row["spatial_bucket_failure_stages"] == {
        "far": {"missing_outcome": 1},
        "mid": {"missing_outcome": 2},
        "near": {"missing_outcome": 1},
    }
