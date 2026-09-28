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

from scripts.benchmark.motion_generation.config import SuiteCfg
from scripts.benchmark.motion_generation.contracts import EvaluationDomain
from scripts.benchmark.motion_generation.generalization import aggregate_generalization
from scripts.benchmark.motion_generation.models import (
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


def _outcome(success: bool) -> CaseOutcome:
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
