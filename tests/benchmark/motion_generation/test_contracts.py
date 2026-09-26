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

"""Contracts for domain provenance and physical benchmark evaluation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from scripts.benchmark.motion_generation import contracts
from scripts.benchmark.motion_generation.artifacts import (
    TrialJsonlWriter,
    write_case_manifest,
)
from scripts.benchmark.motion_generation.config import (
    SuiteCfg,
    load_suite,
    suite_to_dict,
)
from scripts.benchmark.motion_generation.models import (
    AlgorithmRole,
    BenchmarkCase,
    CaseOutcome,
    TrialPhase,
    TrialRecord,
)


def _suite(domain: object = None) -> SuiteCfg:
    return SuiteCfg.from_dict(
        {
            "planners": [{"id": "baseline", "adapter": "ik_interpolate"}],
            "tracks": [{"id": "test", "scenario": "free_space", "domain": domain}],
        }
    )


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        suite_version="test_v1",
        track="test",
        scenario_id="free_space",
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


def test_domain_survives_resolved_suite_roundtrip(tmp_path: Path) -> None:
    suite = _suite({"id": "unseen_objects", "version": "v2", "kind": "held_out"})
    resolved = suite_to_dict(suite)
    assert resolved["tracks"][0]["domain"] == {
        "id": "unseen_objects",
        "version": "v2",
        "kind": "held_out",
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(resolved))
    assert load_suite(str(path)).tracks[
        0
    ].domain.to_identity() == contracts.EvaluationDomain(
        id="unseen_objects", version="v2", kind="held_out"
    )


def test_legacy_suite_is_not_implicitly_nominal() -> None:
    assert _suite().tracks[0].domain is None


@pytest.mark.parametrize(
    "domain",
    [
        {"id": "objects", "version": "", "kind": "held_out"},
        {"id": " ", "version": "v1", "kind": "nominal"},
        {"id": "objects", "version": "v1", "kind": "unknown"},
        {"id": "objects", "version": "v1", "kind": "nominal", "objects": ["box"]},
        "nominal",
    ],
)
def test_invalid_domain_is_rejected_before_execution(domain: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        _suite(domain)


def test_manifest_keeps_domain_and_is_independent_of_planner(tmp_path: Path) -> None:
    domain = contracts.EvaluationDomain(id="nominal", version="v1", kind="nominal")
    case = replace(_case(), domain=domain)
    path = write_case_manifest(tmp_path / "cases.json", [case])
    payload = json.loads(path.read_text())
    assert payload["case_schema_version"] == 3
    assert payload["cases"][0]["domain"] == {
        "id": "nominal",
        "version": "v1",
        "kind": "nominal",
    }
    assert "planner_config_hash" not in payload["cases"][0]


def test_stage_failure_and_unreached_stage_survive_jsonl(tmp_path: Path) -> None:
    stages = (
        contracts.StageOutcome(
            stage_id="grasp", status="failed", failure_code="object_not_grasped"
        ),
        contracts.StageOutcome(stage_id="lift", status="not_reached"),
    )
    outcome = CaseOutcome(
        env_index=0,
        planning_success=True,
        finite=True,
        ordered_waypoints_reached=True,
        motion_valid=True,
        completed_waypoint_ratio=1.0,
        final_translation_err_mm=None,
        final_rotation_err_deg=None,
        waypoint_translation_err_mm_mean=None,
        waypoint_translation_err_mm_p95=None,
        waypoint_translation_err_mm_max=None,
        waypoint_rotation_err_deg_mean=None,
        waypoint_rotation_err_deg_p95=None,
        waypoint_rotation_err_deg_max=None,
        joint_limit_violation=False,
        max_normalized_joint_violation=None,
        joint_path_length_rad=None,
        cartesian_path_length_m=None,
        path_efficiency=None,
        task_success=False,
        failure_code="object_not_grasped",
        stages=stages,
        failure_stage="grasp",
    )
    case = _case()
    record = TrialRecord(
        suite_version=case.suite_version,
        track=case.track,
        scenario_id=case.scenario_id,
        case_id=case.case_id,
        algorithm_id="baseline",
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        model_revision="N/A",
        planner_config_hash="abc",
        seed=11,
        repeat=0,
        batch_size=1,
        waypoint_count=1,
        path_shape="direct",
        start_state_bin="nominal",
        phase=TrialPhase.MEASURED,
        domain=contracts.EvaluationDomain(id="nominal", version="v1", kind="nominal"),
        outcomes=(outcome,),
    )
    path = tmp_path / "trials.jsonl"
    TrialJsonlWriter(path).append(record)
    row = json.loads(path.read_text())
    assert row["domain"]["version"] == "v1"
    assert row["outcomes"][0]["failure_stage"] == "grasp"
    assert [stage["status"] for stage in row["outcomes"][0]["stages"]] == [
        "failed",
        "not_reached",
    ]


def test_incomplete_and_inapplicable_stages_cannot_establish_success() -> None:
    for status in ("not_reached", "not_applicable"):
        result = contracts.PhysicalEvaluation(
            env_index=0,
            stages=(contracts.StageOutcome(stage_id="lift", status=status),),
        )
        assert result.task_success is False
    result = contracts.PhysicalEvaluation(
        env_index=0,
        stages=(
            contracts.StageOutcome(stage_id="grasp", status="not_applicable"),
            contracts.StageOutcome(
                stage_id="lift", status="passed", measurements={"height_m": 0.1}
            ),
        ),
    )
    assert result.task_success is True


@pytest.mark.parametrize(
    "stages",
    [
        (),
        (
            dict(stage_id="grasp", status="passed"),
            dict(stage_id="grasp", status="passed"),
        ),
        (
            dict(stage_id="grasp", status="failed", failure_code="miss"),
            dict(stage_id="lift", status="passed"),
        ),
        (
            dict(stage_id="grasp", status="not_reached"),
            dict(stage_id="lift", status="passed"),
        ),
    ],
)
def test_invalid_stage_chain_is_rejected(stages: tuple) -> None:
    with pytest.raises(ValueError):
        contracts.PhysicalEvaluation(
            env_index=0,
            stages=tuple(contracts.StageOutcome(**stage) for stage in stages),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(stage_id="grasp", status="failed"),
        dict(stage_id="grasp", status="passed", failure_code="miss"),
        dict(stage_id="", status="passed"),
        dict(stage_id="grasp", status="success"),
        dict(stage_id="grasp", status="passed", measurements={"height": float("nan")}),
    ],
)
def test_invalid_stage_evidence_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        contracts.StageOutcome(**kwargs)


def test_batch_results_are_ordered_by_environment_and_failure_is_local() -> None:
    passed = contracts.PhysicalEvaluation(
        env_index=0, stages=(contracts.StageOutcome(stage_id="lift", status="passed"),)
    )
    failed = contracts.PhysicalEvaluation(
        env_index=1,
        stages=(
            contracts.StageOutcome(
                stage_id="lift", status="failed", failure_code="dropped"
            ),
        ),
    )
    rows = contracts.validate_physical_evaluations((failed, passed), batch_size=2)
    assert [row.task_success for row in rows] == [True, False]
    assert rows[1].failure_stage == "lift"
    assert rows[1].failure_code == "dropped"
    for invalid in (
        (passed,),
        (passed, passed),
        (passed, replace(failed, env_index=2)),
    ):
        with pytest.raises(ValueError):
            contracts.validate_physical_evaluations(invalid, batch_size=2)


def test_batch_results_require_the_same_stage_order() -> None:
    rows = tuple(
        contracts.PhysicalEvaluation(
            env_index=index,
            stages=(contracts.StageOutcome(stage_id=stage, status="passed"),),
        )
        for index, stage in enumerate(("grasp", "lift"))
    )
    with pytest.raises(ValueError):
        contracts.validate_physical_evaluations(rows, batch_size=2)
