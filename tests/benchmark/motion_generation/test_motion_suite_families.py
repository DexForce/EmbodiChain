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

"""Contracts separating motion matrices from Atomic integration populations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from embodichain.benchmarks.motion_generation.config import (
    BENCHMARK_ROOT,
    DomainCfg,
    SuiteCfg,
    load_suite,
)
from embodichain.benchmarks.motion_generation.models import BenchmarkCase
from embodichain.benchmarks.motion_generation.planners.base import (
    PlannerAdapter,
    PlannerContext,
)
from embodichain.benchmarks.motion_generation.scenarios.free_space import (
    FreeSpaceScenario,
)

pytestmark = pytest.mark.no_sim


class _ReachableRobot:
    """CPU kinematic stub with valid seven-joint limits and deterministic FK."""

    device = torch.device("cpu")

    def get_qpos_limits(self, *, name: str) -> torch.Tensor:
        """Return finite Panda-shaped position limits."""
        lower = torch.tensor([-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8])
        upper = torch.tensor([2.8, 1.7, 2.8, -0.05, 2.8, 3.7, 2.8])
        return torch.stack((lower, upper), dim=-1).unsqueeze(0)

    def compute_fk(
        self, *, qpos: torch.Tensor, name: str, to_matrix: bool
    ) -> torch.Tensor:
        """Map three joints into a diagnostic Cartesian translation."""
        poses = torch.eye(4).repeat(qpos.shape[0], 1, 1)
        poses[:, :3, 3] = qpos[:, :3]
        return poses


def _load_test_suite(tmp_path: Path, **matrix: Any) -> SuiteCfg:
    """Exercise the production YAML loader with a small reachable population."""
    values = {
        "suite_version": "legacy_v1",
        "planners": [{"id": "joint", "adapter": "joint", "enabled": True}],
        "protocol": {"report_schema_version": "3"},
        "free_space": {
            "batch_sizes": [2],
            "waypoint_counts": [3],
            "path_shapes": ["direct"],
            "start_state_bins": ["nominal"],
            "seeds": [11],
            **matrix,
        },
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return load_suite(str(path))


def _cases(suite: SuiteCfg) -> list[BenchmarkCase]:
    """Generate cases through the production provider without simulation."""
    track = suite.evaluation_tracks()[0]
    return FreeSpaceScenario().generate_cases(
        suite, track, _ReachableRobot(), "arm", batch_size=2
    )


def test_legacy_suite_retains_cartesian_population_and_case_identity(tmp_path):
    suite = _load_test_suite(tmp_path)
    cases = _cases(suite)

    assert suite.protocol.report_schema_version == "3"
    assert suite.free_space.target_types == ["eef"]
    assert len(cases) == 1
    assert cases[0].case_id == "free_space_60ef64d856839cbc"
    assert cases[0].scenario_id == "waypoint_path"


def test_joint_and_cartesian_cases_share_inputs_but_not_identity(tmp_path):
    suite = _load_test_suite(tmp_path, target_types=["eef", "joint"])
    eef, joint = _cases(suite)

    assert eef.case_id != joint.case_id
    assert eef.primary_success == joint.primary_success == "geometry_valid"
    assert joint.scenario_id == "joint_waypoint_path"
    assert joint.case_parameters == {
        "target_type": "joint",
        "motion_validity": "ordered_joint_waypoints",
        "joint_threshold_rad": 0.02,
    }
    assert torch.equal(eef.start_qpos, joint.start_qpos)
    assert torch.equal(eef.reference_qpos, joint.reference_qpos)
    assert torch.equal(eef.target_waypoints, joint.target_waypoints)
    suite.free_space.target_types.reverse()
    repeated = _cases(suite)
    assert [case.case_id for case in repeated] == [joint.case_id, eef.case_id]
    assert torch.equal(repeated[0].reference_qpos, joint.reference_qpos)


@pytest.mark.parametrize(
    ("target_types", "error"),
    [
        ([], ValueError),
        (["eef", "eef"], ValueError),
        (["cartesian"], ValueError),
        ("joint", TypeError),
        ([1], TypeError),
    ],
)
def test_loader_rejects_invalid_target_modalities(tmp_path, target_types, error):
    with pytest.raises(error, match="target_types"):
        _load_test_suite(tmp_path, target_types=target_types)


@pytest.mark.parametrize("threshold", [0.0, -0.1, float("nan"), float("inf"), True])
def test_loader_rejects_invalid_joint_threshold(tmp_path, threshold):
    with pytest.raises(ValueError, match="joint_threshold_rad"):
        _load_test_suite(tmp_path, joint_threshold_rad=threshold)


def test_domain_can_select_joint_targets_and_threshold(tmp_path):
    suite = _load_test_suite(tmp_path)
    suite.tracks[0].domain = DomainCfg(
        id="joint",
        version="v1",
        kind="nominal",
        overrides={"target_types": ["joint"], "joint_threshold_rad": 0.03},
    )
    joint = _cases(suite)[0]

    assert joint.case_parameters["motion_validity"] == "ordered_joint_waypoints"
    assert joint.case_parameters["joint_threshold_rad"] == 0.03


def test_joint_only_adapter_is_eligible_for_joint_population(tmp_path):
    class JointOnlyAdapter(PlannerAdapter):
        capabilities = frozenset({"joint_waypoint", "batched", "empty_world"})

        def build(self) -> None:
            """No resources are needed for support checks."""

        def plan(self, case):
            raise AssertionError("Support checks must precede planning.")

    suite = _load_test_suite(tmp_path, target_types=["eef", "joint"])
    adapter = JointOnlyAdapter(
        suite.planners[0],
        PlannerContext(_ReachableRobot(), "arm", torch.device("cpu"), 40),
    )
    eef, joint = _cases(suite)

    assert FreeSpaceScenario.required_capabilities <= adapter.capabilities
    assert adapter.supports_case(joint) == (True, None)
    assert adapter.supports_case(eef)[0] is False


@pytest.mark.parametrize(
    "suite_name",
    [
        "motion_panda_free_space_smoke",
        "motion_panda_free_space_coverage",
        "motion_ur5_free_space_coverage",
    ],
)
def test_motion_suites_cover_both_targets_with_shared_baselines(suite_name):
    suite = load_suite(suite_name)
    planners = {spec.id: spec for spec in suite.planners}

    assert suite.suite_version.endswith("_v3")
    assert suite.free_space.sampling_method == "cartesian"
    assert suite.free_space.batch_modes == ["homogeneous", "heterogeneous"]
    assert suite.protocol.report_schema_version == "3"
    assert suite.free_space.target_types == ["eef", "joint"]
    assert suite.free_space.joint_threshold_rad == 0.02
    assert {spec.id for spec in suite.planners if spec.enabled} == {
        "curobo",
        "toppra",
        "ik_interpolate",
    }
    assert planners["toppra"].config == {}
    assert suite.protocol.joint_velocity_limit_rad_s is None
    assert suite.protocol.joint_acceleration_limit_rad_s2 == 10.0
    assert suite.protocol.joint_jerk_limit_rad_s3 is None
    assert suite.protocol.warmup_trials >= 1
    assert suite.protocol.measured_trials >= 3
    nominal = planners["nmg"].config
    retimed = dict(planners["nmg_retimed"].config)
    assert retimed.pop("retime") is True
    assert nominal == retimed


@pytest.mark.parametrize(
    "suite_name",
    [
        "atomic_franka_pgi_pose_batch",
        "atomic_franka_pgi_randomized",
        "atomic_ur5_pgi_randomized",
    ],
)
def test_atomic_sweeps_focus_on_interaction_skills(suite_name):
    suite = load_suite(suite_name)
    skill_ids = {
        skill["id"]
        for track in suite.enabled_tracks()
        for skill in track.config["skills"]
    }

    assert {"move_end_effector", "move_joints"}.isdisjoint(skill_ids)
    expected = {"pick_up", "place", "press", "move_held_object"}
    if suite_name.endswith("pose_batch"):
        expected |= {"twist", "slide"}
    assert skill_ids == expected
    assert suite.protocol.report_schema_version == "3"


def test_atomic_smoke_keeps_one_representative_case_per_motion_skill():
    suite = load_suite("atomic_franka_pgi_smoke")
    primitives = [
        skill
        for track in suite.enabled_tracks()
        for skill in track.config["skills"]
        if skill["id"] in {"move_end_effector", "move_joints"}
    ]

    assert {skill["id"] for skill in primitives} == {
        "move_end_effector",
        "move_joints",
    }
    assert all(len(skill["cases"]) == 1 for skill in primitives)


@pytest.mark.parametrize(
    "path", sorted((BENCHMARK_ROOT / "suites").glob("*.yaml")), ids=lambda p: p.stem
)
def test_packaged_suites_pass_production_loader(path):
    assert load_suite(str(path)).evaluation_tracks()


def test_default_suite_loads_canonical_motion_smoke():
    suite = load_suite()

    assert suite.name == "motion_panda_free_space_smoke"
    assert {track.scenario for track in suite.evaluation_tracks()} == {"free_space"}


def test_old_report_schema_requires_explicit_migration(tmp_path):
    suite = _load_test_suite(tmp_path)
    suite.protocol.report_schema_version = "2"
    with pytest.raises(ValueError, match="migrate the suite"):
        suite.validate_benchmark()
