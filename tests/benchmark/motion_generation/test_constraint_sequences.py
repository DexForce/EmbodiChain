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

"""Semantic evaluation does not add hidden constraints or relax ordering."""

from __future__ import annotations

import math
import pytest
import torch

from embodichain_benchmarks.motion_generation.config import (
    FreeSpaceTrackCfg,
)
from embodichain_benchmarks.motion_generation.metrics._constraints import (
    match_constraints,
)
from embodichain_benchmarks.motion_generation.scenarios._free_space_sampling import (
    _project_constraints,
)


def _pose(x=0.0, angle=0.0):
    pose = torch.eye(4)
    pose[:2, :2] = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    pose[0, 3] = x
    return pose


def _match(poses, goals, modalities, q=None, jq=None):
    return match_constraints(
        torch.zeros(len(poses), 2) if q is None else q,
        torch.stack(poses),
        torch.zeros(len(goals), 2) if jq is None else jq,
        torch.stack(goals),
        modalities,
        position_threshold_m=0.01,
        rotation_threshold_rad=0.1,
        joint_threshold_rad=0.01,
    )


def test_position_and_orientation_ignore_unconstrained_payloads():
    assert _match([_pose(angle=2.0)], [_pose()], ["position"])[
        "ordered_waypoints_reached"
    ]
    assert _match([_pose(x=2.0)], [_pose()], ["orientation"])[
        "ordered_waypoints_reached"
    ]
    assert not _match([_pose(x=2.0), _pose(angle=2.0)], [_pose()], ["pose"])[
        "ordered_waypoints_reached"
    ]


def test_mixed_constraints_use_strictly_ordered_distinct_arrivals():
    poses = [_pose(x=1.0, angle=1.0), _pose(x=1.0), _pose()]
    result = _match(
        poses, [_pose(x=1.0), _pose(), _pose()], ["position", "orientation", "joint"]
    )
    assert result["arrival_indices"] == [0, 1, 2]
    assert result["arrival_errors"][0] == {"modality": "position", "position_m": 0.0}
    assert not _match(
        poses[:2],
        [_pose(x=1.0), _pose(), _pose()],
        ["position", "orientation", "joint"],
    )["ordered_waypoints_reached"]


@pytest.mark.parametrize("joints", [6, 7])
def test_joint_constraints_use_maximum_error_and_retain_l2_diagnostic(joints):
    result = _match(
        [_pose()],
        [_pose()],
        ["joint"],
        q=torch.full((1, joints), 0.005),
        jq=torch.zeros(1, joints),
    )
    assert result["ordered_waypoints_reached"]
    assert result["joint_error_norm"] == "linf"
    error = result["arrival_errors"][0]
    assert error["joint_max_rad"] == pytest.approx(0.005)
    assert error["joint_l2_rad"] == pytest.approx(math.sqrt(joints) * 0.005)
    assert error["joint_l2_rad"] > 0.01


@pytest.mark.parametrize("error,reached", [(0.01, True), (0.0101, False)])
def test_joint_constraint_worst_joint_and_threshold_boundary(error, reached):
    result = _match([_pose()], [_pose()], ["joint"], q=torch.tensor([[0.0, error]]))
    assert result["ordered_waypoints_reached"] is reached


@pytest.mark.parametrize("threshold", [0.0, float("nan"), float("inf")])
def test_invalid_thresholds_cannot_make_a_constraint_pass(threshold):
    with pytest.raises(ValueError):
        match_constraints(
            torch.zeros(1, 2),
            torch.eye(4)[None],
            torch.zeros(1, 2),
            torch.eye(4)[None],
            ["pose"],
            position_threshold_m=threshold,
            rotation_threshold_rad=0.1,
            joint_threshold_rad=0.01,
        )


class _Robot:
    device = "cpu"

    def get_qpos_limits(self, name):
        return torch.tensor([[[-2.0, 2.0]] * 7])

    def compute_fk(self, qpos, name, to_matrix):
        poses = torch.eye(4).repeat(qpos.shape[0], 1, 1)
        poses[:, :3, 3] = qpos[:, :3]
        return poses


def test_projected_patterns_share_screened_cartesian_inputs():
    from embodichain_benchmarks.motion_generation.models import BenchmarkCase

    base = BenchmarkCase(
        "test",
        "free",
        "waypoint_path",
        "base",
        31,
        1,
        2,
        "direct",
        "nominal",
        torch.zeros(1, 7),
        torch.eye(4).repeat(1, 2, 1, 1),
        torch.zeros(1, 2, 7),
        case_parameters={"distance_bin": "near", "motion_component": "combined"},
    )
    cfg = FreeSpaceTrackCfg(
        seeds=[31],
        sampling_method="cartesian",
        constraint_patterns=[
            ["joint"],
            ["pose"],
            ["position"],
            ["joint", "orientation"],
        ],
    )
    cases = _project_constraints([base], cfg)
    assert len(cases) == 4
    assert len({case.case_id for case in cases}) == 4
    assert {case.case_parameters["base_case_id"] for case in cases} == {"base"}
    assert all(torch.equal(case.reference_qpos, base.reference_qpos) for case in cases)
    assert [case.case_parameters["motion_validity"] for case in cases] == [
        "ordered_joint_waypoints",
        "ordered_cartesian_waypoints",
        "ordered_constraint_waypoints",
        "ordered_constraint_waypoints",
    ]


def test_empty_trajectory_is_a_failed_semantic_match():
    result = match_constraints(
        torch.empty(0, 2),
        torch.empty(0, 4, 4),
        torch.zeros(1, 2),
        torch.eye(4)[None],
        ["position"],
        position_threshold_m=0.01,
        rotation_threshold_rad=0.1,
        joint_threshold_rad=0.01,
    )
    assert not result["ordered_waypoints_reached"]
    assert result["completed_waypoint_ratio"] == 0.0


@pytest.mark.parametrize(
    "validity,modalities",
    [
        ("ordered_constraint_waypoints", ["position", "joint"]),
        ("ordered_joint_waypoints", ["joint", "joint"]),
    ],
)
def test_case_outcomes_preserve_native_samples_for_masked_and_common_inputs(
    validity, modalities
):
    from embodichain_benchmarks.motion_generation.models import BenchmarkCase
    from embodichain_benchmarks.motion_generation.metrics.trajectory import (
        compute_case_outcomes,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanResult

    class Robot(_Robot):
        def get_qvel_limits(self, name):
            return torch.ones(1, 7) * 100

        def compute_batch_fk(self, qpos, name, to_matrix):
            pose = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
            pose[..., :3, 3] = qpos[..., :3]
            return pose

    q = torch.zeros(1, 3, 7)
    q[0, 1, 0] = 0.1
    q[0, 2, 0] = 0.2
    goals = torch.stack([_pose(x=0.1, angle=2.0), _pose(x=0.2, angle=2.0)])[None]
    case = BenchmarkCase(
        "test",
        "semantics",
        "constraint_sequence",
        "case",
        1,
        1,
        2,
        "mixed",
        "nominal",
        q[:, 0],
        goals,
        q[:, 1:],
        case_parameters={
            "motion_validity": validity,
            "constraint_modalities": modalities,
            "joint_threshold_rad": 0.01,
        },
    )

    def score(positions):
        return compute_case_outcomes(
            PlanResult(
                success=True, positions=positions, dt=torch.tensor([[0.0, 0.1, 0.1]])
            ),
            case,
            Robot(),
            "arm",
            validation_samples=10,
            position_threshold_m=0.01,
            rotation_threshold_rad=0.1,
            joint_limit_tolerance_rad=1e-5,
            timing_is_solved=False,
        )[0]

    outcome = score(q)
    assert outcome.geometry_valid
    assert outcome.constraint_metrics["arrival_indices"] == [5, 10]
    assert outcome.constraint_metrics["sample_grid"] == "native_with_linear_subdivision"
    assert outcome.final_rotation_err_deg is None
    assert outcome.waypoint_rotation_err_deg_max is None
    broken = q.clone()
    broken[0, 1, 0] = float("nan")
    failed = score(broken)
    assert not failed.geometry_valid
    assert failed.failure_code == "non_finite_trajectory"


@pytest.mark.parametrize(
    "patterns,count",
    [
        ([["joint"], ["joint", "joint"]], 2),
        ([["joint", "pose"], ["joint", "pose", "joint", "pose"]], 4),
    ],
)
def test_equivalent_pattern_expansions_are_rejected_by_suite_loader(
    patterns: list[list[str]], count: int
) -> None:
    from embodichain_benchmarks.motion_generation.config import (
        SuiteCfg,
        load_suite,
        suite_to_dict,
    )

    config = suite_to_dict(load_suite())
    config["tracks"][0]["config"].update(
        sampling_method="cartesian",
        constraint_patterns=patterns,
        waypoint_counts=[count],
        path_shapes=["direct"],
    )
    with pytest.raises(ValueError, match="equivalent.*K="):
        SuiteCfg.from_dict(config)
