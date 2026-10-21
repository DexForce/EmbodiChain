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

"""Independent geometric oracles for free-space workload generation."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from embodichain.benchmarks.motion_generation.config import SuiteCfg
from embodichain.benchmarks.motion_generation.scenarios.free_space import (
    FreeSpaceScenario,
)
from embodichain.benchmarks.motion_generation.metrics.trajectory import (
    compute_case_outcomes,
)
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain.utils.math import (
    axis_angle_to_rotation_matrix,
    axis_angle_from_quat,
    quat_from_matrix,
)


class CartesianRobot:
    """Six independently controlled Cartesian coordinates plus a redundant joint."""

    device = torch.device("cpu")

    def get_qpos_limits(self, name):
        return torch.tensor([[[-6.0, 6.0]] * 7])

    def get_qvel_limits(self, name):
        return torch.full((1, 7), 100.0)

    def compute_fk(self, qpos, name, to_matrix):
        poses = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        poses[..., :3, 3] = qpos[..., :3]
        poses[..., :3, :3] = axis_angle_to_rotation_matrix(qpos[..., 3:6])
        return poses

    def compute_batch_fk(self, qpos, name, to_matrix):
        return self.compute_fk(qpos, name, to_matrix)

    def compute_ik(self, pose, joint_seed, name):
        solved = joint_seed.clone()
        solved[:, :3] = pose[:, :3, 3]
        solved[:, 3:6] = axis_angle_from_quat(quat_from_matrix(pose[:, :3, :3]))
        return torch.ones(pose.shape[0], dtype=torch.bool), solved


def suite_for_sampling(**overrides):
    config = {
        "sampling_method": "cartesian",
        "batch_sizes": [4],
        "batch_modes": ["heterogeneous"],
        "distance_bins": ["near"],
        "motion_components": ["translation"],
        "waypoint_counts": [3],
        "path_shapes": ["direct"],
        "target_types": ["eef", "joint"],
        "start_state_bins": ["nominal"],
        "seeds": [11, 23],
        **overrides,
    }
    return SuiteCfg.from_dict(
        {
            "suite_version": "cartesian_test_v1",
            "planners": [{"id": "test", "adapter": "ik_interpolate"}],
            "tracks": [{"id": "cartesian", "scenario": "free_space", "config": config}],
        }
    )


def generate(suite):
    track = suite.enabled_tracks()[0]
    return FreeSpaceScenario().generate_cases(suite, track, CartesianRobot(), "arm", 4)


def test_cartesian_population_is_seeded_and_goal_modalities_share_evidence():
    suite = suite_for_sampling()
    torch.manual_seed(731)
    before = torch.random.get_rng_state().clone()
    first = generate(suite)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(19)
    second = generate(suite)
    for left, right in zip(first, second):
        assert left.case_id == right.case_id
        torch.testing.assert_close(left.target_waypoints, right.target_waypoints)
    assert not torch.equal(first[0].start_qpos, first[2].start_qpos)
    torch.testing.assert_close(first[0].start_qpos, first[1].start_qpos)
    torch.testing.assert_close(first[0].target_waypoints, first[1].target_waypoints)
    torch.testing.assert_close(first[0].reference_qpos, first[1].reference_qpos)
    assert first[0].case_parameters["unique_input_rows"] == 4


@pytest.mark.parametrize("component", ["translation", "rotation", "combined"])
@pytest.mark.parametrize(
    "distance_bin,translation,rotation",
    [
        ("near", (0.02, 0.06), (0.08, 0.20)),
        ("mid", (0.08, 0.18), (0.25, 0.50)),
        ("far", (0.20, 0.35), (0.60, 1.00)),
    ],
)
def test_requested_motion_and_distance_match_measured_tcp(
    component, distance_bin, translation, rotation
):
    case = generate(
        suite_for_sampling(
            motion_components=[component], distance_bins=[distance_bin], seeds=[11]
        )
    )[0]
    robot = CartesianRobot()
    start = robot.compute_fk(case.start_qpos, "arm", True)
    target = case.target_waypoints[:, -1]
    displacement = (target[:, :3, 3] - start[:, :3, 3]).norm(dim=-1)
    if component == "rotation":
        torch.testing.assert_close(target[:, :3, 3], start[:, :3, 3])
    else:
        assert bool(
            (
                (displacement >= translation[0] - 1e-6)
                & (displacement <= translation[1] + 1e-6)
            ).all()
        )
    if component == "translation":
        torch.testing.assert_close(target[:, :3, :3], start[:, :3, :3])
    else:
        angle = torch.tensor(case.case_parameters["tcp_rotation_rad"])
        assert bool(
            ((angle >= rotation[0] - 1e-5) & (angle <= rotation[1] + 1e-5)).all()
        )
    measured = robot.compute_batch_fk(case.reference_qpos, "arm", True)
    torch.testing.assert_close(measured, case.target_waypoints, atol=1e-5, rtol=1e-5)


def test_single_target_and_pure_rotation_do_not_claim_cartesian_turns():
    cases = generate(
        suite_for_sampling(
            waypoint_counts=[1, 3],
            path_shapes=["direct", "l_turn", "s_curve"],
            motion_components=["translation", "rotation"],
            seeds=[11],
        )
    )
    assert all(
        case.path_shape == "direct"
        for case in cases
        if case.num_waypoints == 1
        or case.case_parameters["motion_component"] == "rotation"
    )
    shaped = [case for case in cases if case.path_shape == "l_turn"]
    assert shaped
    for case in shaped:
        start = CartesianRobot().compute_fk(case.start_qpos, "arm", True)[:, :3, 3]
        points = torch.cat((start[:, None], case.target_waypoints[:, :, :3, 3]), dim=1)
        segments = points[:, 1:] - points[:, :-1]
        assert bool(
            (
                torch.cross(segments[:, 0], segments[:, 1], dim=-1).norm(dim=-1) > 1e-6
            ).all()
        )


def test_homogeneous_and_heterogeneous_batches_are_explicit():
    cases = generate(
        suite_for_sampling(batch_modes=["homogeneous", "heterogeneous"], seeds=[11])
    )
    for case in cases:
        same = torch.equal(
            case.start_qpos, case.start_qpos[:1].expand_as(case.start_qpos)
        )
        if case.case_parameters["batch_mode"] == "homogeneous":
            assert same
            assert case.case_parameters["unique_input_rows"] == 1
        else:
            assert not same
            assert case.case_parameters["unique_input_rows"] == 4


def test_conditioning_and_joint_margin_are_measured():
    cases = generate(
        suite_for_sampling(
            start_state_bins=["near_limit", "near_singularity"], seeds=[11]
        )
    )
    for case in cases:
        params = case.case_parameters
        assert len(params["jacobian_condition_ratio"]) == case.batch_size
        assert all(0 <= value <= 1 for value in params["jacobian_condition_ratio"])
        assert len(params["conditioning_class"]) == case.batch_size
        assert all(0 <= value <= 0.5 for value in params["joint_margin_fraction"])
        if case.start_state_bin == "near_limit":
            assert max(params["joint_margin_fraction"]) <= 0.04


def test_unreachable_generation_is_bounded_and_does_not_shrink_population(monkeypatch):
    suite = suite_for_sampling(max_generation_attempts=2)
    calls = []

    def fail(self, pose, joint_seed, name):
        calls.append(pose)
        return torch.zeros(len(pose), dtype=torch.bool), joint_seed.clone()

    monkeypatch.setattr(CartesianRobot, "compute_ik", fail)
    with pytest.raises(ValueError, match="invalid_case.*after 2 attempts"):
        generate(suite)
    assert len(calls) == 2 * 3  # two attempts, three ordered waypoints


def test_trajectory_at_goal_cannot_skip_requested_start():
    case = generate(suite_for_sampling(seeds=[11]))[0]
    goal = case.reference_qpos[:, -1:]
    positions = goal.expand(-1, 2, -1).clone()
    dt = torch.tensor([[0.0, 1.0]]).expand(case.batch_size, -1).clone()
    case = replace(
        case, num_waypoints=1, target_waypoints=case.target_waypoints[:, -1:]
    )
    outcomes = compute_case_outcomes(
        PlanResult(success=True, positions=positions, dt=dt),
        case,
        CartesianRobot(),
        "arm",
        validation_samples=8,
        position_threshold_m=0.01,
        rotation_threshold_rad=0.1,
        joint_limit_tolerance_rad=1e-5,
    )
    assert all(outcome.ordered_waypoints_reached for outcome in outcomes)
    assert all(
        not outcome.geometry_valid and not outcome.timed_motion_valid
        for outcome in outcomes
    )
    assert all(outcome.failure_code == "start_state_mismatch" for outcome in outcomes)


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_modes": ["heterogeneous", "heterogeneous"]},
        {"distance_bins": ["unknown"]},
        {"motion_components": []},
        {"max_generation_attempts": 0},
        {"singularity_candidates": True},
        {"jacobian_position_scale_m": float("nan")},
    ],
)
def test_invalid_sampling_configuration_fails_before_runtime(overrides):
    with pytest.raises((TypeError, ValueError)):
        suite_for_sampling(**overrides)


def test_screening_uses_stricter_suite_thresholds(monkeypatch):
    solve = CartesianRobot.compute_ik

    def biased(self, pose, joint_seed, name):
        success, solved = solve(self, pose, joint_seed, name)
        solved[:, 0] += 0.0005  # 0.5 mm exceeds the requested 0.1 mm threshold.
        return success, solved

    monkeypatch.setattr(CartesianRobot, "compute_ik", biased)
    suite = suite_for_sampling(max_generation_attempts=2)
    suite.protocol.position_threshold_m = 0.0001
    with pytest.raises(ValueError, match="invalid_case"):
        generate(suite)


def test_reference_rotation_is_valid_under_strict_angular_threshold():
    suite = suite_for_sampling(
        seeds=[11],
        motion_components=["rotation"],
        distance_bins=["far"],
        waypoint_counts=[1],
    )
    suite.protocol.rotation_threshold_rad = 0.0001
    case = generate(suite)[0]
    positions = torch.cat((case.start_qpos[:, None], case.reference_qpos), dim=1)
    dt = torch.tensor([[0.0, 1.0]]).expand(case.batch_size, -1).clone()
    outcomes = compute_case_outcomes(
        PlanResult(success=True, positions=positions, dt=dt),
        case,
        CartesianRobot(),
        "arm",
        validation_samples=8,
        position_threshold_m=0.01,
        rotation_threshold_rad=suite.protocol.rotation_threshold_rad,
        joint_limit_tolerance_rad=1e-5,
    )
    assert all(outcome.geometry_valid for outcome in outcomes)
    assert max(outcome.final_rotation_err_deg for outcome in outcomes) < 0.001


def test_workload_tables_preserve_modality_batch_semantics_and_uncertainty():
    from embodichain.benchmarks.motion_generation.aggregation import aggregate_results
    from embodichain.benchmarks.motion_generation.models import (
        AlgorithmRole,
        PlannerMetadata,
        TrialPhase,
        TrialRecord,
    )
    from embodichain.benchmarks.motion_generation.metrics.trajectory import (
        make_failure_outcomes,
    )

    cases = generate(suite_for_sampling(batch_modes=["homogeneous", "heterogeneous"]))
    metadata = PlannerMetadata(
        algorithm_id="test",
        algorithm_role=AlgorithmRole.CANDIDATE,
        adapter="test",
        config_hash="test",
        capabilities=frozenset(),
        timing_is_solved=True,
    )
    records = [
        TrialRecord(
            suite_version=case.suite_version,
            track=case.track,
            scenario_id=case.scenario_id,
            case_id=case.case_id,
            algorithm_id="test",
            algorithm_role=AlgorithmRole.CANDIDATE,
            model_revision="test",
            planner_config_hash="test",
            seed=case.seed,
            repeat=0,
            batch_size=case.batch_size,
            waypoint_count=case.num_waypoints,
            path_shape=case.path_shape,
            start_state_bin=case.start_state_bin,
            phase=TrialPhase.MEASURED,
            primary_success=case.primary_success,
            cost_time_ms=1.0,
            outcomes=make_failure_outcomes(case.batch_size, "fixture"),
        )
        for case in cases
    ]
    aggregates = aggregate_results(
        records, [metadata], cases, 1, track_scenarios={"cartesian": "free_space"}
    )
    rows = aggregates["motion_planning_success_and_metrics"]
    assert len(rows) == 4
    assert {(row["target_type"], row["batch_mode"]) for row in rows} == {
        (target, mode)
        for target in ("eef", "joint")
        for mode in ("homogeneous", "heterogeneous")
    }
    assert all(
        row["seed_clusters"] == 2 and row["coverage_rate"] == 1.0 for row in rows
    )
    assert all(
        row["success_ci_lower"] == 0.0 and row["success_ci_upper"] == 0.0
        for row in rows
    )
    assert all(
        row["unique_input_rows"] == (2 if row["batch_mode"] == "homogeneous" else 8)
        for row in rows
    )
    assert len(aggregates["motion_planning_time_and_memory"]) == 4


def test_balanced_design_bounds_geometry_matrix_without_losing_workload_axes():
    full = generate(
        suite_for_sampling(
            seeds=[11, 23, 37],
            waypoint_counts=[1, 3, 5],
            path_shapes=["direct", "l_turn", "s_curve"],
            motion_components=["translation", "rotation", "combined"],
        )
    )
    balanced = generate(
        suite_for_sampling(
            population_design="balanced",
            seeds=[11, 23, 37],
            waypoint_counts=[1, 3, 5],
            path_shapes=["direct", "l_turn", "s_curve"],
            motion_components=["translation", "rotation", "combined"],
        )
    )
    assert len(balanced) == 3 * 3 * 2  # seeds, motion components, goal modalities
    assert len(balanced) < len(full)
    assert {case.case_parameters["motion_component"] for case in balanced} == {
        "translation",
        "rotation",
        "combined",
    }
    assert len({case.case_id for case in balanced}) == len(balanced)
