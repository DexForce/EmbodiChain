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

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    HeldObjectPoseGoal,
    MoveHeldObjectOptions,
)
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
from scripts.benchmark.motion_generation.scenarios.atomic_task import (
    AtomicTaskScenario,
    create_atomic_skill_provider,
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


def test_press_waypoint_validation_accepts_half_turn_about_z_symmetry():
    target = torch.eye(4)
    target[:3, 0] = -target[:3, 0]
    target[:3, 1] = -target[:3, 1]
    positions = torch.zeros(1, 2, 7)
    strict_case = replace(
        _case(),
        scenario_id="press",
        skill_id="press",
        target_waypoints=target.reshape(1, 1, 4, 4),
    )
    symmetric_case = replace(
        strict_case,
        case_parameters={"waypoint_rotation_symmetry": "half_turn_about_z"},
    )

    strict = compute_case_outcomes(
        _timed_plan_result(positions, success=True),
        strict_case,
        _MetricRobot(),
        "arm",
        validation_samples=8,
        position_threshold_m=1.0e-4,
        rotation_threshold_rad=1.0e-4,
        joint_limit_tolerance_rad=1.0e-5,
    )
    symmetric = compute_case_outcomes(
        _timed_plan_result(positions, success=True),
        symmetric_case,
        _MetricRobot(),
        "arm",
        validation_samples=8,
        position_threshold_m=1.0e-4,
        rotation_threshold_rad=1.0e-4,
        joint_limit_tolerance_rad=1.0e-5,
    )

    assert strict[0].failure_code == "waypoint_miss"
    assert symmetric[0].motion_valid is True
    assert symmetric[0].final_rotation_err_deg == pytest.approx(0.0)


def test_waypoint_errors_use_threshold_greedy_arrivals():
    """Continuous errors must come from the same matching as motion_valid."""
    waypoints = torch.stack([_translated_pose(0.0), _translated_pose(0.10)])
    # Sample 0 hits W0 under threshold. Sample 1 is a high-error later pose that
    # unconstrained DP could prefer for W0 while still sequencing W1 later.
    trajectory = torch.stack(
        [
            _translated_pose(0.0),
            _translated_pose(0.04),
            _translated_pose(0.10),
        ]
    )
    result = match_ordered_waypoints(
        trajectory,
        waypoints,
        position_threshold_m=0.05,
        rotation_threshold_rad=0.3,
    )

    assert result["ordered_waypoints_reached"] is True
    assert result["arrival_indices"] == [0, 2]
    assert max(result["position_errors_m"]) <= 0.05 + 1.0e-9


class _MetricRobot:
    """Minimal FK stub: joint xyz maps to TCP translation."""

    device = torch.device("cpu")
    limit_lo = -1.0
    limit_hi = 1.0

    def get_qpos_limits(self, name: str):  # noqa: ARG002
        limits = torch.tensor([[self.limit_lo, self.limit_hi]]).repeat(7, 1)
        return limits.unsqueeze(0)

    def compute_batch_fk(
        self, qpos: torch.Tensor, name: str, to_matrix: bool
    ):  # noqa: ARG002
        poses = torch.eye(4).repeat(qpos.shape[0], qpos.shape[1], 1, 1)
        poses[..., :3, 3] = qpos[..., :3]
        return poses

    def compute_fk(
        self, qpos: torch.Tensor, name: str, to_matrix: bool
    ):  # noqa: ARG002
        poses = torch.eye(4).repeat(qpos.shape[0], 1, 1)
        poses[:, :3, 3] = qpos[:, :3]
        return poses


def _valid_motion_case_and_positions() -> tuple[BenchmarkCase, torch.Tensor]:
    case = _case()
    case.target_waypoints[0, 0, 0, 3] = 0.1
    positions = torch.zeros(1, 2, 7)
    positions[0, 1, 0] = 0.1
    return case, positions


def _timed_plan_result(
    positions: torch.Tensor,
    *,
    success: bool | torch.Tensor,
) -> PlanResult:
    """Build a synthetic plan with explicit benchmark timing."""
    dt = torch.zeros(positions.shape[:2], device=positions.device)
    if positions.shape[1] > 1:
        dt[:, 1:] = 0.025
    return PlanResult(
        success=success,
        positions=positions,
        dt=dt,
    )


def test_motion_valid_ignores_planner_reported_failure_in_outcomes_and_aggregates():
    case, positions = _valid_motion_case_and_positions()
    outcomes = compute_case_outcomes(
        _timed_plan_result(positions, success=False),
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
    row = aggregate_results([measured], metadata, [case], measured_trials=1)[
        "success_and_metrics"
    ][0]
    assert row["success_rate"] == pytest.approx(1.0)
    assert row["planning_success_rate"] == pytest.approx(0.0)
    assert row["top_failure"] is None


def test_missing_positions_and_joint_limit_violation_fail_motion_valid():
    case = _case()
    missing = compute_case_outcomes(
        PlanResult(success=True, positions=None),
        case,
        _MetricRobot(),
        "arm",
        validation_samples=8,
        position_threshold_m=1.0e-4,
        rotation_threshold_rad=1.0e-4,
        joint_limit_tolerance_rad=1.0e-5,
    )
    assert missing[0].motion_valid is False
    assert missing[0].failure_code == "non_finite_trajectory"

    # Reach the waypoint at x=2 while joint 0 is outside [-1, 1].
    case.target_waypoints[0, 0, 0, 3] = 2.0
    positions = torch.zeros(1, 2, 7)
    positions[0, :, 0] = 2.0
    violated = compute_case_outcomes(
        _timed_plan_result(positions, success=True),
        case,
        _MetricRobot(),
        "arm",
        validation_samples=8,
        position_threshold_m=1.0e-4,
        rotation_threshold_rad=1.0e-4,
        joint_limit_tolerance_rad=1.0e-5,
    )
    assert violated[0].ordered_waypoints_reached is True
    assert violated[0].joint_limit_violation is True
    assert violated[0].motion_valid is False
    assert violated[0].failure_code == "joint_limit_violation"


def test_non_finite_trajectory_skips_joint_limit_metrics():
    case = _case()
    positions = torch.zeros(1, 2, 7)
    positions[0, 1, 0] = float("inf")
    outcomes = compute_case_outcomes(
        _timed_plan_result(positions, success=True),
        case,
        _MetricRobot(),
        "arm",
        validation_samples=8,
        position_threshold_m=1.0e-4,
        rotation_threshold_rad=1.0e-4,
        joint_limit_tolerance_rad=1.0e-5,
    )

    assert outcomes[0].finite is False
    assert outcomes[0].failure_code == "non_finite_trajectory"
    assert outcomes[0].joint_limit_violation is False
    assert outcomes[0].max_normalized_joint_violation is None
    assert outcomes[0].motion_valid is False


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


def test_nmg_model_revision_is_derived_from_runtime_model_path():
    from scripts.benchmark.motion_generation.config import PlannerSpecCfg
    from scripts.benchmark.motion_generation.planners.base import PlannerContext
    from scripts.benchmark.motion_generation.planners.nmg_onnx import NmgOnnxAdapter

    adapter = NmgOnnxAdapter(
        PlannerSpecCfg(
            id="nmg",
            adapter="nmg_onnx",
            role="candidate",
            config={"onnx_model_path": "/models/unified-k3.onnx"},
        ),
        PlannerContext(
            robot=Mock(),
            control_part="arm",
            device=torch.device("cpu"),
            sample_interval=1,
        ),
    )

    assert adapter.metadata.model_revision == "unified-k3"


def test_seed_override_applies_to_atomic_tracks():
    suite = load_suite("atomic_franka_pgi_curobo_randomized")

    _apply_overrides(suite, seeds=[104])

    assert suite.free_space.seeds == [104]
    assert suite.enabled_tracks()[0].config["seeds"] == [104]


def test_atomic_task_antipodal_grasp_uses_standalone_generator(monkeypatch):
    from scripts.tutorials.atomic_action import tutorial_utils

    vertices = torch.tensor(
        [[-0.025, -0.025, -0.025], [0.025, 0.025, 0.025]],
        dtype=torch.float32,
    )
    triangles = torch.tensor([[0, 1, 0]], dtype=torch.int64)
    entity = Mock(uid="atomic_cube")
    entity.get_vertices.return_value = [vertices]
    entity.get_triangles.return_value = [triangles]
    handle = Mock(object_id="cube", entity=entity)

    candidate = torch.eye(4)
    candidate[1, 1] = -1.0
    candidate[2, 2] = -1.0
    generator = Mock()
    generator.get_valid_grasp_poses.return_value = [
        (candidate.unsqueeze(0), torch.tensor([0.25]))
    ]
    generator_factory = Mock(return_value=generator)
    monkeypatch.setattr(
        tutorial_utils,
        "create_parallel_jaw_grasp_pose_generator",
        generator_factory,
    )

    scenario = AtomicTaskScenario()
    scenario.robot = Mock(device=torch.device("cpu"))
    scenario.robot.compute_ik.return_value = (
        torch.tensor([True]),
        torch.zeros(1, 7),
    )
    result = scenario.resolve_antipodal_grasp(
        handle,
        torch.eye(4).unsqueeze(0),
        torch.tensor([0.0, 0.0, -1.0]),
        seed=11,
        start_qpos=torch.zeros(1, 7),
        pre_grasp_distance=0.15,
        lift_height=0.16,
        n_sample=321,
        max_candidates=8,
        alignment_max_angle_deg=10.0,
    )

    generator_factory.assert_called_once_with(n_sample=321, force_refresh=False)
    call = generator.get_valid_grasp_poses.call_args.kwargs
    assert torch.equal(call["mesh_vertices"], vertices)
    assert torch.equal(call["mesh_triangles"], triangles)
    assert torch.equal(result, candidate.unsqueeze(0))


def test_atomic_task_pickup_builds_engine_owned_endpoint_binding():
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 8
    robot.control_parts = {"arm": object(), "hand": object()}
    robot.get_qpos.return_value = torch.zeros(1, 8)
    robot.get_qvel.return_value = torch.zeros(1, 8)
    robot.get_joint_ids.side_effect = lambda name: (
        list(range(7)) if name == "arm" else [7]
    )
    motion_generator = Mock(robot=robot, device=torch.device("cpu"))
    motion_generator.planner.cfg.planner_type = "stub"
    engine = AtomicActionEngine(
        motion_generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(1),
                grasp=torch.ones(1),
            )
        },
    )
    entity = Mock(uid="atomic_cube")
    scenario = AtomicTaskScenario()
    scenario.robot = robot
    scenario.control_part = "arm"
    scenario.end_effector_part = "hand"
    scenario._engine = engine
    scenario._objects["cube"] = Mock(object_id="cube", entity=entity)
    poses = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
    case = BenchmarkCase(
        suite_version="test_v1",
        track="atomic-task",
        scenario_id="pick_up",
        case_id="atomic-task:pick_up:cube:s11",
        seed=11,
        batch_size=1,
        num_waypoints=3,
        path_shape="approach_grasp_lift",
        start_state_bin="pre_pick",
        start_qpos=torch.zeros(1, 7),
        target_waypoints=poses,
        reference_qpos=torch.zeros(1, 3, 7),
        skill_id="pick_up",
        object_id="cube",
        case_parameters={
            "sample_count": 12,
            "approach_direction": [0.0, 0.0, -1.0],
            "pre_grasp_distance_m": 0.15,
            "lift_height_m": 0.16,
            "hand_interp_steps": 4,
        },
    )

    invocation = create_atomic_skill_provider("pick_up").build_invocation(
        scenario,
        case,
        Mock(motion_policy_planner="curobo"),
    )

    assert isinstance(invocation.goal, GraspGoal)
    assert invocation.goal.semantics.entity_id == "atomic_cube"
    assert invocation.binding.owner_id == engine.binding_owner_id
    assert set(invocation.binding.endpoint_keys) == {
        ("primary", "motion"),
        ("primary", "grasp"),
    }


def test_atomic_task_prepare_planner_registers_benchmark_objects_in_scene():
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 7
    robot.control_parts = {"arm": object()}
    robot.get_qpos.return_value = torch.zeros(1, 7)
    robot.get_qvel.return_value = torch.zeros(1, 7)
    motion_generator = Mock(robot=robot, device=torch.device("cpu"))
    motion_generator.planner.cfg.planner_type = "stub"
    adapter = Mock()
    adapter.require_motion_generator.return_value = motion_generator

    object_pose = torch.eye(4).unsqueeze(0)
    entity = Mock(uid="atomic_cube")
    entity.get_local_pose.return_value = object_pose
    scenario = AtomicTaskScenario()
    scenario._objects["cube"] = Mock(object_id="cube", entity=entity)

    scenario.prepare_planner(adapter, Mock())

    scene = scenario.require_engine().initial_context().scene
    assert tuple(scene.entities) == ("atomic_cube",)
    assert torch.equal(scene.entities["atomic_cube"].pose, object_pose)


def test_atomic_task_move_held_object_uses_current_options_contract():
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 8
    robot.control_parts = {"arm": object(), "hand": object()}
    robot.get_qpos.return_value = torch.zeros(1, 8)
    robot.get_qvel.return_value = torch.zeros(1, 8)
    robot.get_joint_ids.side_effect = lambda name: (
        list(range(7)) if name == "arm" else [7]
    )
    motion_generator = Mock(robot=robot, device=torch.device("cpu"))
    motion_generator.planner.cfg.planner_type = "stub"
    engine = AtomicActionEngine(
        motion_generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(1),
                grasp=torch.ones(1),
            )
        },
    )
    scenario = AtomicTaskScenario()
    scenario.robot = robot
    scenario.control_part = "arm"
    scenario.end_effector_part = "hand"
    scenario._engine = engine
    target_object_pose = torch.eye(4).unsqueeze(0)
    case = BenchmarkCase(
        suite_version="test_v1",
        track="atomic-task",
        scenario_id="move_held_object",
        case_id="atomic-task:move_held_object:cube:s11",
        seed=11,
        batch_size=1,
        num_waypoints=1,
        path_shape="held_object_transport",
        start_state_bin="object_held",
        start_qpos=torch.zeros(1, 7),
        target_waypoints=target_object_pose.unsqueeze(1),
        reference_qpos=torch.zeros(1, 1, 7),
        skill_id="move_held_object",
        object_id="cube",
        case_parameters={
            "sample_count": 12,
            "target_object_pose": target_object_pose.tolist(),
        },
    )

    invocation = create_atomic_skill_provider("move_held_object").build_invocation(
        scenario,
        case,
        Mock(),
    )

    assert isinstance(invocation.goal, HeldObjectPoseGoal)
    assert type(invocation.skill_options) is MoveHeldObjectOptions


@pytest.mark.parametrize("override", [{"nmg_pos_eps": 0.0}, {"nmg_rot_eps": -0.1}])
def test_nmg_precision_rejects_non_positive_values(override):
    suite = load_suite("smoke")

    with pytest.raises(ValueError, match="NMG"):
        _apply_overrides(suite, **override)


class _FrankaLimitRobot(_MetricRobot):
    """FK stub with Franka-like joint limits for free-space case generation."""

    def get_qpos_limits(self, name: str):  # noqa: ARG002
        lower = torch.tensor([-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8])
        upper = torch.tensor([2.8, 1.7, 2.8, -0.05, 2.8, 3.7, 2.8])
        return torch.stack([lower, upper], dim=-1).unsqueeze(0)


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
    robot = _FrankaLimitRobot()
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
        suite, track, _FrankaLimitRobot(), "arm", batch_size=2
    )

    assert [case.start_state_bin for case in cases] == ["nominal", "near_limit"]
    assert len({case.case_id for case in cases}) == 2


def test_free_space_starts_align_across_path_shape_and_waypoint_count():
    """random_reachable starts must be shared across shapes/W for fair compares."""
    suite = load_suite("smoke")
    suite.free_space.batch_sizes = [2]
    suite.free_space.waypoint_counts = [1, 5]
    suite.free_space.path_shapes = ["direct", "s_curve"]
    suite.free_space.seeds = [11]
    suite.free_space.start_state_bins = ["random_reachable"]
    track = suite.enabled_tracks()[0]
    cases = create_scenario_provider("free_space").generate_cases(
        suite, track, _FrankaLimitRobot(), "arm", batch_size=2
    )

    assert len(cases) == 4
    assert {(case.path_shape, case.num_waypoints) for case in cases} == {
        ("direct", 1),
        ("direct", 5),
        ("s_curve", 1),
        ("s_curve", 5),
    }
    reference = cases[0].start_qpos
    for case in cases[1:]:
        assert torch.equal(case.start_qpos, reference)


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


def test_leaderboard_latency_p95_uses_case_macro_not_trial_pool():
    """Many fast repeats must not drown a slow case in leaderboard latency_p95."""
    metadata = [
        PlannerMetadata(
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="curobo",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]
    fast = BenchmarkCase(
        suite_version="test_v1",
        track="free-space-common",
        scenario_id="reach",
        case_id="case-fast",
        seed=11,
        batch_size=1,
        num_waypoints=1,
        path_shape="direct",
        start_state_bin="nominal",
        start_qpos=torch.zeros(1, 7),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4),
        reference_qpos=torch.zeros(1, 1, 7),
    )
    slow = BenchmarkCase(
        suite_version="test_v1",
        track="free-space-common",
        scenario_id="reach",
        case_id="case-slow",
        seed=23,
        batch_size=64,
        num_waypoints=5,
        path_shape="s_curve",
        start_state_bin="nominal",
        start_qpos=torch.zeros(64, 7),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4).expand(64, 1, 4, 4).clone(),
        reference_qpos=torch.zeros(64, 1, 7),
    )

    def _latency_record(
        case: BenchmarkCase, *, cost: float, repeat: int
    ) -> TrialRecord:
        return TrialRecord(
            suite_version="test_v1",
            track="free-space-common",
            scenario_id=case.scenario_id,
            case_id=case.case_id,
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            model_revision="curobo-v2",
            planner_config_hash="abc",
            seed=case.seed,
            repeat=repeat,
            batch_size=case.batch_size,
            waypoint_count=case.num_waypoints,
            path_shape=case.path_shape,
            start_state_bin=case.start_state_bin,
            phase=TrialPhase.MEASURED,
            cost_time_ms=cost,
            outcomes=tuple(
                replace(_outcome(), env_index=env_index)
                for env_index in range(case.batch_size)
            ),
        )

    # 20 fast repeats at 10 ms plus one slow case at 1000 ms.
    # Trial-pooled nearest-rank p95 over 21 values is still 10 ms; case-macro
    # p95 over case means [10, 1000] is 1000 ms.
    records = [
        _latency_record(fast, cost=10.0, repeat=index) for index in range(20)
    ] + [_latency_record(slow, cost=1000.0, repeat=0)]

    row = aggregate_results(records, metadata, [fast, slow], measured_trials=1)[
        "leaderboard"
    ][0]

    assert row["latency_p95_ms"] == pytest.approx(1000.0)


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
    assert "motion-validity gate" in text
    assert "n_valid" in text


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


def test_success_table_uses_case_macro_and_counts_cases_not_env_slots():
    metadata = [
        PlannerMetadata(
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="curobo",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint"}),
        )
    ]
    case_a = _case()
    case_b = replace(_case(), case_id="case-missing", seed=23)
    records = [
        TrialRecord(
            suite_version="test_v1",
            track="free-space-common",
            scenario_id="reach",
            case_id=case_a.case_id,
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            model_revision="curobo-v2",
            planner_config_hash="abc",
            seed=case_a.seed,
            repeat=0,
            batch_size=1,
            waypoint_count=1,
            path_shape="direct",
            start_state_bin="nominal",
            phase=TrialPhase.MEASURED,
            cost_time_ms=10.0,
            outcomes=(_outcome(),),
        )
    ]
    row = aggregate_results(records, metadata, [case_a, case_b], measured_trials=1)[
        "success_and_metrics"
    ][0]
    assert row["cases"] == 2
    assert row["n_valid"] == 1
    # One measured success + one missing case (counts as 0) → macro 0.5.
    assert row["success_rate"] == pytest.approx(0.5)
    assert row["coverage_rate"] == pytest.approx(0.5)

    large = replace(
        _case(),
        case_id="case-b64",
        seed=23,
        batch_size=64,
        start_qpos=torch.zeros(64, 7),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4).expand(64, 1, 4, 4).clone(),
        reference_qpos=torch.zeros(64, 1, 7),
    )
    failed = replace(
        _outcome(),
        planning_success=False,
        ordered_waypoints_reached=False,
        motion_valid=False,
        completed_waypoint_ratio=0.0,
        failure_code="waypoint_miss",
    )
    large_record = TrialRecord(
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
        outcomes=tuple(replace(failed, env_index=i) for i in range(64)),
    )
    large_row = aggregate_results([large_record], metadata, [large], measured_trials=1)[
        "success_and_metrics"
    ][0]
    assert large_row["cases"] == 1
    assert large_row["n_valid"] == 0
    assert large_row["success_rate"] == pytest.approx(0.0)


def test_timed_call_reports_null_peak_gpu_without_cuda(monkeypatch):
    from scripts.benchmark.motion_generation.metrics import performance

    monkeypatch.setattr(performance.torch.cuda, "is_available", lambda: False)
    measured = performance.timed_call(lambda: 42)
    assert measured.result == 42
    assert measured.peak_gpu_mb is None


def test_runner_capability_gate_and_fake_adapter_lifecycle(tmp_path):
    """Exercise AVAILABILITY gating and MEASURED aggregation without DexSim."""
    from scripts.benchmark.motion_generation.artifacts import TrialJsonlWriter
    from scripts.benchmark.motion_generation.config import (
        PlannerSpecCfg,
        ProtocolCfg,
        SuiteCfg,
    )
    from scripts.benchmark.motion_generation.planners.base import PlannerAdapter
    from scripts.benchmark.motion_generation.registry import (
        register_planner_adapter,
        unregister_planner_adapter,
    )
    from scripts.benchmark.motion_generation.runner import BenchmarkRunner

    class _CapableFake(PlannerAdapter):
        capabilities = frozenset({"eef_waypoint", "batched", "empty_world"})

        def build(self) -> None:
            return None

        def plan(self, case: BenchmarkCase) -> PlanResult:
            steps = max(case.num_waypoints + 1, 2)
            positions = case.start_qpos.unsqueeze(1).expand(-1, steps, -1).clone()
            return _timed_plan_result(positions, success=True)

    class _IncapableFake(PlannerAdapter):
        capabilities = frozenset({"eef_waypoint"})

        def build(self) -> None:
            return None

        def plan(self, case: BenchmarkCase) -> PlanResult:  # noqa: ARG002
            raise AssertionError("incapable adapter must not plan")

    class _RuntimeUnavailableFake(PlannerAdapter):
        capabilities = frozenset({"eef_waypoint", "batched", "empty_world"})

        def availability(self) -> tuple[bool, str | None]:
            return False, "runtime missing"

        def build(self) -> None:
            return None

        def plan(self, case: BenchmarkCase) -> PlanResult:  # noqa: ARG002
            raise AssertionError("unavailable adapter must not plan")

    class _ContractBrokenFake(PlannerAdapter):
        capabilities = frozenset({"eef_waypoint", "batched", "empty_world"})

        def build(self) -> None:
            return None

        def plan(self, case: BenchmarkCase):  # noqa: ARG002
            return "not-a-plan-result"

    names = (
        "fake_capable",
        "fake_incapable",
        "fake_runtime_unavailable",
        "fake_contract_broken",
    )
    register_planner_adapter("fake_capable", _CapableFake)
    register_planner_adapter("fake_incapable", _IncapableFake)
    register_planner_adapter("fake_runtime_unavailable", _RuntimeUnavailableFake)
    register_planner_adapter("fake_contract_broken", _ContractBrokenFake)
    try:
        suite = SuiteCfg(
            name="motion_generation",
            suite_version="test_fake_v1",
            profile="smoke",
            protocol=ProtocolCfg(
                warmup_trials=0,
                measured_trials=1,
                sample_interval=4,
                validation_samples=4,
                position_threshold_m=1.0,
                rotation_threshold_rad=1.0,
            ),
        )
        specs = [
            PlannerSpecCfg(
                id="capable",
                adapter="fake_capable",
                role=AlgorithmRole.DIAGNOSTIC_BASELINE.value,
                enabled=True,
            ),
            PlannerSpecCfg(
                id="incapable",
                adapter="fake_incapable",
                role=AlgorithmRole.CANDIDATE.value,
                enabled=True,
            ),
            PlannerSpecCfg(
                id="runtime_down",
                adapter="fake_runtime_unavailable",
                role=AlgorithmRole.CANDIDATE.value,
                enabled=True,
            ),
            PlannerSpecCfg(
                id="broken",
                adapter="fake_contract_broken",
                role=AlgorithmRole.CANDIDATE.value,
                enabled=True,
            ),
        ]
        runner = BenchmarkRunner(suite, specs, device="cpu", output_root=tmp_path)
        case = replace(_case(), suite_version=suite.suite_version)
        runner.cases = [case]
        writer = TrialJsonlWriter(tmp_path / "trials.jsonl")

        robot = Mock(device=torch.device("cpu"))
        robot.set_qpos = Mock()
        robot.clear_dynamics = Mock()
        robot.get_qpos_limits = Mock(
            return_value=torch.tensor([[-2.0, 2.0]]).repeat(7, 1).unsqueeze(0)
        )
        robot.compute_batch_fk = Mock(
            side_effect=lambda qpos, name, to_matrix: (  # noqa: ARG005
                torch.eye(4).repeat(qpos.shape[0], qpos.shape[1], 1, 1).to(qpos.device)
            )
        )
        sim = Mock()
        sim.update = Mock()
        required = frozenset({"eef_waypoint", "batched", "empty_world"})
        for spec in specs:
            runner._run_adapter(writer, sim, robot, spec, [case], required)

        phases = {
            (r.algorithm_id, r.phase, r.status, r.failure_code) for r in runner.records
        }
        assert (
            "incapable",
            TrialPhase.AVAILABILITY,
            "unsupported",
            "unsupported_capability",
        ) in phases
        assert (
            "runtime_down",
            TrialPhase.AVAILABILITY,
            "unsupported",
            "runtime_unavailable",
        ) in phases
        assert any(
            r.algorithm_id == "broken"
            and r.phase is TrialPhase.MEASURED
            and r.failure_code == "planner_contract_error"
            for r in runner.records
        )
        assert any(
            r.algorithm_id == "capable"
            and r.phase is TrialPhase.COLD
            and r.outcomes == ()
            for r in runner.records
        )

        aggregates = aggregate_results(
            runner.records,
            list(runner.metadata.values()),
            [case],
            suite.protocol.measured_trials,
        )
        capable = next(
            row for row in aggregates["leaderboard"] if row["algorithm"] == "capable"
        )
        incapable = next(
            row for row in aggregates["leaderboard"] if row["algorithm"] == "incapable"
        )
        assert capable["eligible"] is True
        assert capable["overall_success_rate"] == pytest.approx(1.0)
        assert incapable["eligible"] is False
        assert any("missing required capabilities" in note for note in runner.notes)
    finally:
        for name in names:
            unregister_planner_adapter(name)
