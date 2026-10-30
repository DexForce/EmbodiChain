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
from types import SimpleNamespace

from embodichain_benchmarks.motion_generation.config import (
    FreeSpaceTrackCfg,
)
from embodichain_benchmarks.motion_generation.metrics._constraints import (
    match_constraints,
)
from embodichain_benchmarks.motion_generation.scenarios._free_space_sampling import (
    _project_constraints,
)


def _coupled_fk(qpos):
    """Two bounded joints couple translation and rotation independently."""
    poses = (
        torch.eye(4, dtype=qpos.dtype, device=qpos.device)
        .expand(*qpos.shape[:-1], 4, 4)
        .clone()
    )
    poses[..., 0, 3] = qpos[..., 0] + qpos[..., 1]
    angle = qpos[..., 1]
    poses[..., 0, 0] = poses[..., 1, 1] = angle.cos()
    poses[..., 0, 1], poses[..., 1, 0] = -angle.sin(), angle.sin()
    return poses


@pytest.mark.parametrize("modality", ["position", "orientation"])
def test_masked_baseline_ik_ignores_unused_pose_components(modality):
    from embodichain.compute.kinematics._masked_ik import _solve_masked_ik

    target = _coupled_fk(torch.tensor([[0.4, 0.4]]))
    if modality == "position":
        target[:, :3, :3] = torch.nan
    else:
        target[:, :3, 3] = torch.nan
    valid, qpos = _solve_masked_ik(
        target,
        torch.zeros(1, 2),
        -torch.ones(1, 2),
        torch.ones(1, 2),
        _coupled_fk,
        modality,
    )
    assert valid.tolist() == [True]
    if modality == "position":
        assert float(qpos.sum()) == pytest.approx(0.8, abs=0.005)
        # A full-pose completion at the identity would unnecessarily fix q1.
        assert float(qpos[0, 1]) > 0.1
    else:
        assert float(qpos[0, 1]) == pytest.approx(0.4, abs=0.05)


def _masked_adapter(kind, *, enabled=True):
    from embodichain_benchmarks.motion_generation.config import PlannerSpecCfg
    from embodichain_benchmarks.motion_generation.planners.base import PlannerContext

    robot = SimpleNamespace(
        compute_batch_fk=lambda *, qpos, **_: _coupled_fk(qpos),
        get_qpos_limits=lambda **_: torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]),
    )
    return kind(
        PlannerSpecCfg(
            id="baseline",
            adapter="test",
            role="diagnostic_baseline",
            config={"masked_constraints": enabled},
        ),
        PlannerContext(
            robot=robot,
            control_part="arm",
            device=torch.device("cpu"),
            sample_interval=32,
        ),
    )


@pytest.mark.parametrize("fail_joint", [False, True])
def test_toppra_mixed_sequence_uses_requested_joint_payload_and_masked_ik(fail_joint):
    from embodichain.lab.sim.motion.planners import PlanResult
    from embodichain_benchmarks.motion_generation.models import BenchmarkCase
    from embodichain_benchmarks.motion_generation.planners.toppra import ToppraAdapter

    adapter = _masked_adapter(ToppraAdapter)
    adapter._motion_limits = {
        "velocity": [1.0, 1.0],
        "acceleration": [1.0, 1.0],
        "jerk": None,
    }

    def generate(targets, options):
        positions = torch.stack(
            [options.start_qpos, *[target.qpos for target in targets]], dim=1
        )
        return PlanResult(
            success=torch.ones(1, dtype=torch.bool),
            positions=positions,
            dt=torch.ones(1, positions.shape[1]),
        )

    adapter.motion_generator = SimpleNamespace(generate=generate)
    goals = _coupled_fk(torch.tensor([[[0.4, 0.4], [0.0, -0.3], [0.0, 0.0]]]))
    goals[:, 0, :3, :3] = torch.nan
    goals[:, 1, :3, 3] = torch.nan
    goals[:, 2] = torch.nan
    joint_goals = torch.tensor(
        [[[float("nan"), float("nan")], [float("nan"), float("nan")], [-0.2, 0.1]]]
    )
    case = BenchmarkCase(
        "test",
        "free",
        "waypoint_path",
        "mixed",
        1,
        1,
        3,
        "direct",
        "nominal",
        torch.zeros(1, 2),
        goals,
        joint_goals,
        case_parameters={
            "motion_validity": "ordered_constraint_waypoints",
            "constraint_modalities": ["position", "orientation", "joint"],
        },
    )
    if fail_joint:
        joint_goals[:, -1] = 10.0
    assert adapter.supports_case(case) == (True, None)
    result = adapter.plan(case)
    if fail_joint:
        assert result.success.tolist() == [False]
        torch.testing.assert_close(result.positions, torch.zeros_like(result.positions))
        return
    assert result.success.tolist() == [True]
    qpos = result.positions[0]
    assert float(qpos[1].sum()) == pytest.approx(0.8, abs=0.005)
    assert float(qpos[2, 1]) == pytest.approx(-0.3, abs=0.05)
    torch.testing.assert_close(qpos[3], torch.tensor([-0.2, 0.1]))
    assert not _masked_adapter(ToppraAdapter, enabled=False).supports_case(case)[0]


def test_masked_ik_failure_does_not_return_an_unreachable_witness():
    from embodichain.compute.kinematics._masked_ik import _solve_masked_ik

    goal = _coupled_fk(torch.tensor([[10.0, 0.0], [0.4, 0.4]]))
    seed = torch.zeros(2, 2)
    valid, qpos = _solve_masked_ik(
        goal,
        seed,
        -torch.ones(1, 2),
        torch.ones(1, 2),
        _coupled_fk,
        "position",
        iterations=8,
    )
    assert valid.tolist() == [False, True]
    torch.testing.assert_close(qpos[0], seed[0])
    assert float(qpos[1].sum()) == pytest.approx(0.8, abs=0.005)


def test_masked_ik_orientation_near_pi():
    from embodichain.compute.kinematics._masked_ik import _solve_masked_ik

    target_angle = math.pi - 0.01
    goal = _coupled_fk(torch.tensor([[0.0, target_angle]]))
    valid, qpos = _solve_masked_ik(
        goal,
        torch.zeros(1, 2),
        -torch.full((1, 2), math.pi),
        torch.full((1, 2), math.pi),
        _coupled_fk,
        "orientation",
    )
    assert valid.tolist() == [True]
    difference = float(qpos[0, 1]) - target_angle
    assert abs(math.atan2(math.sin(difference), math.cos(difference))) <= 0.05


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["toppra", "curobo"])
def test_franka_masked_baseline_runtime(backend):
    """Actual FK, component IK/native TrajOpt, timing and independent scoring."""
    from dataclasses import replace
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain_robots import FrankaPandaCfg
    from embodichain_benchmarks.motion_generation.config import PlannerSpecCfg
    from embodichain_benchmarks.motion_generation.models import BenchmarkCase
    from embodichain_benchmarks.motion_generation.planners.base import PlannerContext
    from embodichain_benchmarks.motion_generation.planners.toppra import ToppraAdapter
    from embodichain_benchmarks.motion_generation.planners.curobo import CuroboAdapter

    sim = SimulationManager(
        SimulationManagerCfg(headless=True, sim_device="cuda", num_envs=1)
    )
    adapter = None
    try:
        robot = sim.add_robot(
            cfg=FrankaPandaCfg.from_dict(
                {"uid": "masked_baseline_franka", "robot_type": "panda"}
            )
        )
        sim.prepare()
        start = robot.get_qpos(name="arm").clone()
        offsets = torch.tensor(
            [
                [0.05, 0.03, -0.04, 0.03, 0.04, -0.03, 0.05],
                [-0.04, 0.04, 0.05, -0.04, -0.03, -0.04, 0.03],
                [0.02, -0.03, -0.03, 0.04, 0.05, -0.05, -0.04],
                [-0.03, 0.05, 0.04, 0.05, -0.04, -0.06, 0.04],
            ],
            device=start.device,
        )
        witness = start[:, None] + offsets[None]
        poses = robot.compute_batch_fk(qpos=witness, name="arm", to_matrix=True)
        case = BenchmarkCase(
            "test",
            "free",
            "waypoint_path",
            "mixed",
            1,
            1,
            4,
            "direct",
            "nominal",
            start,
            poses.clone(),
            witness.clone(),
            case_parameters={
                "motion_validity": "ordered_constraint_waypoints",
                "constraint_modalities": ["position", "orientation", "joint", "pose"],
            },
        )
        # Hide all Cartesian reference joints and inactive pose components.
        case.reference_qpos[:, [0, 1, 3]] = torch.nan
        case.target_waypoints[:, 0, :3, :3] = torch.nan
        case.target_waypoints[:, 1, :3, 3] = torch.nan
        case.target_waypoints[:, 2] = torch.nan
        kind = ToppraAdapter if backend == "toppra" else CuroboAdapter
        adapter = kind(
            PlannerSpecCfg(
                id=backend,
                adapter=backend,
                role="primary_baseline",
                config={
                    "masked_constraints": True,
                    "use_cuda_graph": False,
                    "warmup_iterations": 0,
                },
            ),
            PlannerContext(
                robot=robot,
                control_part="arm",
                device=start.device,
                sample_interval=256,
                joint_velocity_limit_rad_s=0.2,
                joint_acceleration_limit_rad_s2=0.5,
            ),
        )
        adapter.build()
        adapter.prepare(case)
        result = adapter.plan(case)
        assert result.success.tolist() == [True]
        achieved = robot.compute_batch_fk(
            qpos=result.positions, name="arm", to_matrix=True
        )
        score = match_constraints(
            result.positions[0],
            achieved[0],
            witness[0],
            poses[0],
            ["position", "orientation", "joint", "pose"],
            position_threshold_m=0.01,
            rotation_threshold_rad=0.1,
            joint_threshold_rad=0.01,
        )
        assert score["ordered_waypoints_reached"], score
        assert bool(torch.isfinite(result.positions).all())
        if backend == "toppra":
            impossible = torch.eye(4, device=start.device).reshape(1, 1, 4, 4)
            impossible[:, :, 0, 3] = 10.0
            failed = adapter.plan(
                replace(
                    case,
                    num_waypoints=1,
                    target_waypoints=impossible,
                    reference_qpos=witness[:, :1],
                    case_parameters={
                        "motion_validity": "ordered_constraint_waypoints",
                        "constraint_modalities": ["position"],
                    },
                )
            )
            assert failed.success.tolist() == [False]
            torch.testing.assert_close(
                failed.positions,
                start.to(failed.positions)[:, None].expand_as(failed.positions),
            )
    finally:
        if adapter is not None:
            adapter.close()
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "validity,modalities",
    [
        ("ordered_constraint_waypoints", ["position", "joint"]),
        ("ordered_joint_waypoints", ["joint", "joint"]),
    ],
)
def test_case_outcomes_preserve_native_samples_for_masked_and_common_inputs(
    validity: str, modalities: list[str], dtype: torch.dtype
) -> None:
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

    q = torch.zeros(1, 3, 7, dtype=dtype)
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
