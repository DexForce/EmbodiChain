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

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock, patch

import pytest
import torch

from embodichain.lab.sim.objects.robot import Robot

from embodichain.lab.sim.motion.planners.base_planner import (
    CollisionWorldInfo,
    PlanOptions,
)
from embodichain.lab.sim.motion.motion_generator import (
    MotionGenerator,
    MotionGenOptions,
)
from embodichain.lab.sim.motion.planners.trapezoidal_planner import (
    TrapezoidalPlanner,
    TrapezoidalPlanOptions,
)
from embodichain.lab.sim.motion.planners.utils import PlanState, PlanResult, MoveType

BATCH_SIZE = 2
CONTROLLED_DOF = 6
SAMPLE_COUNT = 8
STEP_DT = 0.05


def _collision_world_info(
    dynamic_entity_ids: tuple[str, ...] = (),
    *,
    entity_ids: tuple[str, ...] | None = None,
    batch_mode: Literal["shared", "per_env"] | None = "shared",
    supports_updates: bool = True,
) -> CollisionWorldInfo:
    """Build a valid collision-world contract for planner test doubles."""
    return CollisionWorldInfo(
        entity_ids=dynamic_entity_ids if entity_ids is None else entity_ids,
        dynamic_entity_ids=dynamic_entity_ids,
        batch_mode=batch_mode,
        supports_updates=supports_updates,
    )


def _timed_result(
    positions: torch.Tensor,
    *,
    success: bool | torch.Tensor = True,
    step_dt: float = STEP_DT,
) -> PlanResult:
    """Build a planner result that satisfies the explicit timing contract."""
    dt = torch.zeros(positions.shape[:2], device=positions.device)
    if positions.shape[1] > 1:
        dt[:, 1:] = step_dt
    return PlanResult(
        success=success,
        positions=positions,
        dt=dt,
    )


@pytest.mark.parametrize("sample_count", [None, 101])
def test_generate_preserves_trapezoidal_constraint_report(
    sample_count: int | None,
) -> None:
    generator = _trapezoidal_generator()
    planner = generator.planner
    targets = [
        PlanState.from_qpos(torch.tensor([qpos], dtype=torch.float64))
        for qpos in ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0])
    ]
    options = TrapezoidalPlanOptions(
        profile="double_s",
        stop_at_waypoints=False,
        blend_tolerance=0.1,
        sample_interval=101,
    )
    raw = planner.plan(targets, options)
    result = generator.generate(
        targets, MotionGenOptions(plan_opts=options, sample_count=sample_count)
    )
    assert result.constraint_report is not None
    assert torch.equal(result.positions, raw.positions)
    assert torch.equal(result.dt, raw.dt)
    for name, value in raw.constraint_report.items():
        assert torch.equal(result.constraint_report[name], value)


def _trapezoidal_generator(robot: object | None = None) -> MotionGenerator:
    """Build the real CPU planner behind the MotionGenerator facade."""
    planner = object.__new__(TrapezoidalPlanner)
    planner.cfg = SimpleNamespace(planner_type="trapezoidal")
    planner.device = torch.device("cpu")
    planner.robot = (
        robot
        if robot is not None
        else SimpleNamespace(
            joint_names=["first", "second"],
            cfg=SimpleNamespace(fpath="", joint_acceleration_limits=None),
            num_instances=1,
            get_qvel_limits=lambda **kwargs: torch.tensor([[0.2, 0.2]]),
        )
    )
    planner.robot.get_joint_motion_limits = (
        lambda **kwargs: Robot.get_joint_motion_limits(planner.robot, **kwargs)
    )
    generator = object.__new__(MotionGenerator)
    generator.planner = planner
    generator.robot = planner.robot
    generator.device = torch.device("cpu")
    return generator


def test_trapezoidal_generator_prepends_observed_start_for_sparse_joint_goal() -> None:
    """A joint-only backend must start from the observed robot configuration."""
    generator = _trapezoidal_generator()
    generator.interpolate_trajectory = Mock(
        side_effect=AssertionError("sparse joint targets must bypass interpolation")
    )

    start = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    goal = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
    result = generator.generate(
        [PlanState.from_qpos(goal)],
        MotionGenOptions(
            start_qpos=start,
            plan_opts=TrapezoidalPlanOptions(sample_interval=21, backend="torch"),
        ),
    )

    assert result.positions is not None
    torch.testing.assert_close(result.positions[:, 0], start)
    torch.testing.assert_close(result.positions[:, -1], goal)


def test_trapezoidal_generator_holds_stationary_sparse_joint_goal() -> None:
    """A stationary single goal still forms a valid two-waypoint hold."""
    generator = _trapezoidal_generator()

    start = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
    result = generator.generate(
        [PlanState.from_qpos(start)],
        MotionGenOptions(
            start_qpos=start,
            plan_opts=TrapezoidalPlanOptions(sample_interval=2, backend="torch"),
        ),
    )

    assert bool(result.success.all())
    assert result.positions is not None
    assert result.positions.shape[1] == 2
    torch.testing.assert_close(
        result.positions, start.unsqueeze(1).expand_as(result.positions)
    )
    assert result.velocities is not None
    assert result.accelerations is not None
    torch.testing.assert_close(result.velocities, torch.zeros_like(result.velocities))
    torch.testing.assert_close(
        result.accelerations, torch.zeros_like(result.accelerations)
    )


def test_trapezoidal_generator_covers_dense_ik_waypoints() -> None:
    """Quantity sampling grows to retain all Cartesian-to-joint waypoints."""
    generator = _trapezoidal_generator()
    generator.robot = SimpleNamespace(
        compute_fk=lambda *, qpos, name, to_matrix: torch.eye(
            4, dtype=qpos.dtype, device=qpos.device
        ).expand(qpos.shape[0], -1, -1)
    )
    dense_qpos = torch.linspace(0.0, 0.5, 6, dtype=torch.float64).reshape(1, 6, 1)
    dense_qpos = dense_qpos.expand(-1, -1, 2).clone()
    generator._interpolate_waypoints = Mock(
        return_value=(dense_qpos, None, torch.ones(1, dtype=torch.bool))
    )

    result = generator.generate(
        [PlanState.from_xpos(torch.eye(4, dtype=torch.float64).unsqueeze(0))],
        MotionGenOptions(
            start_qpos=torch.zeros(1, 2, dtype=torch.float64),
            sample_count=3,
            is_interpolate=True,
        ),
    )

    assert result.positions is not None
    assert result.positions.shape[1] == dense_qpos.shape[1]


def test_trapezoidal_generator_preserves_native_derivatives() -> None:
    """Native Trapezoidal samples and derivatives survive normalization."""
    generator = _trapezoidal_generator()
    start = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    goal = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
    target_states = [PlanState.from_qpos(start), PlanState.from_qpos(goal)]
    plan_options = TrapezoidalPlanOptions(sample_interval=21, backend="torch")
    expected = generator.planner.plan(target_states, plan_options)

    result = generator.generate(
        target_states,
        MotionGenOptions(
            start_qpos=start,
            sample_count=8,
            plan_opts=plan_options,
        ),
    )

    assert result.positions is not None
    assert expected.positions is not None
    torch.testing.assert_close(result.positions, expected.positions)
    torch.testing.assert_close(result.dt, expected.dt)
    assert result.velocities is not None
    assert expected.velocities is not None
    torch.testing.assert_close(result.velocities, expected.velocities)
    assert result.accelerations is not None
    assert expected.accelerations is not None
    torch.testing.assert_close(result.accelerations, expected.accelerations)


@pytest.mark.parametrize("change", ["resample", "hold_failed_rows", "preserve"])
def test_generate_invalidates_report_only_when_trajectory_changes(change: str) -> None:
    raw = _timed_result(
        torch.ones(BATCH_SIZE, 5, CONTROLLED_DOF),
        success=torch.tensor([True, change != "hold_failed_rows"]),
    )
    report = {"within_limits": torch.ones(BATCH_SIZE, dtype=torch.bool)}
    raw.constraint_report = report
    generator = _mock_generator(result=raw, preserve_plan_samples=change == "preserve")
    result = generator.generate(
        [PlanState.from_xpos(torch.eye(4).repeat(BATCH_SIZE, 1, 1))],
        MotionGenOptions(
            sample_count=8 if change != "hold_failed_rows" else 5,
            start_qpos=torch.zeros(BATCH_SIZE, CONTROLLED_DOF),
        ),
    )
    assert raw.constraint_report is report
    if change == "preserve":
        assert result.constraint_report is report
    else:
        assert result.constraint_report is None
    if change == "resample":
        assert result.positions.shape[1] == 8
    if change == "hold_failed_rows":
        assert torch.count_nonzero(result.positions[1]) == 0
        assert torch.all(raw.positions[1] == 1.0)


@pytest.mark.parametrize(
    ("sample_count", "preserve_samples"), [(5, False), (8, False), (8, True)]
)
def test_preserving_failed_trajectories_keeps_report_unless_resampled(
    sample_count: int, preserve_samples: bool
) -> None:
    raw = _timed_result(
        torch.ones(BATCH_SIZE, 5, CONTROLLED_DOF),
        success=torch.tensor([True, False]),
    )
    report = {"within_limits": torch.tensor([True, False])}
    raw.constraint_report = report
    generator = _mock_generator(result=raw, preserve_plan_samples=preserve_samples)
    generator.planner.preserve_failed_plan_positions = True

    result = generator.generate(
        [PlanState.from_xpos(torch.eye(4).repeat(BATCH_SIZE, 1, 1))],
        MotionGenOptions(
            sample_count=sample_count,
            start_qpos=torch.zeros(BATCH_SIZE, CONTROLLED_DOF),
        ),
    )

    assert torch.equal(result.success, raw.success)
    assert torch.all(result.positions[1] == 1.0)
    assert raw.constraint_report is report
    torch.testing.assert_close(result.duration, raw.duration)
    resampled = sample_count != 5 and not preserve_samples
    assert result.positions.shape[1] == (sample_count if resampled else 5)
    if resampled:
        assert result.constraint_report is None
    else:
        assert result.constraint_report is report


class _DirectCartesianPlanner:
    """Fake backend that consumes raw Cartesian targets (like cuRobo).

    Used to verify ``MotionGenerator`` skips pre-interpolation and forwards the
    runtime context through the generic capability hooks rather than a
    planner-class special case.
    """

    supported_move_types = frozenset({MoveType.EEF_MOVE})
    preserve_plan_samples = True

    def supports_move_type(self, move_type: MoveType) -> bool:
        return move_type in self.supported_move_types

    def default_plan_options(self) -> PlanOptions:
        return PlanOptions()

    def with_motion_context(self, options, *, start_qpos, control_part):
        self.received = (start_qpos.clone(), control_part)
        return options

    def plan(self, target_states, options):
        self.target_states = target_states
        return _timed_result(
            torch.zeros(1, 3, 2),
            success=torch.tensor([True]),
        )


def test_direct_cartesian_planner_skips_preinterpolation_without_mutating_options():
    planner = _DirectCartesianPlanner()
    generator = object.__new__(MotionGenerator)
    generator.planner = planner
    generator.device = torch.device("cpu")
    start = torch.tensor([[0.1, -0.2]])
    goal = PlanState.from_xpos(torch.eye(4).unsqueeze(0))

    options = MotionGenOptions(
        start_qpos=start,
        control_part="arm",
        is_interpolate=True,
    )
    result = generator.generate([goal], options)

    assert result.success.item()
    # The original EEF target reaches the planner unchanged - no IK, no
    # pre-interpolation, no start-pose prepend.
    assert planner.target_states[0].move_type is MoveType.EEF_MOVE
    assert torch.equal(planner.target_states[0].xpos, goal.xpos)
    assert options.is_interpolate is True
    # Runtime context is forwarded through the generic hook.
    assert torch.equal(planner.received[0], start)
    assert planner.received[1] == "arm"


def test_direct_cartesian_planner_rejects_joint_targets():
    planner = _DirectCartesianPlanner()
    generator = object.__new__(MotionGenerator)
    generator.planner = planner
    generator.device = torch.device("cpu")

    with pytest.raises(ValueError, match="JOINT_MOVE"):
        generator.generate(
            [PlanState.from_qpos(torch.zeros(1, 2))],
            MotionGenOptions(plan_opts=PlanOptions()),
        )


def test_motion_generator_dispatches_heterogeneous_waypoints_by_capability():
    planner = Mock()
    planner.supports_heterogeneous_waypoints = True
    planner.supports_move_type.side_effect = lambda move_type: move_type in {
        MoveType.EEF_MOVE,
        MoveType.JOINT_MOVE,
    }
    planner.preserve_plan_samples = True
    planner.default_plan_options.return_value = PlanOptions()
    planner.with_motion_context.side_effect = (
        lambda options, *, start_qpos, control_part: options
    )
    planner.plan.return_value = PlanResult(
        success=torch.ones(1, dtype=torch.bool),
        positions=torch.zeros(1, 5, 2),
        dt=torch.full((1, 5), 0.01),
    )
    generator = object.__new__(MotionGenerator)
    generator.planner = planner
    generator.device = torch.device("cpu")
    targets = [
        PlanState.from_xpos(torch.eye(4).unsqueeze(0)),
        PlanState.from_qpos(torch.zeros(1, 2)),
    ]

    result = generator.generate(
        targets,
        MotionGenOptions(start_qpos=torch.zeros(1, 2), control_part="arm"),
    )

    assert result.success.all().item()
    assert planner.plan.call_args.kwargs["target_states"] is targets


def test_motion_generator_rejects_heterogeneous_waypoints_without_capability():
    planner = _DirectCartesianPlanner()
    generator = object.__new__(MotionGenerator)
    generator.planner = planner
    generator.device = torch.device("cpu")

    with pytest.raises(ValueError, match="does not support heterogeneous"):
        generator.generate(
            [
                PlanState.from_xpos(torch.eye(4).unsqueeze(0)),
                PlanState.from_qpos(torch.zeros(1, 2)),
            ],
            MotionGenOptions(start_qpos=torch.zeros(1, 2), control_part="arm"),
        )


def test_bind_collision_world_copies_caller_options() -> None:
    planner = Mock()
    planner.collision_world_info = _collision_world_info(("obstacle",))
    original = PlanOptions()
    obstacle_pose = torch.eye(4).unsqueeze(0)

    def bind(options, *, obstacle_poses):
        options.bound_obstacle_poses = obstacle_poses
        return options

    planner.with_collision_world.side_effect = bind
    generator = object.__new__(MotionGenerator)
    generator.planner = planner

    bound = generator.bind_collision_world(
        original,
        obstacle_poses={"obstacle": obstacle_pose},
    )

    assert generator.supports_dynamic_collision_world is True
    assert bound is not original
    assert not hasattr(original, "bound_obstacle_poses")
    assert bound.bound_obstacle_poses["obstacle"] is obstacle_pose
    planner.with_collision_world.assert_called_once()


@pytest.mark.parametrize(
    ("configured_ids", "obstacle_poses", "expected"),
    [
        (("cube", "tray"), {"cube": torch.eye(4).unsqueeze(0)}, "missing"),
        (
            ("cube",),
            {
                "cube": torch.eye(4).unsqueeze(0),
                "tray": torch.eye(4).unsqueeze(0),
            },
            "extra",
        ),
    ],
)
def test_bind_collision_world_requires_exact_planner_entity_ids(
    configured_ids, obstacle_poses, expected
) -> None:
    planner = Mock()
    planner.collision_world_info = _collision_world_info(configured_ids)
    generator = object.__new__(MotionGenerator)
    generator.planner = planner

    with pytest.raises(ValueError, match=expected):
        generator.bind_collision_world(None, obstacle_poses=obstacle_poses)

    planner.with_collision_world.assert_not_called()


def test_bind_collision_world_rejects_extra_ids_in_caller_options() -> None:
    planner = Mock()
    planner.collision_world_info = _collision_world_info(("cube",))
    generator = object.__new__(MotionGenerator)
    generator.planner = planner
    options = PlanOptions()
    options.dynamic_obstacle_poses = {"legacy_cube": torch.eye(4).unsqueeze(0)}

    with pytest.raises(ValueError, match="Caller planning options.*legacy_cube"):
        generator.bind_collision_world(
            options,
            obstacle_poses={"cube": torch.eye(4).unsqueeze(0)},
        )

    planner.with_collision_world.assert_not_called()


def test_bind_collision_world_rejects_ids_injected_by_backend() -> None:
    planner = Mock()
    planner.collision_world_info = _collision_world_info(("cube",))

    def bind(options, *, obstacle_poses):
        options.dynamic_obstacle_poses = {
            **obstacle_poses,
            "legacy_cube": torch.eye(4).unsqueeze(0),
        }
        return options

    planner.with_collision_world.side_effect = bind
    generator = object.__new__(MotionGenerator)
    generator.planner = planner

    with pytest.raises(ValueError, match="Bound dynamic collision.*legacy_cube"):
        generator.bind_collision_world(
            PlanOptions(),
            obstacle_poses={"cube": torch.eye(4).unsqueeze(0)},
        )


def test_bind_collision_world_allows_none_for_empty_configured_world() -> None:
    planner = Mock()
    planner.collision_world_info = _collision_world_info()
    planner.default_plan_options.return_value = PlanOptions()

    def bind(options, *, obstacle_poses):
        assert obstacle_poses == {}
        options.dynamic_obstacle_poses = None
        return options

    planner.with_collision_world.side_effect = bind
    generator = object.__new__(MotionGenerator)
    generator.planner = planner

    bound = generator.bind_collision_world(None, obstacle_poses={})

    assert bound.dynamic_obstacle_poses is None


def test_bind_collision_world_rejects_non_string_option_keys() -> None:
    planner = Mock()
    planner.collision_world_info = _collision_world_info()
    generator = object.__new__(MotionGenerator)
    generator.planner = planner
    options = PlanOptions()
    options.dynamic_obstacle_poses = {1: torch.eye(4).unsqueeze(0)}

    with pytest.raises(TypeError, match="keys must be non-empty strings"):
        generator.bind_collision_world(options, obstacle_poses={})

    planner.with_collision_world.assert_not_called()


def test_motion_generator_exposes_collision_integration_metadata() -> None:
    planner = Mock()
    info = _collision_world_info(
        ("cube", "tray"),
        entity_ids=("cube", "tray", "table"),
        batch_mode="per_env",
    )
    planner.collision_world_info = info
    generator = object.__new__(MotionGenerator)
    generator.planner = planner

    assert generator.collision_world_info is info
    assert generator.dynamic_collision_entity_ids == ("cube", "tray")
    assert generator.collision_world_entity_ids == ("cube", "tray", "table")
    assert generator.collision_world_batch_mode == "per_env"


def test_motion_generator_rejects_invalid_collision_world_contract() -> None:
    planner = Mock()
    planner.collision_world_info = object()
    generator = object.__new__(MotionGenerator)
    generator.planner = planner

    with pytest.raises(TypeError, match="CollisionWorldInfo"):
        _ = generator.collision_world_info


def test_bind_collision_world_rejects_unsupported_planner() -> None:
    planner = Mock()
    planner.collision_world_info = None
    generator = object.__new__(MotionGenerator)
    generator.planner = planner

    with pytest.raises(ValueError, match="does not support"):
        generator.bind_collision_world(
            PlanOptions(),
            obstacle_poses={"obstacle": torch.eye(4).unsqueeze(0)},
        )

    assert generator.supports_dynamic_collision_world is False
    planner.with_collision_world.assert_not_called()


def test_bind_collision_world_uses_backend_default_options() -> None:
    planner = Mock()
    planner.collision_world_info = _collision_world_info(("obstacle",))
    defaults = PlanOptions()
    planner.default_plan_options.return_value = defaults
    planner.with_collision_world.return_value = defaults
    generator = object.__new__(MotionGenerator)
    generator.planner = planner

    bound = generator.bind_collision_world(
        None,
        obstacle_poses={"obstacle": torch.eye(4).unsqueeze(0)},
    )

    assert bound is defaults
    planner.default_plan_options.assert_called_once_with()


def _mock_planner(b=3, n=15, dofs=6):
    planner = Mock()
    planner.cfg.planner_type = "toppra"
    planner.supported_move_types = frozenset({MoveType.JOINT_MOVE})
    planner.supports_move_type.side_effect = (
        lambda move_type: move_type in planner.supported_move_types
    )
    planner.robot.num_instances = b
    planner.robot.device = torch.device("cpu")
    planner.plan.return_value = _timed_result(
        torch.zeros(b, n, dofs),
        success=torch.ones(b, dtype=torch.bool),
    )
    planner.preserve_plan_samples = False
    planner.default_plan_options.return_value = PlanOptions()
    planner.with_motion_context.side_effect = (
        lambda options, *, start_qpos, control_part: options
    )
    return planner


def test_resolve_trapezoidal_limits_without_sample_count() -> None:
    planner = Mock()
    planner.cfg.planner_type = "trapezoidal"
    generator = object.__new__(MotionGenerator)
    generator.planner = planner

    options = generator.resolve_plan_options(
        plan_opts=None,
        sample_count=None,
        velocity_limit=0.01,
        acceleration_limit=0.02,
    )

    assert isinstance(options, TrapezoidalPlanOptions)
    assert options.constraints["velocity"] == 0.01
    assert options.constraints["acceleration"] == 0.02


@pytest.mark.parametrize("planner_type", ["toppra", "trapezoidal"])
def test_sample_count_preserves_robot_limit_resolution(planner_type: str) -> None:
    generator = object.__new__(MotionGenerator)
    generator.planner = SimpleNamespace(cfg=SimpleNamespace(planner_type=planner_type))
    options = generator.resolve_plan_options(None, sample_count=100, velocity_limit=0.1)
    assert options.constraints["velocity"] == 0.1
    assert options.constraints["acceleration"] is None


def _mock_generator(
    *,
    batch_size: int = 2,
    controlled_dof: int = 6,
    supported_move_types: frozenset[MoveType] = frozenset(
        {MoveType.EEF_MOVE, MoveType.JOINT_MOVE}
    ),
    preserve_plan_samples: bool = False,
    result: PlanResult | None = None,
) -> MotionGenerator:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.num_instances = batch_size
    robot.get_solver.return_value = SimpleNamespace(supports_continuous_batch_ik=False)
    robot.compute_batch_ik.side_effect = lambda *, pose, **kwargs: (
        torch.ones(pose.shape[:2], dtype=torch.bool),
        torch.zeros(*pose.shape[:2], controlled_dof),
    )
    planner = Mock()
    planner.cfg.planner_type = "toppra"
    planner.robot = robot
    planner.supported_move_types = supported_move_types
    planner.supports_move_type.side_effect = (
        lambda move_type: move_type in supported_move_types
    )
    planner.preserve_plan_samples = preserve_plan_samples
    planner.default_plan_options.return_value = PlanOptions()
    planner.with_motion_context.side_effect = (
        lambda options, *, start_qpos, control_part: options
    )
    planner.plan.return_value = result or _timed_result(
        torch.zeros(batch_size, 5, controlled_dof),
        success=torch.ones(batch_size, dtype=torch.bool),
    )
    generator = object.__new__(MotionGenerator)
    generator.planner = planner
    generator.robot = robot
    generator.device = torch.device("cpu")
    return generator


class TestGenerateBatched:
    def test_generate_passes_batched_states_to_planner(self):
        planner = _mock_planner()
        mg = MotionGenerator.__new__(MotionGenerator)
        mg.planner = planner
        mg.robot = planner.robot
        mg.device = torch.device("cpu")

        B, dofs = 3, 6
        states = [
            PlanState.from_qpos(torch.zeros(B, dofs)),
            PlanState.from_qpos(torch.ones(B, dofs)),
        ]
        r = mg.generate(states, MotionGenOptions(plan_opts=Mock()))
        assert r.success.shape == (B,)
        assert r.positions.shape == (B, 15, dofs)
        # planner.plan received the batched states list as-is
        _, kwargs = planner.plan.call_args
        assert (
            kwargs["target_states"] is states or planner.plan.call_args[0][0] is states
        )

    def test_joint_only_planner_preinterpolates_cartesian_targets(self):
        planner = _mock_planner(b=1, n=2, dofs=6)
        mg = MotionGenerator.__new__(MotionGenerator)
        mg.planner = planner
        mg.robot = planner.robot
        mg.device = torch.device("cpu")
        interpolated_qpos = torch.zeros(1, 2, 6)
        mg.robot.compute_fk.return_value = torch.eye(4).unsqueeze(0)
        mg._interpolate_waypoints = Mock(
            return_value=(interpolated_qpos, None, torch.ones(1, dtype=torch.bool))
        )

        mg.generate(
            [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
            MotionGenOptions(
                is_interpolate=True,
                plan_opts=PlanOptions(),
                start_qpos=torch.zeros(1, 6),
            ),
        )

        target_states = planner.plan.call_args.kwargs["target_states"]
        assert all(target.move_type is MoveType.JOINT_MOVE for target in target_states)


class TestInterpolateBatched:
    def test_interpolate_joint_space_batched(self):
        planner = _mock_planner(b=3, n=10, dofs=6)
        mg = MotionGenerator.__new__(MotionGenerator)
        mg.planner = planner
        mg.robot = planner.robot
        mg.device = torch.device("cpu")
        B, N, D = 3, 4, 6
        qpos_list = torch.zeros(B, N, D)
        qpos_interpolated, _ = mg.interpolate_trajectory(
            control_part="arm",
            xpos_list=None,
            qpos_list=qpos_list,
            options=MotionGenOptions(is_linear=False, interpolate_nums=10),
        )
        assert qpos_interpolated.shape[0] == B


class TestMotionStrategy:
    def test_options_accept_only_declared_strategy_values(self):
        assert MotionGenOptions(strategy="motion_gen").strategy == "motion_gen"
        assert MotionGenOptions(strategy="ik_interp").strategy == "ik_interp"
        with pytest.raises(ValueError, match="strategy"):
            MotionGenOptions(strategy="planner")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="interpolation_dt"):
            MotionGenOptions(interpolation_dt=0.0)

    def test_ik_interp_rejects_missing_timing(self):
        generator = _mock_generator()
        with pytest.raises(ValueError, match="explicit interpolation_dt"):
            generator.generate(
                [PlanState.from_qpos(torch.ones(BATCH_SIZE, CONTROLLED_DOF))],
                MotionGenOptions(
                    strategy="ik_interp",
                    sample_count=SAMPLE_COUNT,
                    start_qpos=torch.zeros(BATCH_SIZE, CONTROLLED_DOF),
                    control_part="arm",
                ),
            )

    def test_ik_interp_solves_batched_poses_without_calling_backend(self):
        generator = _mock_generator()
        generator.robot.compute_batch_ik.side_effect = lambda *, pose, **kwargs: (
            torch.tensor([[1], [0]], dtype=torch.int64).expand(*pose.shape[:2]),
            torch.ones(*pose.shape[:2], CONTROLLED_DOF),
        )
        start = torch.zeros(BATCH_SIZE, CONTROLLED_DOF)
        start[1] = 0.5
        targets = [PlanState.from_xpos(torch.eye(4).repeat(BATCH_SIZE, 1, 1))]

        result = generator.generate(
            targets,
            MotionGenOptions(
                strategy="ik_interp",
                sample_count=SAMPLE_COUNT,
                start_qpos=start,
                control_part="arm",
                interpolation_dt=STEP_DT,
            ),
        )

        assert isinstance(result.success, torch.Tensor)
        assert result.success.tolist() == [True, False]
        assert result.positions is not None
        assert result.positions.shape == (
            BATCH_SIZE,
            SAMPLE_COUNT,
            CONTROLLED_DOF,
        )
        assert torch.allclose(
            result.positions[1],
            start[1].unsqueeze(0).expand(SAMPLE_COUNT, -1),
        )
        generator.planner.plan.assert_not_called()

    def test_linear_cartesian_motion_grounds_every_output_sample_with_ik(self):
        generator = _mock_generator()

        def encode_position(
            pose: torch.Tensor,
            name: str,
            joint_seed: torch.Tensor,
            continuous: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            qpos = joint_seed.clone()
            qpos[..., :3] = pose[..., :3, 3]
            return torch.ones(pose.shape[:2], dtype=torch.bool), qpos

        generator.robot.compute_batch_ik.side_effect = encode_position
        weights = torch.linspace(1.0 / (SAMPLE_COUNT - 1), 1.0, SAMPLE_COUNT - 1)
        targets = []
        for weight in weights:
            pose = torch.eye(4).repeat(BATCH_SIZE, 1, 1)
            pose[:, 0, 3] = weight
            targets.append(PlanState.from_xpos(pose))

        result = generator.generate(
            targets,
            MotionGenOptions(
                strategy="ik_interp",
                sample_count=SAMPLE_COUNT,
                start_qpos=torch.zeros(BATCH_SIZE, CONTROLLED_DOF),
                control_part="arm",
                is_linear=True,
                interpolation_dt=STEP_DT,
                preserve_cartesian_samples=True,
            ),
        )

        assert result.positions is not None
        assert result.positions.shape == (
            BATCH_SIZE,
            SAMPLE_COUNT,
            CONTROLLED_DOF,
        )
        expected_x = torch.linspace(0.0, 1.0, SAMPLE_COUNT)
        assert torch.allclose(
            result.positions[:, :, 0], expected_x.expand(BATCH_SIZE, -1)
        )
        assert generator.robot.compute_batch_ik.call_count == 2
        calls = generator.robot.compute_batch_ik.call_args_list
        assert sum(call.kwargs["pose"].shape[1] for call in calls) == SAMPLE_COUNT - 1
        generator.robot.compute_ik.assert_not_called()
        generator.planner.plan.assert_not_called()

    def test_motion_gen_delegates_and_resamples_backend_result(self):
        raw_sample_count = 5
        generator = _mock_generator(
            result=_timed_result(
                torch.zeros(
                    BATCH_SIZE,
                    raw_sample_count,
                    CONTROLLED_DOF,
                ),
            )
        )
        targets = [PlanState.from_xpos(torch.eye(4).repeat(BATCH_SIZE, 1, 1))]

        result = generator.generate(
            targets,
            MotionGenOptions(
                strategy="motion_gen",
                sample_count=SAMPLE_COUNT,
                start_qpos=torch.zeros(BATCH_SIZE, CONTROLLED_DOF),
                control_part="arm",
            ),
        )

        assert result.positions is not None
        assert result.positions.shape == (
            BATCH_SIZE,
            SAMPLE_COUNT,
            CONTROLLED_DOF,
        )
        assert result.dt is not None
        assert result.duration is not None
        assert result.dt.shape == (BATCH_SIZE, SAMPLE_COUNT)
        assert result.duration.tolist() == pytest.approx(
            [STEP_DT * (raw_sample_count - 1)] * BATCH_SIZE
        )
        generator.planner.plan.assert_called_once()

    def test_motion_gen_preserves_backend_samples_when_required(self):
        raw_sample_count = 5
        generator = _mock_generator(
            preserve_plan_samples=True,
            result=_timed_result(
                torch.zeros(
                    BATCH_SIZE,
                    raw_sample_count,
                    CONTROLLED_DOF,
                ),
            ),
        )

        result = generator.generate(
            [PlanState.from_xpos(torch.eye(4).repeat(BATCH_SIZE, 1, 1))],
            MotionGenOptions(
                strategy="motion_gen",
                sample_count=SAMPLE_COUNT,
                start_qpos=torch.zeros(BATCH_SIZE, CONTROLLED_DOF),
                control_part="arm",
            ),
        )

        assert result.positions is not None
        assert result.positions.shape[1] == raw_sample_count

    def test_generate_does_not_mutate_caller_plan_options(self):
        generator = _mock_generator()
        caller_options = PlanOptions()
        options = MotionGenOptions(
            strategy="motion_gen",
            sample_count=SAMPLE_COUNT,
            start_qpos=torch.zeros(BATCH_SIZE, CONTROLLED_DOF),
            control_part="arm",
            plan_opts=caller_options,
        )

        generator.generate(
            [PlanState.from_xpos(torch.eye(4).repeat(BATCH_SIZE, 1, 1))],
            options,
        )

        forwarded = generator.planner.with_motion_context.call_args.args[0]
        assert forwarded is not caller_options
        assert forwarded is not options.plan_opts


class TestNormalizedPlanResult:
    def test_non_finite_positions_are_rejected(self):
        positions = torch.zeros(BATCH_SIZE, 5, CONTROLLED_DOF)
        positions[0, 0, 0] = float("nan")
        generator = _mock_generator(result=_timed_result(positions))

        with pytest.raises(ValueError, match="non-finite"):
            generator.generate(
                [PlanState.from_xpos(torch.eye(4).repeat(BATCH_SIZE, 1, 1))],
                MotionGenOptions(
                    start_qpos=torch.zeros(BATCH_SIZE, CONTROLLED_DOF),
                    control_part="arm",
                ),
            )

    def test_missing_positions_are_rejected(self):
        generator = _mock_generator(
            result=PlanResult(
                success=torch.ones(BATCH_SIZE, dtype=torch.bool),
                positions=None,
            )
        )

        with pytest.raises(ValueError, match="positions"):
            generator.generate(
                [PlanState.from_xpos(torch.eye(4).repeat(BATCH_SIZE, 1, 1))],
                MotionGenOptions(
                    start_qpos=torch.zeros(BATCH_SIZE, CONTROLLED_DOF),
                    control_part="arm",
                ),
            )

    def test_failed_rows_hold_start_qpos(self):
        positions = torch.zeros(BATCH_SIZE, 5, CONTROLLED_DOF)
        positions[1] = 1.0
        generator = _mock_generator(
            result=_timed_result(
                positions,
                success=torch.tensor([True, False]),
            )
        )
        start = torch.zeros(BATCH_SIZE, CONTROLLED_DOF)
        start[1] = 0.5

        result = generator.generate(
            [PlanState.from_xpos(torch.eye(4).repeat(BATCH_SIZE, 1, 1))],
            MotionGenOptions(start_qpos=start, control_part="arm"),
        )

        assert result.positions is not None
        assert torch.allclose(
            result.positions[1],
            start[1].unsqueeze(0).expand(positions.shape[1], -1),
        )


def test_ik_interpolation_materializes_reference_velocity() -> None:
    generator = _mock_generator()
    start = torch.zeros(BATCH_SIZE, CONTROLLED_DOF)
    result = generator.generate(
        [PlanState.from_qpos(torch.ones_like(start))],
        MotionGenOptions(
            strategy="ik_interp",
            sample_count=5,
            start_qpos=start,
            interpolation_dt=0.25,
        ),
    )
    assert result.velocities is not None
    torch.testing.assert_close(result.velocities, torch.ones_like(result.positions))


def test_normalization_preserves_native_derivatives_and_zeros_failed_rows() -> None:
    generator = _mock_generator()
    positions = (
        torch.arange(3.0).view(1, 3, 1).expand(BATCH_SIZE, -1, CONTROLLED_DOF).clone()
    )
    native = torch.full_like(positions, 0.3)
    result = generator._normalize_plan_result(
        PlanResult(
            success=torch.tensor([True, False]),
            positions=positions,
            velocities=native,
            dt=torch.tensor([[0.0, 1.0, 1.0]]).expand(BATCH_SIZE, -1),
        ),
        [PlanState.from_qpos(positions[:, -1])],
        MotionGenOptions(start_qpos=positions[:, 0]),
    )
    torch.testing.assert_close(result.velocities[0], native[0])
    assert torch.equal(result.velocities[1], torch.zeros_like(native[1]))


def test_normalization_resamples_by_time_and_recomputes_derivatives() -> None:
    generator = _mock_generator()
    positions = (
        torch.tensor([0.0, 1.0, 2.0])
        .view(1, 3, 1)
        .expand(BATCH_SIZE, -1, CONTROLLED_DOF)
        .clone()
    )
    result = generator._normalize_plan_result(
        PlanResult(
            success=True,
            positions=positions,
            velocities=torch.full_like(positions, 99.0),
            dt=torch.tensor([[0.0, 0.75, 0.25]]).expand(BATCH_SIZE, -1),
        ),
        [PlanState.from_qpos(positions[:, -1])],
        MotionGenOptions(start_qpos=positions[:, 0], sample_count=5),
    )
    torch.testing.assert_close(
        result.positions[0, :, 0], torch.tensor([0.0, 1 / 3, 2 / 3, 1.0, 2.0])
    )
    assert result.velocities is not None
    torch.testing.assert_close(result.velocities[0, 1, 0], torch.tensor(4 / 3))


class _CartesianRobot:
    """CPU kinematics double: one joint translates the TCP along x."""

    joint_names = ["joint"]
    cfg = SimpleNamespace(fpath="", joint_acceleration_limits=None)

    device = torch.device("cpu")
    num_instances = 2

    def get_joint_ids(self, name: str) -> list[int]:
        return [0]

    def get_qvel_limits(self, *, joint_ids: list[int]) -> torch.Tensor:
        return torch.full((self.num_instances, 1), 0.2)[:, joint_ids]

    def get_solver(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(supports_continuous_batch_ik=False)

    def compute_fk(self, *, qpos: torch.Tensor, **kwargs: object) -> torch.Tensor:
        pose = torch.eye(4, device=qpos.device).repeat(qpos.shape[0], 1, 1)
        pose[:, 0, 3] = qpos[:, 0]
        return pose

    def compute_batch_fk(self, *, qpos: torch.Tensor, **kwargs: object) -> torch.Tensor:
        pose = torch.eye(4, device=qpos.device).repeat(*qpos.shape[:2], 1, 1)
        pose[..., 0, 3] = qpos[..., 0]
        return pose

    def compute_ik(
        self, *, pose: torch.Tensor, joint_seed: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = torch.ones(pose.shape[0], dtype=torch.bool, device=pose.device)
        valid[1] = pose[1, 0, 3] < 0.75
        return valid, pose[:, 0, 3:4].clone()

    def compute_batch_ik(
        self, *, pose: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = torch.ones(pose.shape[:2], dtype=torch.bool)
        valid[1] = pose[1, :, 0, 3] < 0.75
        return valid, pose[..., 0, 3:4].clone()


@pytest.mark.parametrize(
    "batch_size, count", [(1, 1), (1, 7), (1, 65), (2, 9), (16, 65), (64, 65)]
)
def test_cartesian_ik_batches_both_environments_and_waypoints(
    batch_size: int, count: int
) -> None:
    generator = _mock_generator(batch_size=batch_size, controlled_dof=1)
    poses = torch.eye(4).repeat(batch_size, count, 1, 1)
    poses[..., 0, 3] = torch.linspace(0.1, 0.5, count)
    generator.robot.compute_batch_ik.side_effect = lambda *, pose, **kwargs: (
        torch.ones(pose.shape[:2], dtype=torch.bool),
        pose[..., 0, 3:4].clone(),
    )

    success, positions = generator._solve_cartesian_waypoints(
        poses, torch.zeros(batch_size, 1), "arm"
    )

    assert success.all()
    torch.testing.assert_close(positions, poses[..., 0, 3:4])
    calls = generator.robot.compute_batch_ik.call_args_list
    assert sum(call.kwargs["pose"].shape[1] for call in calls) == count
    assert all(call.kwargs["pose"].shape[0] == batch_size for call in calls)
    assert all(call.kwargs["pose"].shape[1] <= 32 for call in calls)
    assert all(call.kwargs["pose"].numel() // 16 <= 512 for call in calls)
    if count > 2:
        assert len(calls) < count
    generator.robot.compute_ik.assert_not_called()


def test_independent_batch_ik_interpolates_seeds_and_carries_chunk_end() -> None:
    generator = _mock_generator(controlled_dof=1)
    start = torch.tensor([[0.0], [10.0]])
    poses = torch.eye(4).repeat(2, 9, 1, 1)
    poses[..., 0, 3] = start + torch.arange(1, 10)
    generator.robot.compute_batch_ik.side_effect = lambda *, pose, **kwargs: (
        torch.ones(pose.shape[:2], dtype=torch.bool),
        pose[..., 0, 3:4].clone(),
    )
    with patch("embodichain.lab.sim.motion.motion_generator._MAX_IK_BATCH_TARGETS", 8):
        success, positions = generator._solve_cartesian_waypoints(poses, start, "arm")

    assert success.all()
    torch.testing.assert_close(positions, poses[..., 0, 3:4])
    calls = generator.robot.compute_batch_ik.call_args_list
    assert [call.kwargs["pose"].shape[1] for call in calls] == [1, 3, 1, 3, 1]
    torch.testing.assert_close(calls[0].kwargs["joint_seed"], start[:, None])
    torch.testing.assert_close(calls[1].kwargs["joint_seed"], poses[:, :3, 0, 3:4])
    torch.testing.assert_close(calls[2].kwargs["joint_seed"], poses[:, 3:4, 0, 3:4])
    torch.testing.assert_close(calls[3].kwargs["joint_seed"], poses[:, 4:7, 0, 3:4])
    torch.testing.assert_close(calls[4].kwargs["joint_seed"], poses[:, 7:8, 0, 3:4])
    assert all(call.kwargs.get("continuous", False) is False for call in calls)


def test_continuous_batch_ik_uses_capability_and_path_seed_across_chunks() -> None:
    generator = _mock_generator(controlled_dof=1)
    generator.robot.get_solver.return_value.supports_continuous_batch_ik = True
    poses = torch.eye(4).repeat(2, 9, 1, 1)
    poses[..., 0, 3] = torch.arange(1, 10)
    generator.robot.compute_batch_ik.side_effect = lambda *, pose, **kwargs: (
        torch.ones(pose.shape[:2], dtype=torch.bool),
        pose[..., 0, 3:4].clone(),
    )
    with patch("embodichain.lab.sim.motion.motion_generator._MAX_IK_BATCH_TARGETS", 8):
        success, positions = generator._solve_cartesian_waypoints(
            poses, torch.zeros(2, 1), "arm"
        )

    assert success.all()
    torch.testing.assert_close(positions, poses[..., 0, 3:4])
    calls = generator.robot.compute_batch_ik.call_args_list
    assert [call.kwargs["pose"].shape[1] for call in calls] == [4, 4, 1]
    assert all(call.kwargs["continuous"] is True for call in calls)
    assert all(call.kwargs["joint_seed"].shape == (2, 1) for call in calls)
    torch.testing.assert_close(calls[1].kwargs["joint_seed"], poses[:, 3, 0, 3:4])
    torch.testing.assert_close(calls[2].kwargs["joint_seed"], poses[:, 7, 0, 3:4])


@pytest.mark.parametrize("continuous", [False, True])
def test_batch_ik_failure_keeps_last_valid_seed_at_chunk_boundary(
    continuous: bool,
) -> None:
    generator = _mock_generator(controlled_dof=1)
    generator.robot.get_solver.return_value.supports_continuous_batch_ik = continuous
    poses = torch.eye(4).repeat(2, 8, 1, 1)
    poses[..., 0, 3] = torch.arange(1, 9)

    def solve(*, pose: torch.Tensor, joint_seed: torch.Tensor, **kwargs: object):
        assert torch.isfinite(joint_seed).all()
        qpos = pose[..., 0, 3:4].clone()
        valid = torch.ones(pose.shape[:2], dtype=torch.bool)
        valid[1] = qpos[1, :, 0] != 4.0
        qpos[1, ~valid[1]] = float("nan")
        return valid, qpos

    generator.robot.compute_batch_ik.side_effect = solve
    with patch("embodichain.lab.sim.motion.motion_generator._MAX_IK_BATCH_TARGETS", 8):
        success, positions = generator._solve_cartesian_waypoints(
            poses, torch.zeros(2, 1), "arm"
        )

    assert success.tolist() == [True, False]
    torch.testing.assert_close(positions[0], poses[0, :, 0, 3:4])
    torch.testing.assert_close(positions[1], torch.zeros_like(positions[1]))
    calls = generator.robot.compute_batch_ik.call_args_list
    boundary_seed = calls[1 if continuous else 2].kwargs["joint_seed"]
    torch.testing.assert_close(
        boundary_seed.reshape(2, 1), torch.tensor([[4.0], [3.0]])
    )


@pytest.mark.parametrize("malformed", ["success_shape", "qpos_shape", "nonbinary_mask"])
def test_batch_ik_rejects_malformed_solver_results(malformed: str) -> None:
    generator = _mock_generator(controlled_dof=1)
    generator.robot.compute_batch_ik.side_effect = None
    valid = torch.ones(2, 1, dtype=torch.bool)
    qpos = torch.zeros(2, 1, 1)
    error = ValueError
    match = "IK success"
    if malformed == "success_shape":
        valid = valid[:, 0]
    elif malformed == "qpos_shape":
        qpos = qpos[:, 0]
        match = "IK qpos"
    else:
        valid = torch.full((2, 1), 2, dtype=torch.int64)
        error = TypeError
        match = "binary integer"
    generator.robot.compute_batch_ik.return_value = valid, qpos
    with pytest.raises(error, match=match):
        generator._solve_cartesian_waypoints(
            torch.eye(4).repeat(2, 1, 1, 1), torch.zeros(2, 1), "arm"
        )
    generator.robot.compute_ik.assert_not_called()


@pytest.mark.parametrize("strategy", ["motion_gen", "ik_interp"])
def test_cartesian_ik_failure_holds_only_failed_environment(strategy: str) -> None:
    """A failed final target cannot turn into a successful truncated path."""
    generator = _trapezoidal_generator(robot=_CartesianRobot())
    goal = torch.eye(4).repeat(2, 1, 1)
    goal[:, 0, 3] = 1.0
    options = MotionGenOptions(
        strategy=strategy,
        start_qpos=torch.zeros(2, 1),
        control_part="arm",
        sample_count=20,
        interpolation_dt=0.05 if strategy == "ik_interp" else None,
        is_interpolate=strategy == "motion_gen",
        interpolate_position_step=0.25,
    )
    result = generator.generate([PlanState.from_xpos(goal)], options)
    assert result.success.tolist() == [True, False]
    torch.testing.assert_close(result.positions[0, -1], torch.ones(1))
    torch.testing.assert_close(
        result.positions[1], torch.zeros_like(result.positions[1])
    )
    torch.testing.assert_close(
        result.velocities[1], torch.zeros_like(result.velocities[1])
    )


@pytest.mark.parametrize("pose_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("entry_point", ["generate", "interpolate_trajectory"])
def test_cartesian_preparation_accepts_mixed_pose_and_fk_precision(
    pose_dtype: torch.dtype, entry_point: str
) -> None:
    """Cartesian input precision may differ from the robot's FK buffers."""
    generator = _trapezoidal_generator(robot=_CartesianRobot())
    poses = torch.eye(4, dtype=pose_dtype).repeat(2, 2, 1, 1)
    poses[:, 1, 0, 3] = 0.5
    original_poses = poses.clone()
    start = torch.zeros(2, 1, dtype=torch.float32)
    options = MotionGenOptions(
        start_qpos=start,
        control_part="arm",
        sample_count=20,
        is_interpolate=True,
        interpolate_position_step=0.25,
    )

    if entry_point == "generate":
        result = generator.generate([PlanState.from_xpos(poses[:, 1])], options)
        assert result.success.tolist() == [True, True]
        positions = result.positions
    else:
        positions, targets = generator.interpolate_trajectory(
            control_part="arm", xpos_list=poses, options=options
        )
        assert targets.dtype == pose_dtype

    assert positions.dtype == start.dtype
    torch.testing.assert_close(positions[:, 0], start)
    torch.testing.assert_close(positions[:, -1], torch.full((2, 1), 0.5))
    torch.testing.assert_close(poses, original_poses)


@pytest.mark.parametrize("continuous", [False, True])
def test_cartesian_preparation_preserves_fk_confirmed_singular_start(
    continuous: bool,
) -> None:
    """An observed start needs no new IK solution, even at a singularity."""
    generator = _trapezoidal_generator(robot=_CartesianRobot())
    generator.robot.get_solver = lambda name: SimpleNamespace(
        supports_continuous_batch_ik=continuous
    )
    solved_poses = []

    def solve(*, pose, joint_seed, **kwargs):
        solved_poses.append(pose.clone())
        qpos = pose[..., 0, 3:4].clone()
        # Model an analytic solver that fails only at the zero wrist pose.
        return qpos[..., 0] > 0.0, qpos

    generator.robot.compute_batch_ik = solve
    goal = torch.eye(4).repeat(2, 1, 1)
    goal[:, 0, 3] = 0.5
    start = torch.zeros(2, 1)
    result = generator.generate(
        [PlanState.from_xpos(goal)],
        MotionGenOptions(
            start_qpos=start,
            control_part="arm",
            sample_count=20,
            is_interpolate=True,
            interpolate_position_step=0.25,
        ),
    )

    assert result.success.tolist() == [True, True]
    assert solved_poses and all(
        bool((pose[..., 0, 3] > 0.0).all()) for pose in solved_poses
    )
    torch.testing.assert_close(result.positions[:, 0], start)
    torch.testing.assert_close(result.positions[:, -1], torch.full((2, 1), 0.5))
    torch.testing.assert_close(start, torch.zeros_like(start))


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("max_targets", [2, 512])
def test_cartesian_preparation_requires_ik_for_unconfirmed_start_rows(
    continuous: bool, max_targets: int
) -> None:
    """Reusing one row's known start cannot hide another row's IK failure."""
    generator = _trapezoidal_generator(robot=_CartesianRobot())
    generator.robot.get_solver = lambda name: SimpleNamespace(
        supports_continuous_batch_ik=continuous
    )

    def solve(*, pose, joint_seed, **kwargs):
        qpos = pose[..., 0, 3:4].clone()
        return qpos[..., 0] > 0.25, qpos

    generator.robot.compute_batch_ik = solve
    poses = torch.eye(4).repeat(2, 2, 1, 1)
    poses[1, 0, 0, 3] = 0.25  # This row's first pose differs from its seed FK.
    poses[:, 1, 0, 3] = 0.5
    with patch(
        "embodichain.lab.sim.motion.motion_generator._MAX_IK_BATCH_TARGETS", max_targets
    ):
        positions, _, success = generator._interpolate_waypoints(
            control_part="arm",
            xpos_list=poses,
            options=MotionGenOptions(
                start_qpos=torch.zeros(2, 1), interpolate_position_step=0.5
            ),
        )

    assert success.tolist() == [True, False]
    torch.testing.assert_close(positions[0, 0], torch.zeros(1))
    torch.testing.assert_close(positions[0, -1], torch.full((1,), 0.5))
    torch.testing.assert_close(positions[1], torch.zeros_like(positions[1]))


@pytest.mark.parametrize("continuous", [False, True])
def test_batch_ik_accepts_fk_confirmed_only_waypoint(continuous: bool) -> None:
    """A path containing only its observed start does not dispatch IK."""
    generator = _mock_generator(controlled_dof=1)
    generator.robot.get_solver.return_value.supports_continuous_batch_ik = continuous
    generator.robot.compute_batch_ik.side_effect = AssertionError(
        "An FK-confirmed start must not require IK"
    )
    start = torch.tensor([[0.2], [0.4]])
    poses = torch.eye(4).repeat(2, 1, 1, 1)
    poses[:, 0, 0, 3] = start[:, 0]
    success, positions = generator._solve_cartesian_waypoints(
        poses, start, "arm", known_start=torch.ones(2, dtype=torch.bool)
    )
    assert success.all()
    torch.testing.assert_close(positions, start[:, None, :])
    generator.robot.compute_batch_ik.assert_not_called()
    generator.robot.compute_ik.assert_not_called()


def test_motion_gen_rejects_fixed_cartesian_samples_instead_of_bypassing_limits() -> (
    None
):
    generator = _trapezoidal_generator(robot=_CartesianRobot())
    goals = [torch.eye(4).repeat(2, 1, 1) for _ in range(2)]
    for goal, position in zip(goals, (0.1, 0.2)):
        goal[:, 0, 3] = position
    with pytest.raises(ValueError, match="preserve_cartesian_samples"):
        generator.generate(
            [PlanState.from_xpos(goal) for goal in goals],
            MotionGenOptions(
                strategy="motion_gen",
                start_qpos=torch.zeros(2, 1),
                control_part="arm",
                sample_count=3,
                interpolation_dt=0.01,
                preserve_cartesian_samples=True,
                plan_opts=TrapezoidalPlanOptions(
                    constraints={"velocity": 0.1, "acceleration": 0.2, "jerk": 1.0},
                    sample_interval=3,
                ),
            ),
        )


@pytest.mark.parametrize(
    "unused_option",
    [
        {"plan_opts": TrapezoidalPlanOptions()},
        {"velocity_limit": 0.1},
        {"acceleration_limit": 0.2},
    ],
)
def test_ik_interp_rejects_unused_backend_constraints(unused_option: dict) -> None:
    generator = _trapezoidal_generator()
    with pytest.raises(ValueError, match="ik_interp"):
        generator.generate(
            [PlanState.from_qpos(torch.ones(1, 1))],
            MotionGenOptions(
                strategy="ik_interp",
                start_qpos=torch.zeros(1, 1),
                sample_count=3,
                interpolation_dt=0.01,
                **unused_option,
            ),
        )


def test_motion_gen_rejects_unsupported_joint_goals_without_fallback() -> None:
    generator = _mock_generator(supported_move_types=frozenset({MoveType.EEF_MOVE}))
    with pytest.raises(ValueError, match="JOINT_MOVE"):
        generator.generate(
            [PlanState.from_qpos(torch.ones(BATCH_SIZE, CONTROLLED_DOF))],
            MotionGenOptions(
                strategy="motion_gen",
                start_qpos=torch.zeros(BATCH_SIZE, CONTROLLED_DOF),
                sample_count=SAMPLE_COUNT,
                interpolation_dt=STEP_DT,
            ),
        )


@pytest.mark.parametrize("backend_control_part", [None, "other_arm"])
def test_resampled_collision_trajectory_is_checked_on_the_output_grid(
    backend_control_part: str | None,
) -> None:
    """New midpoint samples must not inherit an endpoint-only collision check."""
    generator = _mock_generator(
        controlled_dof=1,
        result=_timed_result(torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]])),
    )
    generator.planner.supports_joint_trajectory_validation = True
    obstacle_poses = {"obstacle": torch.eye(4).repeat(2, 1, 1)}
    options = PlanOptions()
    options.dynamic_obstacle_poses = obstacle_poses
    options.control_part = backend_control_part

    def validate(trajectory, *, control_part, obstacle_poses):
        assert control_part == (backend_control_part or "arm")
        assert torch.equal(obstacle_poses["obstacle"], torch.eye(4).repeat(2, 1, 1))
        assert trajectory.shape == (2, 3, 1)
        valid = torch.ones(2, 3, dtype=torch.bool)
        valid[1] = trajectory[1, :, 0] != 0.5
        return valid

    generator.planner.validate_joint_trajectory.side_effect = validate
    result = generator.generate(
        [PlanState.from_qpos(torch.ones(2, 1))],
        MotionGenOptions(
            start_qpos=torch.zeros(2, 1),
            control_part="arm",
            sample_count=3,
            plan_opts=options,
        ),
    )
    assert result.success.tolist() == [True, False]
    torch.testing.assert_close(result.positions[0, :, 0], torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(result.positions[1], torch.zeros(3, 1))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cartesian_preinterpolation_accepts_cpu_seed_on_gpu() -> None:
    generator = _trapezoidal_generator()
    generator.device = torch.device("cuda:0")
    generator.robot = _CartesianRobot()
    poses = torch.eye(4, device=generator.device).repeat(2, 2, 1, 1)
    poses[:, 1, 0, 3] = 1.0
    positions, _, success = generator._interpolate_waypoints(
        control_part="arm",
        xpos_list=poses,
        options=MotionGenOptions(
            start_qpos=torch.zeros(2, 1), interpolate_position_step=0.25
        ),
    )
    assert positions.device == generator.device
    assert success.tolist() == [True, False]
    torch.testing.assert_close(positions[0, -1], torch.ones(1, device=generator.device))
    torch.testing.assert_close(positions[1], torch.zeros_like(positions[1]))


@pytest.mark.parametrize("strategy", ["motion_gen", "ik_interp"])
def test_intermediate_ik_failure_stays_failed_after_reachable_goal(
    strategy: str,
) -> None:
    generator = _trapezoidal_generator(robot=_CartesianRobot())
    solved_positions: list[torch.Tensor] = []

    def solve(*, pose, joint_seed, **kwargs):
        qpos = pose[..., 0, 3:4].clone()
        solved_positions.append(qpos)
        success = torch.ones(pose.shape[:2], dtype=torch.bool)
        success[1] = ~torch.isclose(qpos[1, :, 0], torch.tensor(0.5))
        return success, qpos

    generator.robot.compute_batch_ik = solve
    goals = [torch.eye(4).repeat(2, 1, 1) for _ in range(2)]
    for pose, position in zip(goals, (0.5, 1.0)):
        pose[:, 0, 3] = position
    result = generator.generate(
        [PlanState.from_xpos(pose) for pose in goals],
        MotionGenOptions(
            strategy=strategy,
            start_qpos=torch.zeros(2, 1),
            control_part="arm",
            sample_count=20,
            interpolation_dt=0.05 if strategy == "ik_interp" else None,
            is_interpolate=strategy == "motion_gen",
            interpolate_position_step=0.25,
        ),
    )
    solved_positions = torch.cat(solved_positions, dim=1)
    assert (solved_positions[1, :, 0] == 0.5).any()
    assert (solved_positions[1, :, 0] == 1.0).any()
    assert result.success.tolist() == [True, False]
    torch.testing.assert_close(result.positions[0, -1], torch.ones(1))
    torch.testing.assert_close(
        result.positions[1], torch.zeros_like(result.positions[1])
    )


def test_ik_interp_rejects_ambiguous_linear_path_request() -> None:
    with pytest.raises(ValueError, match="preserve_cartesian_samples"):
        MotionGenOptions(strategy="ik_interp", is_linear=True)


def test_public_interpolation_rejects_failed_required_waypoints() -> None:
    generator = _trapezoidal_generator(robot=_CartesianRobot())
    poses = torch.eye(4).repeat(2, 2, 1, 1)
    poses[:, 1, 0, 3] = 1.0
    with pytest.raises(ValueError, match="required IK waypoints"):
        generator.interpolate_trajectory(
            control_part="arm",
            xpos_list=poses,
            options=MotionGenOptions(
                start_qpos=torch.zeros(2, 1), interpolate_position_step=0.25
            ),
        )


@pytest.mark.parametrize("strategy", ["motion_gen", "ik_interp"])
def test_nonfinite_ik_solution_fails_only_its_environment(strategy: str) -> None:
    generator = _trapezoidal_generator(robot=_CartesianRobot())
    original_ik = generator.robot.compute_batch_ik

    def nonfinite_ik(**kwargs):
        success, qpos = original_ik(**kwargs)
        success[:] = True
        qpos[1] = float("nan")
        return success, qpos

    generator.robot.compute_batch_ik = nonfinite_ik
    goal = torch.eye(4).repeat(2, 1, 1)
    goal[:, 0, 3] = 0.5
    result = generator.generate(
        [PlanState.from_xpos(goal)],
        MotionGenOptions(
            strategy=strategy,
            start_qpos=torch.zeros(2, 1),
            control_part="arm",
            sample_count=10,
            interpolation_dt=0.05 if strategy == "ik_interp" else None,
            is_interpolate=strategy == "motion_gen",
            interpolate_position_step=0.25,
        ),
    )
    assert result.success.tolist() == [True, False]
    assert torch.isfinite(result.positions).all()
    torch.testing.assert_close(
        result.positions[1], torch.zeros_like(result.positions[1])
    )


def test_mutated_options_cannot_restore_implicit_planner_bypass() -> None:
    generator = _mock_generator()
    options = MotionGenOptions(strategy="ik_interp", preserve_cartesian_samples=True)
    options.strategy = "motion_gen"
    with pytest.raises(ValueError, match="preserve_cartesian_samples"):
        generator.generate(
            [PlanState.from_xpos(torch.eye(4).repeat(BATCH_SIZE, 1, 1))], options
        )
