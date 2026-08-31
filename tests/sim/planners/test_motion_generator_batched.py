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

from typing import Literal
from unittest.mock import Mock, patch

import pytest
import torch

from embodichain.lab.sim.planners.base_planner import CollisionWorldInfo, PlanOptions
from embodichain.lab.sim.planners.motion_generator import (
    MotionGenerator,
    MotionGenOptions,
)
from embodichain.lab.sim.planners.utils import PlanState, PlanResult, MoveType

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


@pytest.fixture(autouse=True)
def _torch_resampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use deterministic Torch resampling without initializing Warp."""

    def resample(
        trajectory: torch.Tensor,
        interp_num: int,
        device: torch.device,
    ) -> torch.Tensor:
        indices = torch.linspace(
            0,
            trajectory.shape[1] - 1,
            interp_num,
            device=device,
        )
        lower = indices.floor().to(torch.long)
        upper = indices.ceil().to(torch.long)
        weights = (indices - lower).view(1, -1, 1)
        return torch.lerp(trajectory[:, lower], trajectory[:, upper], weights)

    monkeypatch.setattr(
        "embodichain.lab.sim.planners.motion_generator.resample_with_distance",
        resample,
    )


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


def test_direct_cartesian_planner_requires_joint_fallback_inputs():
    planner = _DirectCartesianPlanner()
    generator = object.__new__(MotionGenerator)
    generator.planner = planner
    generator.device = torch.device("cpu")

    with pytest.raises(ValueError, match="start_qpos"):
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
    robot.compute_ik.return_value = (
        torch.ones(batch_size, dtype=torch.bool),
        torch.zeros(batch_size, controlled_dof),
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
        mg.interpolate_trajectory = Mock(return_value=(interpolated_qpos, None))

        mg.generate(
            [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
            MotionGenOptions(is_interpolate=True, plan_opts=PlanOptions()),
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
        generator.robot.compute_ik.return_value = (
            torch.tensor([1, 0], dtype=torch.int64),
            torch.ones(BATCH_SIZE, CONTROLLED_DOF),
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
        ) -> tuple[torch.Tensor, torch.Tensor]:
            qpos = joint_seed.clone()
            qpos[:, :3] = pose[:, :3, 3]
            return torch.ones(BATCH_SIZE, dtype=torch.bool), qpos

        generator.robot.compute_ik.side_effect = encode_position
        weights = torch.linspace(1.0 / (SAMPLE_COUNT - 1), 1.0, SAMPLE_COUNT - 1)
        targets = []
        for weight in weights:
            pose = torch.eye(4).repeat(BATCH_SIZE, 1, 1)
            pose[:, 0, 3] = weight
            targets.append(PlanState.from_xpos(pose))

        result = generator.generate(
            targets,
            MotionGenOptions(
                strategy="motion_gen",
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
        assert generator.robot.compute_ik.call_count == SAMPLE_COUNT - 1
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

    def test_joint_target_falls_back_when_backend_has_no_joint_capability(self):
        generator = _mock_generator(supported_move_types=frozenset({MoveType.EEF_MOVE}))
        start = torch.zeros(BATCH_SIZE, CONTROLLED_DOF)
        target = torch.ones(BATCH_SIZE, CONTROLLED_DOF)

        result = generator.generate(
            [PlanState.from_qpos(target)],
            MotionGenOptions(
                strategy="motion_gen",
                sample_count=SAMPLE_COUNT,
                start_qpos=start,
                control_part="arm",
                interpolation_dt=STEP_DT,
            ),
        )

        assert result.positions is not None
        assert result.positions.shape == (
            BATCH_SIZE,
            SAMPLE_COUNT,
            CONTROLLED_DOF,
        )
        assert torch.allclose(result.positions[:, 0], start)
        assert torch.allclose(result.positions[:, -1], target)
        generator.planner.plan.assert_not_called()

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
