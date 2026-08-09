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

import torch
import pytest
from unittest.mock import Mock, patch

from embodichain.lab.sim.planners.motion_generator import (
    MotionGenerator,
    MotionGenOptions,
)
from embodichain.lab.sim.planners.base_planner import PlanOptions
from embodichain.lab.sim.planners.utils import PlanState, PlanResult, MoveType


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
        return PlanResult(
            success=torch.tensor([True]),
            positions=torch.zeros(1, 3, 2),
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


def test_bind_collision_world_copies_caller_options() -> None:
    planner = Mock()
    planner.supports_collision_world_updates = True
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


def test_bind_collision_world_rejects_unsupported_planner() -> None:
    planner = Mock()
    planner.supports_collision_world_updates = False
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
    planner.supports_collision_world_updates = True
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
    planner.supported_move_types = frozenset({MoveType.JOINT_MOVE})
    planner.supports_move_type.side_effect = (
        lambda move_type: move_type in planner.supported_move_types
    )
    planner.robot.num_instances = b
    planner.robot.device = torch.device("cpu")
    planner.plan.return_value = PlanResult(
        success=torch.ones(b, dtype=torch.bool),
        positions=torch.zeros(b, n, dofs),
    )
    planner.default_plan_options.return_value = None
    return planner


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
