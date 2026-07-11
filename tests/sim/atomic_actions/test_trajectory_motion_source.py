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

"""Tests for TrajectoryBuilder motion_source branching."""

from __future__ import annotations

import torch
import pytest
from unittest.mock import Mock

from embodichain.lab.sim.atomic_actions.trajectory import TrajectoryBuilder
from embodichain.lab.sim.planners.utils import PlanState, PlanResult, MoveType
from embodichain.lab.sim.atomic_actions.core import ActionCfg


def _mock_mg(num_envs=2, arm_dof=6, planner_type="toppra"):
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = arm_dof
    robot.get_qpos = lambda name=None: torch.zeros(num_envs, arm_dof)

    def compute_ik(pose=None, name=None, joint_seed=None, **kw):
        return torch.ones(num_envs, dtype=torch.bool), torch.zeros(num_envs, arm_dof)

    robot.compute_ik = compute_ik
    mg = Mock()
    mg.robot = robot
    mg.device = torch.device("cpu")
    planner = Mock()
    planner.cfg.planner_type = planner_type
    # TOPPRA allows MotionGenerator pre-interpolation; neural/curobo do not.
    planner.preinterpolate_targets = planner_type == "toppra"
    planner.preserve_plan_samples = planner_type == "curobo"
    mg.planner = planner
    return mg


def _mock_curobo_motion_generator(result_positions, success=None):
    """Fake MotionGenerator whose planner is a cuRobo backend."""
    num_envs = result_positions.shape[0]
    arm_dof = result_positions.shape[-1]
    mg = _mock_mg(num_envs=num_envs, arm_dof=arm_dof, planner_type="curobo")
    if success is None:
        success = torch.ones(num_envs, dtype=torch.bool)
    mg.generate.return_value = PlanResult(success=success, positions=result_positions)
    return mg


def _pose_targets_for_two_envs():
    return [
        [PlanState(xpos=torch.eye(4), move_type=MoveType.EEF_MOVE)] for _ in range(2)
    ]


class TestPlanArmTrajMotionGen:
    def test_motion_gen_path_delegates_to_generate(self):
        mg = _mock_mg(num_envs=3, arm_dof=6)
        from embodichain.lab.sim.planners.utils import PlanResult

        mg.generate.return_value = PlanResult(
            success=torch.ones(3, dtype=torch.bool),
            positions=torch.zeros(3, 12, 6),
        )
        builder = TrajectoryBuilder(mg)
        cfg = ActionCfg(
            motion_source="motion_gen", planner_type="toppra", control_part="arm"
        )
        start_qpos = torch.zeros(3, 6)
        # per-env list[list[PlanState]] with single-env PlanStates (action contract)
        target_states_list = [
            [
                PlanState(xpos=torch.eye(4), move_type=MoveType.EEF_MOVE),
                PlanState(xpos=torch.eye(4), move_type=MoveType.EEF_MOVE),
            ]
            for _ in range(3)
        ]
        ok, traj = builder.plan_arm_traj(
            target_states_list,
            start_qpos,
            12,
            control_part="arm",
            arm_dof=6,
            cfg=cfg,
        )
        assert ok.shape == (3,)
        assert ok.all().item()
        assert traj.shape == (3, 12, 6)
        mg.generate.assert_called_once()

    def test_ik_interp_path_unchanged(self):
        mg = _mock_mg(num_envs=2, arm_dof=6)
        builder = TrajectoryBuilder(mg)
        cfg = ActionCfg(motion_source="ik_interp", control_part="arm")
        start_qpos = torch.zeros(2, 6)
        target_states_list = [
            [PlanState(xpos=torch.eye(4), move_type=MoveType.EEF_MOVE)]
            for _ in range(2)
        ]
        ok, traj = builder.plan_arm_traj(
            target_states_list,
            start_qpos,
            10,
            control_part="arm",
            arm_dof=6,
            cfg=cfg,
        )
        assert ok.all().item()
        assert traj.shape[0] == 2


class TestCuroboBuilderDispatch:
    def test_curobo_builder_preserves_cartesian_targets_and_samples(self):
        mg = _mock_curobo_motion_generator(result_positions=torch.zeros(2, 7, 6))
        builder = TrajectoryBuilder(mg)
        success, trajectory = builder.plan_arm_traj(
            _pose_targets_for_two_envs(),
            torch.zeros(2, 6),
            n_waypoints=20,
            control_part="arm",
            arm_dof=6,
            cfg=ActionCfg(
                motion_source="motion_gen",
                planner_type="curobo",
                control_part="arm",
            ),
        )
        assert success.tolist() == [True, True]
        # preserve_plan_samples -> returned length is the planner's (7), not 20.
        assert trajectory.shape == (2, 7, 6)
        # No pre-interpolation; original EEF target reaches the generator.
        assert mg.generate.call_args.kwargs["options"].is_interpolate is False
        assert mg.generate.call_args.args[0][0].move_type is MoveType.EEF_MOVE

    def test_mismatched_planner_type_raises(self):
        # MotionGenerator owns toppra, action requests curobo.
        mg = _mock_mg(num_envs=2, arm_dof=6, planner_type="toppra")
        builder = TrajectoryBuilder(mg)
        with pytest.raises(ValueError, match="planner_type"):
            builder.plan_arm_traj(
                _pose_targets_for_two_envs(),
                torch.zeros(2, 6),
                n_waypoints=10,
                control_part="arm",
                arm_dof=6,
                cfg=ActionCfg(
                    motion_source="motion_gen",
                    planner_type="curobo",
                    control_part="arm",
                ),
            )

    def test_invalid_motion_source_raises(self):
        with pytest.raises(ValueError, match="motion_source"):
            ActionCfg(motion_source="bogus")

    def test_motion_gen_without_planner_type_raises(self):
        with pytest.raises(ValueError, match="planner_type is required"):
            ActionCfg(motion_source="motion_gen")

    def test_ik_interp_with_planner_type_raises(self):
        with pytest.raises(ValueError, match="planner_type is only valid"):
            ActionCfg(motion_source="ik_interp", planner_type="toppra")

    def test_nan_positions_rejected(self):
        positions = torch.zeros(2, 5, 6)
        positions[0, 0, 0] = float("nan")
        mg = _mock_curobo_motion_generator(result_positions=positions)
        builder = TrajectoryBuilder(mg)
        with pytest.raises(ValueError, match="non-finite"):
            builder.plan_arm_traj(
                _pose_targets_for_two_envs(),
                torch.zeros(2, 6),
                n_waypoints=10,
                control_part="arm",
                arm_dof=6,
                cfg=ActionCfg(
                    motion_source="motion_gen",
                    planner_type="curobo",
                    control_part="arm",
                ),
            )

    def test_none_positions_rejected(self):
        mg = _mock_curobo_motion_generator(result_positions=torch.zeros(2, 5, 6))
        mg.generate.return_value = PlanResult(
            success=torch.ones(2, dtype=torch.bool), positions=None
        )
        builder = TrajectoryBuilder(mg)
        with pytest.raises(ValueError, match="positions"):
            builder.plan_arm_traj(
                _pose_targets_for_two_envs(),
                torch.zeros(2, 6),
                n_waypoints=10,
                control_part="arm",
                arm_dof=6,
                cfg=ActionCfg(
                    motion_source="motion_gen",
                    planner_type="curobo",
                    control_part="arm",
                ),
            )

    def test_failed_row_holds_start_qpos(self):
        positions = torch.zeros(2, 5, 6)
        positions[1] = 1.0  # env 1 "succeeds" numerically but we mark it failed
        mg = _mock_curobo_motion_generator(
            result_positions=positions,
            success=torch.tensor([True, False]),
        )
        builder = TrajectoryBuilder(mg)
        start = torch.zeros(2, 6)
        start[1] = 0.5
        success, trajectory = builder.plan_arm_traj(
            _pose_targets_for_two_envs(),
            start,
            n_waypoints=10,
            control_part="arm",
            arm_dof=6,
            cfg=ActionCfg(
                motion_source="motion_gen",
                planner_type="curobo",
                control_part="arm",
            ),
        )
        assert success.tolist() == [True, False]
        # Failed env held at its start qpos across all samples.
        assert torch.allclose(trajectory[1], start[1].unsqueeze(0).repeat(5, 1))
