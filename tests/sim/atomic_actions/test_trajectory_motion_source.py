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
from embodichain.lab.sim.planners.utils import PlanState, MoveType
from embodichain.lab.sim.atomic_actions.core import ActionCfg


def _mock_mg(num_envs=2, arm_dof=6):
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
    return mg


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
