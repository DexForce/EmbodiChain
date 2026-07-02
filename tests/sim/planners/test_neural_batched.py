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


class TestNeuralParseWaypoints:
    def test_parse_waypoints_batched(self):
        from embodichain.lab.sim.planners.neural_planner import NeuralPlanner
        from embodichain.lab.sim.planners.utils import PlanState, MoveType

        # Build a minimal planner shell by stubbing __init__
        planner = NeuralPlanner.__new__(NeuralPlanner)
        planner.device = torch.device("cpu")
        planner._num_waypoints = 4
        B = 3
        states = [
            PlanState.from_xpos(
                torch.eye(4).unsqueeze(0).repeat(B, 1, 1) * 1.0,
                move_type=MoveType.EEF_MOVE,
            )
            for _ in range(2)
        ]
        pos, quat, mask, k = planner._parse_waypoints(states)
        assert pos.shape == (B, 4, 3)
        assert quat.shape == (B, 4, 4)
        assert mask.shape == (B, 4)
        assert k == 2


class TestNeuralPlanBatched:
    def test_plan_returns_batched_success(self, monkeypatch):
        from embodichain.lab.sim.planners.neural_planner import (
            NeuralPlanner,
            NeuralPlanOptions,
        )
        from embodichain.lab.sim.planners.utils import PlanState, MoveType

        planner = NeuralPlanner.__new__(NeuralPlanner)
        planner.device = torch.device("cpu")
        planner._num_waypoints = 4
        planner._action_dim = 7
        planner._max_steps = 5
        planner._pos_eps = 1e9  # always reached
        planner._rot_eps = 1e9
        planner._intermediate_orientation = True
        planner._use_relative_obs = False
        planner._obs_dim = 57  # 7+7+4*3+4*4+4+4+7
        planner.cfg = type(
            "c",
            (),
            {
                "action_scale": 0.0,
                "dt": 0.01,
                "control_part": "arm",
                "num_arm_joints": 7,
            },
        )()

        # stub actor: returns zeros so qpos never changes but eps is huge -> reached
        planner._actor = lambda obs: torch.zeros(obs.shape[0], 7)
        planner._normalizer = type("n", (), {"normalize": lambda self, o: o})()

        # stub robot FK + limits
        class _Robot:
            num_instances = 3
            device = torch.device("cpu")

            def get_qpos(self, name=None):
                return torch.zeros(3, 7)

            def get_qpos_limits(self, name=None):
                return (torch.zeros(7, 2),)

            def compute_fk(self, qpos=None, name=None, to_matrix=True):
                m = torch.eye(4).unsqueeze(0).repeat(qpos.shape[0], 1, 1)
                return (
                    m
                    if to_matrix
                    else torch.cat([m[:, :3, 3], torch.zeros(qpos.shape[0], 4)], dim=-1)
                )

        planner.robot = _Robot()

        B = 3
        states = [
            PlanState.from_xpos(
                torch.eye(4).unsqueeze(0).repeat(B, 1, 1), move_type=MoveType.EEF_MOVE
            )
            for _ in range(2)
        ]
        r = planner.plan(states, NeuralPlanOptions(control_part="arm", max_steps=3))
        assert r.success.shape == (B,)
        assert r.success.all().item()
        assert r.positions.shape[0] == B
