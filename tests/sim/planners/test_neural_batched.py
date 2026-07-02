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
