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
from embodichain.lab.sim.planners.utils import PlanState, PlanResult, MoveType


def _mock_planner(b=3, n=15, dofs=6):
    planner = Mock()
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
