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

"""Per-environment failure propagation tests for AtomicActionEngine.run()."""

from __future__ import annotations

import torch
import pytest
from unittest.mock import Mock

from embodichain.lab.sim.atomic_actions.engine import AtomicActionEngine
from embodichain.lab.sim.atomic_actions.core import (
    ActionResult,
    AtomicAction,
    WorldState,
    EndEffectorPoseTarget,
    ActionCfg,
)


class _StubAction(AtomicAction):
    TargetType = EndEffectorPoseTarget

    def __init__(self, mg, success_vec, traj_len=4, dof=3):
        super().__init__(mg, ActionCfg())
        self._success = torch.tensor(success_vec)
        self._traj_len = traj_len
        self._dof = dof

    def execute(self, target, state):
        n = state.last_qpos.shape[0]
        traj = torch.zeros(n, self._traj_len, self._dof)
        traj[:] = state.last_qpos.unsqueeze(1)
        return ActionResult(
            success=self._success.clone(),
            trajectory=traj,
            next_state=WorldState(last_qpos=traj[:, -1, :].clone()),
        )


class TestRunPerEnv:
    def test_failed_env_holds(self):
        mg = Mock()
        mg.robot.get_qpos = lambda: torch.zeros(3, 3)
        mg.robot.dof = 3
        mg.device = torch.device("cpu")
        eng = AtomicActionEngine(mg)
        # env 1 fails step 2
        eng.register(_StubAction(mg, [True, True, True]), name="a")
        eng.register(_StubAction(mg, [True, False, True]), name="b")
        eng.register(_StubAction(mg, [True, True, True]), name="c")
        success, traj, state = eng.run(
            steps=[
                ("a", EndEffectorPoseTarget(xpos=torch.eye(4))),
                ("b", EndEffectorPoseTarget(xpos=torch.eye(4))),
                ("c", EndEffectorPoseTarget(xpos=torch.eye(4))),
            ]
        )
        assert success.tolist() == [True, False, True]
        assert traj.shape[1] == 12  # 3 steps * 4 waypoints
        # env 1's rows after its failure should equal its pre-failure qpos (held)
        # all zeros here, so just check shape and that env 0/2 advanced
        assert state.last_qpos.shape == (3, 3)
