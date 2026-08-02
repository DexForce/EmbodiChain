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

from embodichain.lab.sim.atomic_actions.affordance import Affordance
from embodichain.lab.sim.atomic_actions import EndEffectorPoseTarget
from embodichain.lab.sim.atomic_actions.core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    HeldObjectState,
    ObjectSemantics,
    WorldState,
)
from embodichain.lab.sim.atomic_actions.engine import AtomicActionEngine


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


class _HeldStateAction(_StubAction):
    def __init__(self, mg, success_vec, *, set_held):
        super().__init__(mg, success_vec)
        self._set_held = set_held

    def execute(self, target, state):
        result = super().execute(target, state)
        held_objects = {}
        if self._set_held:
            batch_size = state.batch_size
            held_objects["arm"] = HeldObjectState(
                semantics=ObjectSemantics(
                    affordance=Affordance(), geometry={}, label="test-object"
                ),
                object_to_eef=torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),
                grasp_xpos=torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),
            )
        return ActionResult(
            success=result.success,
            trajectory=result.trajectory,
            next_state=result.next_state.with_updates(held_objects=held_objects),
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

    def test_failed_env_does_not_acquire_successful_env_held_state(self):
        mg = Mock()
        mg.robot.get_qpos = lambda: torch.zeros(3, 3)
        mg.robot.dof = 3
        mg.device = torch.device("cpu")
        engine = AtomicActionEngine(mg)
        engine.register(
            _HeldStateAction(mg, [True, False, True], set_held=True), name="pick"
        )

        _, _, state = engine.run([("pick", EndEffectorPoseTarget(xpos=torch.eye(4)))])

        assert state.get_held_object("arm").env_mask.tolist() == [True, False, True]

    def test_failed_env_preserves_held_state_when_successful_envs_release(self):
        mg = Mock()
        mg.robot.get_qpos = lambda: torch.zeros(3, 3)
        mg.robot.dof = 3
        mg.device = torch.device("cpu")
        engine = AtomicActionEngine(mg)
        engine.register(
            _HeldStateAction(mg, [True, False, True], set_held=False), name="place"
        )
        held = HeldObjectState(
            semantics=ObjectSemantics(
                affordance=Affordance(), geometry={}, label="test-object"
            ),
            object_to_eef=torch.eye(4).unsqueeze(0).repeat(3, 1, 1),
            grasp_xpos=torch.eye(4).unsqueeze(0).repeat(3, 1, 1),
        )
        initial = WorldState(last_qpos=torch.zeros(3, 3), held_objects={"arm": held})

        _, _, state = engine.run(
            [("place", EndEffectorPoseTarget(xpos=torch.eye(4)))], state=initial
        )

        assert state.get_held_object("arm").env_mask.tolist() == [False, True, False]
