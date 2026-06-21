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

"""Tests for atomic_actions.engine."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import Mock

from embodichain.lab.sim.atomic_actions.affordance import Affordance
from embodichain.lab.sim.atomic_actions.core import (
    ActionResult,
    AtomicAction,
    GraspTarget,
    HeldObjectState,
    HeldObjectTarget,
    ObjectSemantics,
    PoseTarget,
    WorldState,
)
from embodichain.lab.sim.atomic_actions.engine import (
    AtomicActionEngine,
    get_registered_actions,
    register_action,
    unregister_action,
)

# ---------------------------------------------------------------------------
# Global registry (kept from old design)
# ---------------------------------------------------------------------------


class TestGlobalRegistry:
    def teardown_method(self):
        unregister_action("_test_dummy")

    def test_register_and_retrieve(self):
        cls = Mock()
        register_action("_test_dummy", cls)
        assert get_registered_actions()["_test_dummy"] is cls

    def test_unregister(self):
        register_action("_test_dummy", Mock())
        unregister_action("_test_dummy")
        assert "_test_dummy" not in get_registered_actions()

    def test_unregister_nonexistent_is_noop(self):
        unregister_action("_does_not_exist")

    def test_get_registered_actions_returns_copy(self):
        out = get_registered_actions()
        out["_should_not_persist"] = Mock()
        assert "_should_not_persist" not in get_registered_actions()


# ---------------------------------------------------------------------------
# Engine run() semantics
# ---------------------------------------------------------------------------


NUM_ENVS = 2
TOTAL_DOF = 8


def _make_mg():
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = TOTAL_DOF
    robot.get_qpos.return_value = torch.zeros(NUM_ENVS, TOTAL_DOF)

    mg = Mock()
    mg.robot = robot
    mg.device = torch.device("cpu")
    return mg


def _fake_action(name, target_type, *, sets_held=False, clears_held=False, fails=False):
    action = Mock(spec=AtomicAction)
    action.TargetType = target_type
    action.cfg = Mock()
    action.cfg.name = name

    def execute(target, state):
        if fails:
            return ActionResult(
                success=False,
                trajectory=torch.empty(NUM_ENVS, 0, TOTAL_DOF),
                next_state=state,
            )
        held = state.held_object
        if sets_held:
            sem = ObjectSemantics(affordance=Affordance(), geometry={}, label="x")
            held = HeldObjectState(
                semantics=sem,
                object_to_eef=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
                grasp_xpos=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
            )
        if clears_held:
            held = None
        traj = torch.zeros(NUM_ENVS, 5, TOTAL_DOF)
        return ActionResult(
            success=True,
            trajectory=traj,
            next_state=WorldState(
                last_qpos=traj[:, -1, :].clone(),
                held_object=held,
            ),
        )

    action.execute = Mock(side_effect=execute)
    return action


class TestEngineRun:
    def setup_method(self):
        self.mg = _make_mg()
        self.engine = AtomicActionEngine(self.mg)

    def test_register_and_lookup(self):
        action = _fake_action("pick_up", GraspTarget, sets_held=True)
        self.engine.register(action)
        assert "pick_up" in self.engine.actions

    def test_register_with_explicit_name_overrides_cfg(self):
        action = _fake_action("pick_up", GraspTarget, sets_held=True)
        self.engine.register(action, name="custom")
        assert "custom" in self.engine.actions

    def test_run_concatenates_trajectories(self):
        a = _fake_action("a", PoseTarget)
        b = _fake_action("b", PoseTarget)
        self.engine.register(a, name="a")
        self.engine.register(b, name="b")
        ok, traj, _ = self.engine.run(
            [("a", PoseTarget(torch.eye(4))), ("b", PoseTarget(torch.eye(4)))]
        )
        assert ok is True
        assert traj.shape == (NUM_ENVS, 10, TOTAL_DOF)

    def test_run_threads_world_state(self):
        pick = _fake_action("pick", GraspTarget, sets_held=True)
        move = _fake_action("move", HeldObjectTarget)
        place = _fake_action("place", PoseTarget, clears_held=True)
        self.engine.register(pick, name="pick")
        self.engine.register(move, name="move")
        self.engine.register(place, name="place")
        sem = ObjectSemantics(affordance=Affordance(), geometry={}, label="x")
        ok, _, final_state = self.engine.run(
            [
                ("pick", GraspTarget(sem)),
                ("move", HeldObjectTarget(torch.eye(4))),
                ("place", PoseTarget(torch.eye(4))),
            ]
        )
        assert ok is True
        # The move action saw a non-None held_object (set by pick).
        move_state_arg = move.execute.call_args_list[0].args[1]
        assert move_state_arg.held_object is not None
        # Final state cleared by place.
        assert final_state.held_object is None

    def test_run_stops_on_first_failure(self):
        a = _fake_action("a", PoseTarget)
        b = _fake_action("b", PoseTarget, fails=True)
        c = _fake_action("c", PoseTarget)
        self.engine.register(a, name="a")
        self.engine.register(b, name="b")
        self.engine.register(c, name="c")
        ok, traj, _ = self.engine.run(
            [
                ("a", PoseTarget(torch.eye(4))),
                ("b", PoseTarget(torch.eye(4))),
                ("c", PoseTarget(torch.eye(4))),
            ]
        )
        assert ok is False
        # `c` should not have been called.
        c.execute.assert_not_called()
        # We still get back the partial trajectory accumulated from `a`.
        assert traj.shape == (NUM_ENVS, 5, TOTAL_DOF)

    def test_run_raises_on_unknown_action_name(self):
        with pytest.raises(KeyError, match="ghost"):
            self.engine.run([("ghost", PoseTarget(torch.eye(4)))])

    def test_run_raises_on_target_type_mismatch(self):
        a = _fake_action("a", PoseTarget)
        self.engine.register(a, name="a")
        with pytest.raises(TypeError, match="target"):
            self.engine.run([("a", HeldObjectTarget(torch.eye(4)))])

    def test_run_seeds_state_from_robot_when_none_provided(self):
        a = _fake_action("a", PoseTarget)
        self.engine.register(a, name="a")
        self.engine.run([("a", PoseTarget(torch.eye(4)))])
        # First call's state argument
        state_arg = a.execute.call_args_list[0].args[1]
        assert state_arg.last_qpos.shape == (NUM_ENVS, TOTAL_DOF)
        assert state_arg.held_object is None
