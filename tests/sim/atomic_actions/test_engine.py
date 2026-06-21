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

"""Tests for atomic action engine (registry, SemanticAnalyzer, AtomicActionEngine)."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock, Mock

from embodichain.lab.sim.atomic_actions.affordance import Affordance
from embodichain.lab.sim.atomic_actions.core import (
    ActionCfg,
    MoveObjectTarget,
    ObjectSemantics,
)
from embodichain.lab.sim.atomic_actions.engine import (
    AtomicActionEngine,
    SemanticAnalyzer,
    get_registered_actions,
    register_action,
    unregister_action,
)

# ---------------------------------------------------------------------------
# Global Action Registry
# ---------------------------------------------------------------------------


class TestGlobalRegistry:
    """Tests for register_action / unregister_action / get_registered_actions."""

    def teardown_method(self):
        # Clean up any test registrations
        unregister_action("_test_dummy")

    def test_register_and_retrieve(self):
        mock_cls = Mock()
        register_action("_test_dummy", mock_cls)
        registry = get_registered_actions()
        assert "_test_dummy" in registry
        assert registry["_test_dummy"] is mock_cls

    def test_unregister_removes_entry(self):
        register_action("_test_dummy", Mock())
        unregister_action("_test_dummy")
        assert "_test_dummy" not in get_registered_actions()

    def test_unregister_nonexistent_is_noop(self):
        # Should not raise
        unregister_action("_nonexistent_action")

    def test_get_registered_actions_returns_copy(self):
        """Mutating the returned dict should not affect the global registry."""
        result = get_registered_actions()
        result["_should_not_persist"] = Mock()
        assert "_should_not_persist" not in get_registered_actions()


# ---------------------------------------------------------------------------
# SemanticAnalyzer
# ---------------------------------------------------------------------------


class TestSemanticAnalyzer:
    """Tests for SemanticAnalyzer."""

    def setup_method(self):
        self.analyzer = SemanticAnalyzer()

    def test_analyze_returns_object_semantics(self):
        sem = self.analyzer.analyze("mug")
        assert isinstance(sem, ObjectSemantics)
        assert sem.label == "mug"
        assert isinstance(sem.affordance, Affordance)

    def test_analyze_caches_by_default(self):
        sem1 = self.analyzer.analyze("bottle")
        sem2 = self.analyzer.analyze("bottle")
        assert sem1 is sem2

    def test_analyze_bypasses_cache_with_geometry(self):
        sem1 = self.analyzer.analyze("bottle")
        sem2 = self.analyzer.analyze(
            "bottle", geometry={"bounding_box": [0.2, 0.2, 0.2]}
        )
        assert sem1 is not sem2

    def test_analyze_no_cache(self):
        sem1 = self.analyzer.analyze("cup", use_cache=False)
        sem2 = self.analyzer.analyze("cup", use_cache=False)
        assert sem1 is not sem2

    def test_clear_cache(self):
        self.analyzer.analyze("can")
        self.analyzer.clear_cache()
        # After clearing, a new object should be created
        sem1 = self.analyzer.analyze("can")
        sem2 = self.analyzer.analyze("can")
        assert sem1 is sem2  # re-cached after clear


# ---------------------------------------------------------------------------
# AtomicActionEngine._resolve_target
# ---------------------------------------------------------------------------


class TestResolveTarget:
    """Tests for AtomicActionEngine._resolve_target with various input types."""

    def setup_method(self):
        self.robot = Mock()
        self.robot.device = torch.device("cpu")
        self.robot.dof = 6
        self.robot.get_qpos.return_value = torch.zeros(1, 6)
        self.robot.get_joint_ids.return_value = list(range(6))

        self.mg = Mock()
        self.mg.robot = self.robot
        self.mg.device = torch.device("cpu")

        self.engine = AtomicActionEngine(self.mg, actions_cfg_list=[])

    def test_tensor_passthrough(self):
        tensor = torch.eye(4)
        result = self.engine._resolve_target(tensor)
        assert result is tensor

    def test_object_semantics_passthrough(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={})
        result = self.engine._resolve_target(sem)
        assert result is sem

    def test_move_object_target_passthrough(self):
        target = MoveObjectTarget(object_target_pose=torch.eye(4))
        result = self.engine._resolve_target(target)
        assert result is target

    def test_string_resolved_via_semantic_analyzer(self):
        result = self.engine._resolve_target("mug")
        assert isinstance(result, ObjectSemantics)
        assert result.label == "mug"

    def test_dict_with_pose_key(self):
        pose = torch.eye(4)
        result = self.engine._resolve_target({"pose": pose})
        assert result is pose

    def test_dict_with_pose_raises_on_non_tensor(self):
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            self.engine._resolve_target({"pose": "not_a_tensor"})

    def test_dict_with_object_target_pose(self):
        pose = torch.eye(4)
        result = self.engine._resolve_target({"object_target_pose": pose})
        assert isinstance(result, MoveObjectTarget)
        assert result.object_target_pose is pose

    def test_dict_with_object_target_pose_raises_on_non_tensor(self):
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            self.engine._resolve_target({"object_target_pose": "not_a_tensor"})

    def test_dict_with_semantics_key(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={}, label="bottle")
        result = self.engine._resolve_target({"semantics": sem})
        assert result is sem

    def test_dict_with_semantics_raises_on_wrong_type(self):
        with pytest.raises(TypeError, match="must be an ObjectSemantics"):
            self.engine._resolve_target({"semantics": "wrong"})

    def test_dict_with_label_uses_analyzer(self):
        result = self.engine._resolve_target({"label": "apple"})
        assert isinstance(result, ObjectSemantics)
        assert result.label == "apple"

    def test_dict_without_label_raises(self):
        with pytest.raises(ValueError, match="must provide 'label'"):
            self.engine._resolve_target({"geometry": {}})

    def test_dict_with_non_string_label_raises(self):
        with pytest.raises(TypeError, match="must be a string"):
            self.engine._resolve_target({"label": 123})

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="target must be"):
            self.engine._resolve_target(42)


# ---------------------------------------------------------------------------
# AtomicActionEngine held-object state contract
# ---------------------------------------------------------------------------


class TestHeldObjectStateContract:
    """Tests for explicit held-object state updates in execute_static."""

    def setup_method(self):
        self.robot = Mock()
        self.robot.device = torch.device("cpu")
        self.robot.dof = 6
        self.robot.get_qpos.return_value = torch.zeros(1, 6)
        self.robot.get_joint_ids.return_value = list(range(6))

        self.mg = Mock()
        self.mg.robot = self.robot
        self.mg.device = torch.device("cpu")

        self.engine = AtomicActionEngine(self.mg, actions_cfg_list=[])

    def _make_action(self, *, updates_held_object_state: bool, held_state):
        action = Mock()
        action.control_part = "arm"
        action.updates_held_object_state = updates_held_object_state
        action.execute.return_value = (
            True,
            torch.zeros(1, 2, 6),
            list(range(6)),
        )
        action.get_held_object_state.return_value = held_state
        return action

    def test_execute_static_clears_stale_context_before_running_actions(self):
        held_state = object()
        action = self._make_action(
            updates_held_object_state=False,
            held_state=None,
        )
        self.engine._actions = {"noop": action}
        self.engine._action_context["held_object_state"] = held_state

        is_success, _ = self.engine.execute_static([torch.eye(4)])

        assert is_success is True
        action.get_held_object_state.assert_not_called()
        assert "held_object_state" not in self.engine._action_context

    def test_execute_static_updates_state_when_action_declares_contract(self):
        held_state = object()
        action = self._make_action(
            updates_held_object_state=True,
            held_state=held_state,
        )
        self.engine._actions = {"producer": action}

        is_success, _ = self.engine.execute_static([torch.eye(4)])

        assert is_success is True
        action.get_held_object_state.assert_called_once_with()
        assert self.engine._action_context["held_object_state"] is held_state

    def test_execute_static_clears_state_when_action_returns_none(self):
        held_state = object()
        action = self._make_action(
            updates_held_object_state=True,
            held_state=None,
        )
        self.engine._actions = {"release": action}
        self.engine._action_context["held_object_state"] = held_state

        is_success, _ = self.engine.execute_static([torch.eye(4)])

        assert is_success is True
        action.get_held_object_state.assert_called_once_with()
        assert "held_object_state" not in self.engine._action_context

    def test_execute_static_keeps_state_within_single_action_sequence(self):
        held_state = object()
        producer = self._make_action(
            updates_held_object_state=True,
            held_state=held_state,
        )
        release = self._make_action(
            updates_held_object_state=True,
            held_state=None,
        )
        self.engine._actions = {"producer": producer, "release": release}

        is_success, _ = self.engine.execute_static([torch.eye(4), torch.eye(4)])

        assert is_success is True
        assert release.execute.call_args.kwargs["held_object_state"] is held_state
        assert "held_object_state" not in self.engine._action_context


if __name__ == "__main__":
    test = TestSemanticAnalyzer()
    test.setup_method()
    test.test_analyze_returns_object_semantics()
