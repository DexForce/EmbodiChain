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

import pytest
import torch

from embodichain.lab.gym.envs.managers import (
    ActionManager,
    ActionTerm,
    DeltaQposTerm,
    QposTerm,
    QposNormalizedTerm,
    QvelTerm,
    QfTerm,
)
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg


class MockEnv:
    """Minimal mock env for ActionTerm tests."""

    def __init__(self, num_envs: int = 4, action_dim: int = 6):
        self.num_envs = num_envs
        self.active_joint_ids = list(range(action_dim))
        self.device = torch.device("cpu")

    def get_qpos(self):
        return torch.zeros(self.num_envs, len(self.active_joint_ids), device=self.device)

    @property
    def robot(self):
        """DeltaQposTerm uses env.robot.get_qpos()."""
        return self


class MockEnvWithLimits(MockEnv):
    """Mock env with qpos_limits for QposNormalizedTerm."""

    def __init__(self, num_envs: int = 4, action_dim: int = 6):
        super().__init__(num_envs, action_dim)
        # qpos_limits shape: (1, dof, 2) for [low, high]
        self._qpos_limits = torch.zeros(1, action_dim, 2)
        self._qpos_limits[..., 0] = -1.0
        self._qpos_limits[..., 1] = 1.0

    @property
    def robot(self):
        return self

    @property
    def body_data(self):
        class BodyData:
            def __init__(_, limits):
                _.qpos_limits = limits

        return BodyData(self._qpos_limits)


def test_delta_qpos_term_process_action():
    """DeltaQposTerm: qpos = current_qpos + scale * action."""
    env = MockEnv(num_envs=4, action_dim=6)
    cfg = ActionTermCfg(func=DeltaQposTerm, params={"scale": 0.1})
    term = DeltaQposTerm(cfg, env)

    action = torch.ones(4, 6) * 2.0
    result = term.process_action(action)

    assert "qpos" in result
    expected = env.get_qpos() + 0.1 * action
    torch.testing.assert_close(result["qpos"], expected)
    assert term.action_dim == 6


def test_qpos_term_process_action():
    """QposTerm: qpos = scale * action."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = ActionTermCfg(func=QposTerm, params={"scale": 0.5})
    term = QposTerm(cfg, env)

    action = torch.ones(2, 3)
    result = term.process_action(action)

    assert "qpos" in result
    torch.testing.assert_close(result["qpos"], torch.ones(2, 3) * 0.5)
    assert term.action_dim == 3


def test_qpos_normalized_term_process_action():
    """QposNormalizedTerm: [-1,1] -> [low, high] with scale=1."""
    env = MockEnvWithLimits(num_envs=2, action_dim=2)
    cfg = ActionTermCfg(func=QposNormalizedTerm, params={"scale": 1.0})
    term = QposNormalizedTerm(cfg, env)

    # action=-1 -> low, action=1 -> high
    action = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    result = term.process_action(action)

    assert "qpos" in result
    # low=-1, high=1: (-1+1)*0.5*(1-(-1)) = 0 for action=-1; (1+1)*0.5*2 = 2 for action=1
    expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    torch.testing.assert_close(result["qpos"], expected)
    assert term.action_dim == 2


def test_qvel_term_process_action():
    """QvelTerm: qvel = scale * action."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = ActionTermCfg(func=QvelTerm, params={"scale": 0.2})
    term = QvelTerm(cfg, env)

    action = torch.ones(2, 3)
    result = term.process_action(action)

    assert "qvel" in result
    torch.testing.assert_close(result["qvel"], torch.ones(2, 3) * 0.2)


def test_qf_term_process_action():
    """QfTerm: qf = scale * action."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = ActionTermCfg(func=QfTerm, params={"scale": 10.0})
    term = QfTerm(cfg, env)

    action = torch.ones(2, 3)
    result = term.process_action(action)

    assert "qf" in result
    torch.testing.assert_close(result["qf"], torch.ones(2, 3) * 10.0)


def test_action_manager_tensor_input():
    """ActionManager passes tensor to first (active) term."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = {
        "delta_qpos": ActionTermCfg(func=DeltaQposTerm, params={"scale": 0.1}),
    }
    manager = ActionManager(cfg, env)

    action = torch.ones(2, 3)
    result = manager.process_action(action)

    assert "qpos" in result
    expected = env.get_qpos() + 0.1 * action
    torch.testing.assert_close(result["qpos"], expected)
    assert manager.action_type == "delta_qpos"
    assert manager.total_action_dim == 3


def test_action_manager_dict_input():
    """ActionManager uses key to select term for dict input."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = {
        "delta_qpos": ActionTermCfg(func=DeltaQposTerm, params={"scale": 0.1}),
        "qpos": ActionTermCfg(func=QposTerm, params={"scale": 1.0}),
    }
    manager = ActionManager(cfg, env)

    action_dict = {"qpos": torch.ones(2, 3) * 0.5}
    result = manager.process_action(action_dict)

    assert "qpos" in result
    torch.testing.assert_close(result["qpos"], torch.ones(2, 3) * 0.5)


def test_action_manager_invalid_dict_raises():
    """ActionManager raises when dict has no matching term key."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = {"delta_qpos": ActionTermCfg(func=DeltaQposTerm, params={"scale": 0.1})}
    manager = ActionManager(cfg, env)

    with torch.no_grad():
        with pytest.raises(ValueError, match="No valid action keys"):
            manager.process_action({"unknown_key": torch.ones(2, 3)})
